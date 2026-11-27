//! Global Descriptor Table (GDT) and Task State Segment (TSS).
//!
//! The GDT holds the kernel code segment, a kernel data segment, and
//! the TSS descriptor. The TSS holds a dedicated interrupt stack for
//! the double-fault handler: when the CPU double-faults (e.g. a stack
//! overflow in an exception handler), it switches to this known-good
//! stack via the Interrupt Stack Table (IST), so the handler can run
//! even when the normal stack is broken.
//!
//! The data segment is not optional: `iretq` at the end of an
//! interrupt handler pops SS from the interrupt frame and reloads its
//! descriptor from the GDT. The bootloader leaves SS holding selector
//! 0x10 (index 2), which in the bootloader's GDT was a data segment
//! — but in a GDT without a data segment, index 2 is the TSS
//! descriptor, which can never be a stack segment, so the first
//! `iretq` faults with #GP(0x10). This is exactly the 2026-11-21
//! #GP at the end of `timer_interrupt_handler`: the timer IRQ
//! delivered fine, EOI went out, and the handler's `iretq` faulted
//! because there was no data segment for SS to reload. Normal
//! execution never notices (it runs off the cached SS descriptor),
//! which is why only the interrupt return faults.
//!
//! See `crate::interrupts` for the IST index wiring.

use lazy_static::lazy_static;
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
use x86_64::structures::tss::TaskStateSegment;
use x86_64::VirtAddr;

/// IST index used by the double-fault handler (entry 0 of the IST).
pub const DOUBLE_FAULT_IST_INDEX: u16 = 0;

/// Size of the dedicated double-fault stack, in bytes.
pub const DOUBLE_FAULT_STACK_SIZE: usize = 4096 * 5;

lazy_static! {
    static ref TSS: TaskStateSegment = {
        let mut tss = TaskStateSegment::new();
        tss.interrupt_stack_table[DOUBLE_FAULT_IST_INDEX as usize] = {
            // A function-local `static mut`: lives for the whole run,
            // but is only touched here, once, at first use.
            static mut STACK: [u8; DOUBLE_FAULT_STACK_SIZE] = [0; DOUBLE_FAULT_STACK_SIZE];
            let stack_start = VirtAddr::from_ptr(&raw const STACK);
            stack_start + DOUBLE_FAULT_STACK_SIZE as u64
        };
        tss
    };
}

/// Set the kernel stack the CPU switches to on a privilege change
/// (interrupt/syscall from ring 3): `TSS.rsp[0]`.
///
/// The CPU loads `rsp` from `TSS.rsp[0]` whenever it takes an
/// interrupt or trap from a lower privilege level — it is the ONLY
/// stack a ring-3 task's interrupts and syscalls can land on, so it
/// must always point at the *currently running* task's kernel
/// stack top. The kernel updates it on every switch (see
/// `crate::interrupts::switch_to_task` and friends) *before* the
/// register switch, while interrupts are off: a timer IRQ landing
/// after the switch but before the update would push its frame onto
/// the previous task's kernel stack and corrupt it.
///
/// Soundness of the raw-pointer write: the TSS is a `lazy_static`
/// at a fixed address for the whole run (the GDT's TSS descriptor
/// points at it), this is single-CPU, and every caller runs with
/// interrupts disabled, so no interrupt can read the TSS
/// concurrently with the write — and the CPU only reads `rsp[0]`
/// on a privilege change, which cannot happen while interrupts are
/// off and the CPU is in ring 0.
pub fn set_kernel_stack_top(top: VirtAddr) {
    let tss = &raw const *TSS as *mut TaskStateSegment;
    unsafe {
        (*tss).privilege_stack_table[0] = top;
    }
}

/// The `TSS.rsp[0]` value right now, for tests and diagnostics.
/// (Only meaningful after [`init`] has run; before that it is 0 —
/// the TSS starts zeroed, which is exactly why a ring-3 entry
/// before the first `set_kernel_stack_top` would fault.)
pub fn kernel_stack_top() -> VirtAddr {
    TSS.privilege_stack_table[0]
}

// GDT layout (indices after the null descriptor at 0):
// * code segment — index 1, selector 0x08
// * data segment — index 2, selector 0x10
// * TSS descriptor — index 3 (occupies entries 3–4), selector 0x18
// * user DATA segment — index 5, selector 0x28 (0x2B with RPL 3)
// * user CODE segment — index 6, selector 0x30 (0x33 with RPL 3)
//
// The user segments are data-first, code-second ON PURPOSE, and
// the order is load-bearing for the `syscall`/`sysret` entry
// (Batch 11): `sysret` derives its selectors from the STAR MSR as
// CS = base + 16, SS = base + 8, so the 64-bit user code segment
// must sit exactly one GDT entry AFTER the user data segment
// (base 0x23: CS = 0x23 + 16 = 0x33, SS = 0x23 + 8 = 0x2B —
// the same layout Linux uses: DEFAULT_USER_DS then
// DEFAULT_USER_CS). With code-before-data (the Batch 8 order) no
// STAR value can produce a valid (CS, SS) pair at all, so `sysret`
// could never return to ring 3. The `iretq` entry path does not
// care about the order (it takes explicit selectors — see
// `USER_CODE_INDEX`/`USER_DATA_INDEX` and
// `rustos_scheduler::Ring3Frame`), which is why the old order
// worked until the MSR entry arrived.
lazy_static! {
    static ref GDT: (
        GlobalDescriptorTable,
        SegmentSelector,
        SegmentSelector,
        SegmentSelector,
        SegmentSelector,
        SegmentSelector,
    ) = {
        let mut gdt = GlobalDescriptorTable::new();
        let code_selector = gdt.append(Descriptor::kernel_code_segment());
        let data_selector = gdt.append(Descriptor::kernel_data_segment());
        let tss_selector = gdt.append(Descriptor::tss_segment(&TSS));
        // User segments, data FIRST then code (see the layout note
        // above): the `sysret` STAR encoding needs code exactly one
        // entry after data.
        let user_data_selector = gdt.append(Descriptor::user_data_segment());
        let user_code_selector = gdt.append(Descriptor::user_code_segment());
        (
            gdt,
            code_selector,
            data_selector,
            tss_selector,
            user_data_selector,
            user_code_selector,
        )
    };
}

/// Load the GDT, switch to the kernel code segment, reload the data
/// segment registers with the kernel data segment, and load the TSS.
///
/// Returns the kernel code segment selector. IDT entries must be built
/// with this explicit selector (see `crate::interrupts::build_idt`):
/// the x86_64 crate's `set_handler_fn` fills each gate's code selector
/// from the live `CS` at build time, and an explicit selector removes
/// that variable from the table entirely.
///
/// The SS/DS/ES reload is load-bearing: the bootloader leaves those
/// registers pointing at selectors in *its* GDT, and SS still holds
/// 0x10 when the first interrupt is delivered. `iretq` reloads SS
/// from the kernel's GDT, so index 2 must be a real data segment —
/// without it the first interrupt return #GPs (2026-11-21, error
/// code 0x10, RIP = the handler's `iretq`).
///
/// Must be called once, early in `kernel_main`, before interrupts are
/// enabled or exceptions are expected.
pub fn init() -> SegmentSelector {
    use x86_64::instructions::segmentation::{Segment, CS, DS, ES, SS};
    use x86_64::instructions::tables::load_tss;

    GDT.0.load();
    unsafe {
        CS::set_reg(GDT.1);
        SS::set_reg(GDT.2);
        DS::set_reg(GDT.2);
        ES::set_reg(GDT.2);
        load_tss(GDT.3);
    }
    GDT.1
}

/// The kernel code segment selector: the first GDT entry after the
/// null descriptor (index 1, value 0x08). Safe to call any time; the
/// GDT table is built lazily on first use.
pub fn code_selector() -> SegmentSelector {
    GDT.1
}

/// The kernel data segment selector (index 2, value 0x10). This is
/// what SS/DS/ES are reloaded with in [`init`].
pub fn data_selector() -> SegmentSelector {
    GDT.2
}

/// The TSS segment selector (index 3, value 0x18; the descriptor
/// occupies GDT entries 3–4).
pub fn tss_selector() -> SegmentSelector {
    GDT.3
}

/// The user code segment selector (index 6). `append` sets the
/// selector's RPL from the descriptor's DPL, so this already reads
/// 0x33 (index 6, RPL 3) — exactly the value that goes in a ring-3
/// `iretq` frame's CS slot, and the value `sysret` derives from the
/// STAR base (base + 16). DPL 3, 64-bit code.
pub fn user_code_selector() -> SegmentSelector {
    GDT.5
}

/// The user data segment selector (index 5). Like the code
/// selector, this already reads 0x2B (index 5, RPL 3) — the value
/// that goes in a ring-3 `iretq` frame's SS slot, and the value
/// `sysret` derives from the STAR base (base + 8). DPL 3, writable
/// data.
pub fn user_data_selector() -> SegmentSelector {
    GDT.4
}

/// GDT index of the user code segment (6). [`Ring3Frame::new`] takes
/// indices, not selectors — see `rustos_scheduler::Ring3Frame`.
pub const USER_CODE_INDEX: u16 = 6;

/// GDT index of the user data segment (5).
pub const USER_DATA_INDEX: u16 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdt_layout_selectors() {
        assert_eq!(code_selector().index(), 1);
        assert_eq!(data_selector().index(), 2);
        assert_eq!(tss_selector().index(), 3);
        // User segments are data-first (index 5), code-second
        // (index 6) — the `sysret` STAR encoding needs code exactly
        // one entry after data (Batch 11).
        assert_eq!(user_data_selector().index(), 5);
        assert_eq!(user_code_selector().index(), 6);
        // Values are index << 3, RPL 0 (kernel, Ring 0).
        assert_eq!(code_selector().0, 0x08);
        assert_eq!(data_selector().0, 0x10);
        assert_eq!(tss_selector().0, 0x18);
        // The user selectors already carry RPL 3 (`append` sets the
        // RPL from the descriptor's DPL): 0x33 for CS, 0x2B for SS —
        // the values a ring-3 iretq frame carries, and the values
        // `sysret` derives from the STAR base (0x23 + 16 / + 8).
        assert_eq!(user_code_selector().0, 0x33);
        assert_eq!(user_data_selector().0, 0x2b);
        assert_eq!(USER_CODE_INDEX, 6);
        assert_eq!(USER_DATA_INDEX, 5);
    }

    #[test]
    fn init_returns_code_selector() {
        // init() itself cannot run under the host test target (it
        // executes lgdt/ltr), so assert the accessor it returns agrees
        // with the code selector the IDT is built from.
        assert_eq!(code_selector(), GDT.1);
        assert_eq!(data_selector(), GDT.2);
        assert_eq!(tss_selector(), GDT.3);
    }
}
