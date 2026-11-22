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

// GDT layout (indices after the null descriptor at 0):
// * code segment — index 1, selector 0x08
// * data segment — index 2, selector 0x10
// * TSS descriptor — index 3 (occupies entries 3–4), selector 0x18
lazy_static! {
    static ref GDT: (
        GlobalDescriptorTable,
        SegmentSelector,
        SegmentSelector,
        SegmentSelector,
    ) = {
        let mut gdt = GlobalDescriptorTable::new();
        let code_selector = gdt.append(Descriptor::kernel_code_segment());
        let data_selector = gdt.append(Descriptor::kernel_data_segment());
        let tss_selector = gdt.append(Descriptor::tss_segment(&TSS));
        (gdt, code_selector, data_selector, tss_selector)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdt_layout_selectors() {
        assert_eq!(code_selector().index(), 1);
        assert_eq!(data_selector().index(), 2);
        assert_eq!(tss_selector().index(), 3);
        // Values are index << 3, RPL 0 (kernel, Ring 0).
        assert_eq!(code_selector().0, 0x08);
        assert_eq!(data_selector().0, 0x10);
        assert_eq!(tss_selector().0, 0x18);
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
