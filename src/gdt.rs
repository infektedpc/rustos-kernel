//! Global Descriptor Table (GDT) and Task State Segment (TSS).
//!
//! The GDT holds the kernel code segment and the TSS descriptor. The TSS
//! holds a dedicated interrupt stack for the double-fault handler: when
//! the CPU double-faults (e.g. a stack overflow in an exception handler),
//! it switches to this known-good stack via the Interrupt Stack Table
//! (IST), so the handler can run even when the normal stack is broken.
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

lazy_static! {
    static ref GDT: (GlobalDescriptorTable, SegmentSelector, SegmentSelector) = {
        let mut gdt = GlobalDescriptorTable::new();
        let code_selector = gdt.append(Descriptor::kernel_code_segment());
        let tss_selector = gdt.append(Descriptor::tss_segment(&TSS));
        (gdt, code_selector, tss_selector)
    };
}

/// Load the GDT, switch to the kernel code segment, and load the TSS.
///
/// Must be called once, early in `kernel_main`, before interrupts are
/// enabled or exceptions are expected.
pub fn init() {
    use x86_64::instructions::segmentation::{Segment, CS};
    use x86_64::instructions::tables::load_tss;

    GDT.0.load();
    unsafe {
        CS::set_reg(GDT.1);
        load_tss(GDT.2);
    }
}
