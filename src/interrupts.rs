//! Interrupt Descriptor Table (IDT).
//!
//! Handlers installed here:
//!
//! * **Breakpoint** (`int3`) — prints the stack frame and returns, so
//!   the kernel can prove its exception handling works by deliberately
//!   triggering one.
//! * **Double fault** — runs on the dedicated IST stack set up in
//!   `crate::gdt`; panics (there is no recovering from a double fault).
//! * **Page fault** — prints the faulting address (`CR2`), the error
//!   code, and the stack frame, then halts.
//!
//! Deliberately *not* installed yet: timer and keyboard interrupts.
//! Those need the PICs remapped first (the CPU's default IRQ routing
//! collides with CPU exceptions 0–15), which is a follow-up. Until then
//! no hardware interrupts are enabled, so the IDT only serves CPU
//! exceptions.

use crate::{gdt, hlt_loop, println};
use lazy_static::lazy_static;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        unsafe {
            idt.double_fault
                .set_handler_fn(double_fault_handler)
                .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
        }
        idt.page_fault.set_handler_fn(page_fault_handler);
        idt
    };
}

/// Load the IDT. Call once, early in `kernel_main`, after `gdt::init()`.
pub fn init_idt() {
    IDT.load();
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame,
    _error_code: u64,
) -> ! {
    panic!("EXCEPTION: DOUBLE FAULT\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn page_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: PageFaultErrorCode,
) {
    use x86_64::registers::control::Cr2;

    println!("EXCEPTION: PAGE FAULT");
    println!("Accessed Address: {:?}", Cr2::read());
    println!("Error Code: {:?}", error_code);
    println!("{:#?}", stack_frame);
    hlt_loop();
}
