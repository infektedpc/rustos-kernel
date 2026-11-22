//! Interrupt Descriptor Table (IDT) + hardware interrupt setup.
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
//! * **Timer** (vector 32, PIC IRQ0) — see [`handle_timer_irq`]: each
//!   PIT tick advances the [`TimerIrq`]; at a quantum boundary (every
//!   tick at 100 Hz) it calls `Scheduler::tick()`, and a `Preempt`
//!   decision is recorded in `NEED_RESCHED` for the idle loop to act
//!   on. EOI goes to whichever controller owns the IRQs (see below).
//! * **Keyboard** (vector 33, PIC IRQ1) — see
//!   [`handle_keyboard_irq`]: reads the scancode from port `0x60`,
//!   decodes it, and queues a [`KeyEvent`] for the idle loop to drain.
//!   EOI is always sent, even when no byte was waiting or the ring
//!   was full, so the controller never wedges.
//!
//! Every CPU exception vector (0-21, plus 28-30) has a handler, so no
//! fault ever escalates silently to a double fault again: each prints
//! the vector name, the error code where the CPU pushes one, and the
//! stack frame to VGA and serial, then halts. (Breakpoint returns;
//! double fault panics.) All gates are built with the code selector
//! returned by [`gdt::init()`] — never the live `CS` the x86_64 crate
//! would otherwise read at build time.
//!
//! The LAPIC stays **off** until [`switch_to_apic_mode`] is actually
//! called. On 2026-11-21 the boot path enabled the LAPIC (spurious
//! vector `0xFF`, timer + LINTs masked) while the PIC still owned
//! delivery — and the first timer IRQ after `sti` arrived with no
//! valid vector attached (QEMU `-d int` showed `v=ffffffff`, then
//! #GP with error `0xfffa`). The gate dump had already proved the
//! IDT gates good, so the fault was in the controller, not the table:
//! with LINT0 masked, the PIC's INTR reached the CPU through the
//! enabled LAPIC and there was no vector to deliver. Boot is
//! therefore PIC-only from end to end; the APIC pages are mapped and
//! `init_apic` runs only inside `switch_to_apic_mode`.
//!
//! ## Boot sequence ([`init_interrupts`])
//!
//! 1. Interrupts disabled.
//! 2. PIC remap (master -> 32, slave -> 40), IRQ0 + IRQ1 unmasked.
//! 3. PIT programmed to 100 Hz.
//! 4. The IDT gates for vectors 32/33 are dumped to serial a second
//!    time, right before `sti`. If a page mapping made after the
//!    `init_idt` dump corrupted the IDT, this is where it shows up —
//!    and it discriminates that from a controller fault: gates good
//!    here plus a #GP at `sti` means the interrupt controller, not
//!    the table.
//! 5. Interrupts enabled (`sti`). From here the timer fires at 100 Hz
//!    and keys queue into the ring buffer.
//!
//! ## PIC vs APIC mode
//!
//! Boot is always PIC mode: the 8259s deliver IRQ0/IRQ1 and EOI goes
//! to the PIC. [`switch_to_apic_mode`] maps the APIC MMIO pages,
//! runs [`init_apic`], then hands delivery to the IOAPIC (masks the
//! PIC, unmasks the IOAPIC entries) and switches EOI to the LAPIC —
//! but it is **not** called from `kernel_main` yet: the switch has
//! not been verified on real hardware (no QEMU on this VM), and a
//! wrong IOAPIC base or APIC id would silently lose the timer. The
//! APIC bases are the defaults, not discovered from ACPI/MADT, and
//! the APIC base is not read from the MSR — see the
//! `rustos-interrupts` README.

use crate::memory::map_mmio_page;
use crate::{gdt, hlt_loop, println, serial_println};
use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use lazy_static::lazy_static;
use rustos_interrupts::apic::{IoApic, LocalApic, VolatileRegWindow, LAPIC_BASE, IOAPIC_BASE};
use rustos_interrupts::idt::{KEYBOARD_VECTOR, PIC_IRQ_BASE, TIMER_VECTOR};
use rustos_interrupts::pic::{Pic, IRQ_KEYBOARD, IRQ_TIMER};
use rustos_interrupts::{
    init_apic, switch_to_apic, ControllerMode, KeyEvent, KeyboardIrq, RealPorts, TimerIrq,
};
use rustos_interrupts::{pit};
use rustos_scheduler::{switch, Scheduler, TaskId, TickOutcome};
use spin::Mutex;
use x86_64::structures::gdt::SegmentSelector;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use x86_64::structures::paging::mapper::MapToError;
use x86_64::structures::paging::{FrameAllocator, OffsetPageTable, Size4KiB};

/// PIT interrupt frequency, in Hz. Must match the scheduler's
/// `TICK_HZ` (100) so one PIT tick is one scheduler timeslice.
pub const TIMER_HZ: u32 = 100;

// The 8259 PIC pair (real ports). Owned by this module; the timer
// and keyboard stubs send EOI through it in PIC mode.
lazy_static! {
    static ref PIC: Mutex<Pic<RealPorts>> = Mutex::new(Pic::new(RealPorts));
}
// Local APIC, once its MMIO page is mapped (see
// `switch_to_apic_mode`). `None` until then — and forever in PIC
// mode: the LAPIC stays off at boot (see the module docs).
lazy_static! {
    static ref LAPIC: Mutex<Option<LocalApic<VolatileRegWindow>>> = Mutex::new(None);
}
// IOAPIC, once its MMIO page is mapped.
lazy_static! {
    static ref IOAPIC: Mutex<Option<IoApic<VolatileRegWindow>>> = Mutex::new(None);
}
// Timer IRQ state: PIT ticks grouped into scheduler quanta.
lazy_static! {
    static ref TIMER_IRQ: Mutex<TimerIrq> = Mutex::new(TimerIrq::default());
}
// The scheduler the timer IRQ drives. Empty (no tasks) until the
// kernel adds some — `tick()` then just counts ticks and returns
// `Continue`, so the timer path is live but has nothing to preempt.
lazy_static! {
    static ref SCHEDULER: Mutex<Scheduler> = Mutex::new(Scheduler::new());
}
// Keyboard IRQ state: PS/2 decode + event ring.
lazy_static! {
    static ref KEYBOARD_IRQ: Mutex<KeyboardIrq<RealPorts>> =
        Mutex::new(KeyboardIrq::new(RealPorts));
}

/// Which controller owns IRQ delivery: 0 = PIC, 1 = APIC.
static MODE: AtomicU8 = AtomicU8::new(0);
/// Task id the timer IRQ asked to switch to (0 = none). Written by
/// the timer handler, consumed by [`maybe_reschedule`] in the idle
/// loop — never switched inside the interrupt handler itself.
static NEED_RESCHED: AtomicU64 = AtomicU64::new(0);

/// Build the IDT with an explicit code selector on every entry.
///
/// The x86_64 crate's `set_handler_fn` fills each gate's code
/// selector from the live `CS` register at build time. That was the
/// first suspect for the 2026-11-21 #GP (error code 0xFFFA) at the
/// first timer IRQ after `sti` — but the `init_idt` gate dump proved
/// the gates good (index 1, present), and the fault turned out to be
/// the LAPIC being enabled while the PIC still owned delivery (see
/// the module docs). The explicit selector stays anyway: it removes
/// one variable from the table entirely, and `init_idt` dumps gates
/// 32/33 to serial so every boot proves it.
fn build_idt(cs: SegmentSelector) -> InterruptDescriptorTable {
    let mut idt = InterruptDescriptorTable::new();
    unsafe {
        idt.divide_error
            .set_handler_fn(divide_error_handler)
            .set_code_selector(cs);
        idt.debug
            .set_handler_fn(debug_handler)
            .set_code_selector(cs);
        idt.non_maskable_interrupt
            .set_handler_fn(nmi_handler)
            .set_code_selector(cs);
        idt.breakpoint
            .set_handler_fn(breakpoint_handler)
            .set_code_selector(cs);
        idt.overflow
            .set_handler_fn(overflow_handler)
            .set_code_selector(cs);
        idt.bound_range_exceeded
            .set_handler_fn(bound_range_handler)
            .set_code_selector(cs);
        idt.invalid_opcode
            .set_handler_fn(invalid_opcode_handler)
            .set_code_selector(cs);
        idt.device_not_available
            .set_handler_fn(device_not_available_handler)
            .set_code_selector(cs);
        idt.double_fault
            .set_handler_fn(double_fault_handler)
            .set_code_selector(cs)
            .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
        idt.invalid_tss
            .set_handler_fn(invalid_tss_handler)
            .set_code_selector(cs);
        idt.segment_not_present
            .set_handler_fn(segment_not_present_handler)
            .set_code_selector(cs);
        idt.stack_segment_fault
            .set_handler_fn(stack_segment_fault_handler)
            .set_code_selector(cs);
        idt.general_protection_fault
            .set_handler_fn(general_protection_fault_handler)
            .set_code_selector(cs);
        idt.page_fault
            .set_handler_fn(page_fault_handler)
            .set_code_selector(cs);
        idt.x87_floating_point
            .set_handler_fn(x87_floating_point_handler)
            .set_code_selector(cs);
        idt.alignment_check
            .set_handler_fn(alignment_check_handler)
            .set_code_selector(cs);
        idt.machine_check
            .set_handler_fn(machine_check_handler)
            .set_code_selector(cs);
        idt.simd_floating_point
            .set_handler_fn(simd_floating_point_handler)
            .set_code_selector(cs);
        idt.virtualization
            .set_handler_fn(virtualization_handler)
            .set_code_selector(cs);
        idt.cp_protection_exception
            .set_handler_fn(cp_protection_handler)
            .set_code_selector(cs);
        idt.hv_injection_exception
            .set_handler_fn(hv_injection_handler)
            .set_code_selector(cs);
        idt.vmm_communication_exception
            .set_handler_fn(vmm_communication_handler)
            .set_code_selector(cs);
        idt.security_exception
            .set_handler_fn(security_exception_handler)
            .set_code_selector(cs);
        idt[TIMER_VECTOR]
            .set_handler_fn(timer_interrupt_handler)
            .set_code_selector(cs);
        idt[KEYBOARD_VECTOR]
            .set_handler_fn(keyboard_interrupt_handler)
            .set_code_selector(cs);
    }
    idt
}

lazy_static! {
    static ref IDT: InterruptDescriptorTable = build_idt(gdt::code_selector());
}

/// Load the IDT. Call once, early in `kernel_main`, after `gdt::init()`.
///
/// After loading, dumps the IDTR and the timer/keyboard gates to
/// serial: handler address + code selector for vectors 32/33, so a
/// bad gate is visible in the boot log before the first IRQ fires.
pub fn init_idt() {
    IDT.load();
    dump_idt_gates();
}

/// Print the loaded IDT's base/limit and the timer + keyboard gates
/// (handler address, code selector, present bit) to serial.
fn dump_idt_gates() {
    use x86_64::instructions::tables::sidt;

    let idtr = sidt();
    serial_println!(
        "idt: base={:#x} limit={:#x}",
        idtr.base.as_u64(),
        idtr.limit
    );
    serial_println!("idt[32] timer: {:?}", IDT[TIMER_VECTOR]);
    serial_println!("idt[33] keyboard: {:?}", IDT[KEYBOARD_VECTOR]);
}

/// Which controller currently delivers hardware IRQs.
pub fn mode() -> ControllerMode {
    if MODE.load(Ordering::SeqCst) == 1 {
        ControllerMode::Apic
    } else {
        ControllerMode::Pic
    }
}

/// Total PIT ticks handled since boot (100 Hz in PIC mode).
pub fn timer_ticks() -> u64 {
    TIMER_IRQ.lock().ticks()
}

/// Pop the oldest queued key event, or `None` when the ring is empty.
pub fn pop_key_event() -> Option<KeyEvent> {
    KEYBOARD_IRQ.lock().pop_event()
}

/// Key events dropped because the ring was full.
pub fn keyboard_dropped() -> u64 {
    KEYBOARD_IRQ.lock().dropped()
}

/// Initialise the interrupt controllers and enable interrupts.
///
/// PIC-only by design. Order: interrupts off -> PIC remap + unmask
/// IRQ0/IRQ1 -> PIT at 100 Hz -> re-dump the IDT gates -> interrupts
/// on. The LAPIC is deliberately left off here: on 2026-11-21
/// enabling it (with LINT0 masked) while the PIC still owned
/// delivery made the first timer IRQ arrive with no valid vector
/// (`v=ffffffff` in the QEMU log, then #GP) — so the APIC pages are
/// mapped and `init_apic` runs only inside [`switch_to_apic_mode`],
/// when the APIC-mode switch is actually requested.
///
/// Call after `gdt::init()`, `init_idt()`, paging, and the heap are
/// up. Takes no mapper: nothing in the PIC-only boot path maps
/// pages anymore.
pub fn init_interrupts() {
    use x86_64::instructions::interrupts;

    interrupts::disable();

    {
        let mut pic = PIC.lock();
        pic.remap_default();
        pic.unmask_irq(IRQ_TIMER);
        pic.unmask_irq(IRQ_KEYBOARD);
    }
    {
        let mut ports = RealPorts;
        pit::init(&mut ports, TIMER_HZ);
    }

    // Second gate dump, right before `sti`: if a page mapping made
    // after the `init_idt` dump corrupted the IDT, this is where it
    // shows up. Gates good here plus a #GP at `sti` would mean the
    // interrupt controller, not the table.
    serial_println!("interrupts: IDT gates before sti:");
    dump_idt_gates();

    MODE.store(0, Ordering::SeqCst);
    interrupts::enable();
}

/// Map the LAPIC and IOAPIC MMIO pages identity-mapped.
fn map_apic_pages(
    mapper: &mut OffsetPageTable,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> Result<(), MapToError<Size4KiB>> {
    map_mmio_page(mapper, frame_allocator, LAPIC_BASE)?;
    map_mmio_page(mapper, frame_allocator, IOAPIC_BASE)?;
    Ok(())
}

/// Switch IRQ delivery from the PIC to the IOAPIC: map the APIC MMIO
/// pages and run [`init_apic`] first (LAPIC on, IOAPIC entries
/// programmed but still masked), then mask every PIC line, unmask
/// the IOAPIC entries for IRQ0/IRQ1, and route EOI to the LAPIC from
/// here on.
///
/// NOT called from `kernel_main` yet. The switch changes which
/// controller owns the timer and keyboard, and it has not been
/// verified on real hardware (no QEMU on this VM). Only call it when
/// the APIC-mode switch is actually wanted: the LAPIC must stay off
/// while the PIC owns delivery (see the module docs). If the APIC
/// pages cannot be mapped it prints a warning and stays in PIC mode.
pub fn switch_to_apic_mode(
    mapper: &mut OffsetPageTable,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) {
    match map_apic_pages(mapper, frame_allocator) {
        Ok(()) => {
            let mut lapic =
                LocalApic::new(unsafe { VolatileRegWindow::new(LAPIC_BASE as usize) });
            let mut ioapic =
                IoApic::new(unsafe { VolatileRegWindow::new(IOAPIC_BASE as usize) });
            // BSP APIC id 0. Entries stay masked until the switch
            // below unmasks them.
            init_apic(&mut lapic, &mut ioapic, 0);
            *LAPIC.lock() = Some(lapic);
            *IOAPIC.lock() = Some(ioapic);
        }
        Err(e) => {
            serial_println!(
                "switch_to_apic_mode: APIC pages could not be mapped ({:?}); staying in PIC mode",
                e
            );
            return;
        }
    }
    let mut ioapic_guard = IOAPIC.lock();
    let Some(ioapic) = ioapic_guard.as_mut() else {
        serial_println!("switch_to_apic_mode: APIC not initialised; staying in PIC mode");
        return;
    };
    let mut pic = PIC.lock();
    switch_to_apic(&mut pic, ioapic);
    MODE.store(1, Ordering::SeqCst);
    serial_println!("interrupts: delivery switched to APIC (IOAPIC)");
}

/// Send end-of-interrupt for `vector` to the controller that owns
/// delivery right now: the LAPIC in APIC mode, the PIC in PIC mode
/// (slave first for IRQ >= 8). Called at the end of every hardware
/// IRQ handler, unconditionally.
fn send_eoi(vector: u8) {
    if MODE.load(Ordering::SeqCst) == 1 {
        if let Some(lapic) = LAPIC.lock().as_mut() {
            lapic.eoi();
            return;
        }
        // APIC mode without a LAPIC: fall through to the PIC rather
        // than dropping the EOI entirely.
    }
    if (PIC_IRQ_BASE..PIC_IRQ_BASE + 16).contains(&vector) {
        PIC.lock().send_eoi(vector - PIC_IRQ_BASE);
    }
}

/// One timer IRQ against explicit state — the handler body, factored
/// so host tests can drive it with a local timer + scheduler.
/// Returns the scheduler's decision at a quantum boundary, or `None`
/// on a plain tick.
fn timer_irq_step(timer: &mut TimerIrq, scheduler: &mut Scheduler) -> Option<TickOutcome> {
    timer.handle_irq_with(|| scheduler.tick())
}

/// Handle one timer IRQ (vector 32): advance the PIT tick counter and
/// run the scheduler tick at each quantum boundary. A `Preempt`
/// decision is recorded for the idle loop; the switch itself never
/// happens inside the interrupt handler.
fn handle_timer_irq() {
    let outcome = {
        let mut timer = TIMER_IRQ.lock();
        let mut scheduler = SCHEDULER.lock();
        timer_irq_step(&mut timer, &mut scheduler)
    };
    if let Some(TickOutcome::Preempt(next)) = outcome {
        NEED_RESCHED.store(next.0, Ordering::SeqCst);
    }
}

/// One keyboard IRQ against explicit state — the handler body,
/// factored so host tests can drive it with mock ports.
fn keyboard_irq_step<P: rustos_interrupts::PortIo>(kb: &mut KeyboardIrq<P>) -> bool {
    kb.on_irq()
}

/// Handle one keyboard IRQ (vector 33): read a waiting scancode (if
/// any), decode it, and queue the event. Always returns promptly; EOI
/// is sent by the caller afterwards, whether or not a byte was read.
fn handle_keyboard_irq() {
    let mut kb = KEYBOARD_IRQ.lock();
    keyboard_irq_step(&mut kb);
}

/// Act on a pending preempt request from the timer IRQ, if any.
/// Called from the idle loop, never from an interrupt handler: the
/// register switch runs on the kernel's own stack, with the
/// scheduler's bookkeeping done first. If no task is running yet
/// (the kernel has no tasks), the request is dropped — the scheduler
/// already moved the preempted task back to `Ready`, so nothing is
/// lost except a switch that had nowhere to go.
pub fn maybe_reschedule() {
    let next_id = NEED_RESCHED.swap(0, Ordering::SeqCst);
    if next_id == 0 {
        return;
    }
    let next = TaskId(next_id);
    let mut scheduler = SCHEDULER.lock();
    if scheduler.current_id().is_none() {
        return;
    }
    let Some((from, to)) = (unsafe { scheduler.context_switch(next) }) else {
        return;
    };
    // Bookkeeping first: `context_switch` only borrowed the task
    // list's contexts, and `switch_to` does not reallocate it, so the
    // raw pointers stay valid. Then the register switch itself.
    scheduler.switch_to(next);
    unsafe {
        switch(from, to);
    }
}

extern "x86-interrupt" fn timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
    handle_timer_irq();
    send_eoi(TIMER_VECTOR);
}

extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    handle_keyboard_irq();
    send_eoi(KEYBOARD_VECTOR);
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
}

/// Report an exception on VGA and serial, then halt. The `hlt_loop`
/// at the end never returns, so this is the body for every fault
/// handler except breakpoint (which returns) and double fault (which
/// panics with the frame instead).
fn report_exception(name: &str, error_code: Option<u64>, stack_frame: &InterruptStackFrame) -> ! {
    println!("EXCEPTION: {}", name);
    if let Some(code) = error_code {
        println!("Error Code: {:#x}", code);
    }
    println!("{:#?}", stack_frame);
    serial_println!("EXCEPTION: {}", name);
    if let Some(code) = error_code {
        serial_println!("Error Code: {:#x}", code);
    }
    serial_println!("{:#?}", stack_frame);
    hlt_loop();
}

/// Define an `extern "x86-interrupt"` handler for an exception that
/// pushes no error code: report it and halt.
macro_rules! exception_handler {
    ($name:ident, $label:expr) => {
        extern "x86-interrupt" fn $name(stack_frame: InterruptStackFrame) {
            report_exception($label, None, &stack_frame);
        }
    };
}

/// Define an `extern "x86-interrupt"` handler for an exception that
/// pushes an error code: report it (with the code) and halt.
macro_rules! exception_handler_with_err_code {
    ($name:ident, $label:expr) => {
        extern "x86-interrupt" fn $name(stack_frame: InterruptStackFrame, error_code: u64) {
            report_exception($label, Some(error_code), &stack_frame);
        }
    };
}

exception_handler!(divide_error_handler, "DIVIDE ERROR");
exception_handler!(debug_handler, "DEBUG");
exception_handler!(nmi_handler, "NON-MASKABLE INTERRUPT");
exception_handler!(overflow_handler, "OVERFLOW");
exception_handler!(bound_range_handler, "BOUND RANGE EXCEEDED");
exception_handler!(invalid_opcode_handler, "INVALID OPCODE");
exception_handler!(device_not_available_handler, "DEVICE NOT AVAILABLE");
exception_handler!(x87_floating_point_handler, "X87 FLOATING POINT");
exception_handler!(simd_floating_point_handler, "SIMD FLOATING POINT");
exception_handler!(virtualization_handler, "VIRTUALIZATION");
exception_handler!(hv_injection_handler, "HV INJECTION");

exception_handler_with_err_code!(invalid_tss_handler, "INVALID TSS");
exception_handler_with_err_code!(segment_not_present_handler, "SEGMENT NOT PRESENT");
exception_handler_with_err_code!(stack_segment_fault_handler, "STACK SEGMENT FAULT");
exception_handler_with_err_code!(alignment_check_handler, "ALIGNMENT CHECK");
exception_handler_with_err_code!(cp_protection_handler, "CONTROL PROTECTION");
exception_handler_with_err_code!(vmm_communication_handler, "VMM COMMUNICATION");
exception_handler_with_err_code!(security_exception_handler, "SECURITY EXCEPTION");

extern "x86-interrupt" fn machine_check_handler(stack_frame: InterruptStackFrame) -> ! {
    report_exception("MACHINE CHECK", None, &stack_frame);
}

extern "x86-interrupt" fn general_protection_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    // The #GP error code carries the faulting segment selector in
    // bits 15:3 (bit 0 = external event, bit 1 = IDT vs GDT/LDT), so
    // print it decoded: the next boot log names the bad gate directly
    // (2026-11-21: 0xfffa = selector index 0x1fff).
    println!("EXCEPTION: GENERAL PROTECTION FAULT");
    println!(
        "Error Code: {:#x} (segment selector index {:#x}, {})",
        error_code,
        error_code >> 3,
        if error_code & 1 == 1 { "external" } else { "internal" }
    );
    println!("{:#?}", stack_frame);
    serial_println!("EXCEPTION: GENERAL PROTECTION FAULT");
    serial_println!(
        "Error Code: {:#x} (segment selector index {:#x}, {})",
        error_code,
        error_code >> 3,
        if error_code & 1 == 1 { "external" } else { "internal" }
    );
    serial_println!("{:#?}", stack_frame);
    hlt_loop();
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

    // Mirror to serial as well as VGA: if the fault was on the VGA
    // buffer itself (or VGA is unavailable), the VGA prints are
    // no-ops and the serial log is the only record of the fault.
    println!("EXCEPTION: PAGE FAULT");
    println!("Accessed Address: {:?}", Cr2::read());
    println!("Error Code: {:?}", error_code);
    println!("{:#?}", stack_frame);
    serial_println!("EXCEPTION: PAGE FAULT");
    serial_println!("Accessed Address: {:?}", Cr2::read());
    serial_println!("Error Code: {:?}", error_code);
    serial_println!("{:#?}", stack_frame);
    hlt_loop();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustos_interrupts::{KeyboardIrq, MockPorts, TimerIrq};
    use rustos_scheduler::{Scheduler, Task, TaskId};

    fn idle_entry() {}

    /// Scheduler with two same-priority tasks, `a` running.
    /// (Stack buffers are leaked on purpose, as in the scheduler's
    /// own tests: tasks only store addresses.)
    fn sched_with_two_running() -> (Scheduler, TaskId, TaskId) {
        let mut sched = Scheduler::new();
        let mut ids = Vec::new();
        for name in ["a", "b"] {
            let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 8192]));
            let rsp = unsafe { Task::prepare_stack(buf.as_mut_ptr(), buf.len(), idle_entry) }
                .expect("stack");
            let top = buf.as_ptr() as u64 + buf.len() as u64;
            ids.push(sched.add_task(name, 5, rsp, top, buf.len(), idle_entry));
        }
        sched.run_next();
        (sched, ids[0], ids[1])
    }

    #[test]
    fn timer_step_preempts_on_first_quantum_at_100hz() {
        // Kernel default: TimerIrq quantum 1 == one tick per slice.
        let mut timer = TimerIrq::default();
        let (mut sched, _a, b) = sched_with_two_running();
        assert_eq!(timer_irq_step(&mut timer, &mut sched), Some(TickOutcome::Preempt(b)));
        assert_eq!(timer.ticks(), 1);
        assert_eq!(timer.quanta(), 1);
    }

    #[test]
    fn timer_step_quiet_before_quantum_then_preempts() {
        let mut timer = TimerIrq::new(10);
        let (mut sched, _a, b) = sched_with_two_running();
        for _ in 0..9 {
            assert_eq!(timer_irq_step(&mut timer, &mut sched), None);
        }
        assert_eq!(timer_irq_step(&mut timer, &mut sched), Some(TickOutcome::Preempt(b)));
        assert_eq!(timer.ticks(), 10);
        assert_eq!(timer.quanta(), 1);
    }

    #[test]
    fn timer_step_with_no_tasks_is_a_noop_but_counts() {
        let mut timer = TimerIrq::default();
        let mut sched = Scheduler::new();
        assert_eq!(timer_irq_step(&mut timer, &mut sched), Some(TickOutcome::Continue));
        assert_eq!(sched.ticks(), 1);
        assert_eq!(timer.ticks(), 1);
    }

    #[test]
    fn keyboard_step_queues_events_for_the_idle_loop() {
        let mut kb = KeyboardIrq::new(MockPorts::new());
        // 'a' make, shift make, 'b' make -> 'B'.
        for code in [0x1Eu8, 0x2A, 0x30] {
            kb.ports().set(rustos_interrupts::keyboard::PS2_STATUS_PORT, 0x01);
            kb.ports().set(rustos_interrupts::keyboard::PS2_DATA_PORT, code);
            assert!(keyboard_irq_step(&mut kb));
        }
        assert_eq!(kb.pop_event().unwrap().ch, Some('a'));
        assert_eq!(kb.pop_event().unwrap().ch, None); // shift make
        assert_eq!(kb.pop_event().unwrap().ch, Some('B'));
        assert_eq!(kb.pop_event(), None);
    }

    #[test]
    fn keyboard_vector_eoi_goes_to_the_pic() {
        // The EOI half of the vector-33 handler, against mock ports:
        // a keyboard IRQ must end with an EOI to the master PIC.
        let mut pic = Pic::new(MockPorts::new());
        rustos_interrupts::Eoi::eoi(&mut pic, KEYBOARD_VECTOR);
        assert_eq!(pic.ports().write_count(rustos_interrupts::pic::PIC1_COMMAND, 0x20), 1);
    }

    #[test]
    fn maybe_reschedule_with_no_current_task_drops_the_request() {
        NEED_RESCHED.store(99, Ordering::SeqCst);
        maybe_reschedule(); // no tasks: must not panic, flag cleared
        assert_eq!(NEED_RESCHED.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn timer_and_keyboard_state_start_empty() {
        assert_eq!(timer_ticks(), 0);
        assert_eq!(pop_key_event(), None);
        assert_eq!(mode(), ControllerMode::Pic);
    }

    #[test]
    fn idt_entries_carry_explicit_kernel_code_selector() {
        // The kernel code segment is the first GDT entry after the
        // null descriptor: index 1, value 0x08.
        let cs = gdt::code_selector();
        assert_eq!(cs.0, 0x08);
        assert_eq!(cs.index(), 1);

        let idt = build_idt(cs);
        // Every installed entry must point at a real handler and carry
        // the explicit selector — never whatever live CS the crate
        // would otherwise read at build time (0xFFFA on the
        // 2026-11-21 boot, which #GP'd the first timer IRQ). The
        // Debug form prints `code_selector: SegmentSelector { index:
        // 1, ... }` and `present: true` for a good gate.
        let gates = [
            format!("{:?}", idt.divide_error),
            format!("{:?}", idt.debug),
            format!("{:?}", idt.non_maskable_interrupt),
            format!("{:?}", idt.breakpoint),
            format!("{:?}", idt.overflow),
            format!("{:?}", idt.bound_range_exceeded),
            format!("{:?}", idt.invalid_opcode),
            format!("{:?}", idt.device_not_available),
            format!("{:?}", idt.double_fault),
            format!("{:?}", idt.invalid_tss),
            format!("{:?}", idt.segment_not_present),
            format!("{:?}", idt.stack_segment_fault),
            format!("{:?}", idt.general_protection_fault),
            format!("{:?}", idt.page_fault),
            format!("{:?}", idt.x87_floating_point),
            format!("{:?}", idt.alignment_check),
            format!("{:?}", idt.machine_check),
            format!("{:?}", idt.simd_floating_point),
            format!("{:?}", idt.virtualization),
            format!("{:?}", idt.cp_protection_exception),
            format!("{:?}", idt.hv_injection_exception),
            format!("{:?}", idt.vmm_communication_exception),
            format!("{:?}", idt.security_exception),
            format!("{:?}", idt[TIMER_VECTOR]),
            format!("{:?}", idt[KEYBOARD_VECTOR]),
        ];
        assert_eq!(gates.len(), 25);
        for gate in &gates {
            assert!(gate.contains("index: 1"), "gate lost its code selector: {gate}");
            assert!(gate.contains("present: true"), "gate not present: {gate}");
            assert!(
                !gate.contains("handler_addr: 0x0"),
                "gate has no handler: {gate}"
            );
        }
        // The double-fault handler keeps its dedicated IST stack.
        assert!(
            gates[8].contains("stack_index: Some(0)"),
            "double fault lost its IST stack: {}",
            gates[8]
        );
    }

    #[test]
    fn apic_statics_start_unpopulated_and_mode_is_pic() {
        // Boot is PIC-only by design: the LAPIC stays off until
        // switch_to_apic_mode is actually called (enabling it while
        // the PIC owned delivery #GP'd the first timer IRQ on
        // 2026-11-21 — the interrupt arrived with no valid vector,
        // v=ffffffff in the QEMU log). Nothing in the boot path may
        // populate these anymore; switch_to_apic_mode is the only
        // place that does.
        assert!(LAPIC.lock().is_none(), "LAPIC populated before any APIC-mode switch");
        assert!(IOAPIC.lock().is_none(), "IOAPIC populated before any APIC-mode switch");
        assert_eq!(MODE.load(Ordering::SeqCst), 0, "delivery mode is not PIC at boot");
        assert_eq!(mode(), ControllerMode::Pic);
    }
}
