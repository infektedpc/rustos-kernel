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
//! ## Boot sequence ([`init_interrupts`])
//!
//! 1. Interrupts disabled.
//! 2. PIC remap (master -> 32, slave -> 40), IRQ0 + IRQ1 unmasked.
//! 3. PIT programmed to 100 Hz.
//! 4. APIC MMIO pages mapped (LAPIC `0xFEE00000`, IOAPIC
//!    `0xFEC00000`) and [`init_apic`] run: LAPIC enabled (spurious
//!    vector `0xFF`), APIC timer + LINTs masked, IOAPIC entries for
//!    IRQ0/IRQ1 programmed but **masked** — delivery stays with the
//!    PIC, so nothing is double-delivered. If the pages cannot be
//!    mapped, boot continues in PIC mode with a serial warning.
//! 5. Interrupts enabled (`sti`). From here the timer fires at 100 Hz
//!    and keys queue into the ring buffer.
//!
//! ## PIC vs APIC mode
//!
//! Boot is always PIC mode: the 8259s deliver IRQ0/IRQ1 and EOI goes
//! to the PIC. [`switch_to_apic_mode`] hands delivery to the IOAPIC
//! (masks the PIC, unmasks the IOAPIC entries) and switches EOI to
//! the LAPIC — but it is **not** called from `kernel_main` yet: the
//! switch has not been verified on real hardware (no QEMU on this
//! VM), and a wrong IOAPIC base or APIC id would silently lose the
//! timer. The APIC bases are the defaults, not discovered from
//! ACPI/MADT, and the APIC base is not read from the MSR — see the
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
// Local APIC, once its MMIO page is mapped (see `init_interrupts`).
// `None` until then — and forever, if the pages could not be mapped.
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
        idt[TIMER_VECTOR].set_handler_fn(timer_interrupt_handler);
        idt[KEYBOARD_VECTOR].set_handler_fn(keyboard_interrupt_handler);
        idt
    };
}

/// Load the IDT. Call once, early in `kernel_main`, after `gdt::init()`.
pub fn init_idt() {
    IDT.load();
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
/// Order: interrupts off -> PIC remap + unmask IRQ0/IRQ1 -> PIT at
/// 100 Hz -> map APIC pages + `init_apic` (LAPIC on, IOAPIC entries
/// programmed but masked) -> interrupts on. Delivery stays with the
/// PIC; see the module docs for the APIC mode switch.
///
/// Call after `gdt::init()`, `init_idt()`, paging, and the heap are
/// up (the APIC page mapping needs the mapper + frame allocator).
pub fn init_interrupts(
    mapper: &mut OffsetPageTable,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) {
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

    match map_apic_pages(mapper, frame_allocator) {
        Ok(()) => {
            let mut lapic =
                LocalApic::new(unsafe { VolatileRegWindow::new(LAPIC_BASE as usize) });
            let mut ioapic =
                IoApic::new(unsafe { VolatileRegWindow::new(IOAPIC_BASE as usize) });
            // BSP APIC id 0. Entries stay masked: the PIC still owns
            // delivery until switch_to_apic_mode() is called.
            init_apic(&mut lapic, &mut ioapic, 0);
            *LAPIC.lock() = Some(lapic);
            *IOAPIC.lock() = Some(ioapic);
        }
        Err(e) => {
            serial_println!(
                "interrupts: APIC pages could not be mapped ({:?}); continuing in PIC mode",
                e
            );
        }
    }

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

/// Switch IRQ delivery from the PIC to the IOAPIC: mask every PIC
/// line, unmask the IOAPIC entries for IRQ0/IRQ1, and route EOI to
/// the LAPIC from here on.
///
/// NOT called from `kernel_main` yet. The switch changes which
/// controller owns the timer and keyboard, and it has not been
/// verified on real hardware (no QEMU on this VM). Only call it
/// after [`init_interrupts`] has mapped the APIC pages; if the APIC
/// was never initialised it prints a warning and stays in PIC mode.
pub fn switch_to_apic_mode() {
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
}
