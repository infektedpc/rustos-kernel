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
//!   decision switches to the next task **right there, inside the
//!   handler** (see [`switch_to_task`]). EOI goes to whichever
//!   controller owns the IRQs (see below), and it goes out *before*
//!   any switch: the PIC holds IRQ0 in service until EOI, and the
//!   handler suspends mid-flight when it switches, so an EOI after
//!   the switch would not run until the task resumes — wedging the
//!   timer. The switch itself never happens with the `SCHEDULER`
//!   lock held (see `switch_to_task` for why that deadlocks).
//! * **Keyboard** (vector 33, PIC IRQ1) — see
//!   [`handle_keyboard_irq`]: reads the scancode from port `0x60`,
//!   decodes it, and queues a [`KeyEvent`] for the idle loop to drain.
//!   EOI is always sent, even when no byte was waiting or the ring
//!   was full, so the controller never wedges.
//! * **Keyboard** (vector 33, PIC IRQ1) — see
//!   [`handle_keyboard_irq`]: reads the scancode from port `0x60`,
//!   decodes it, and queues a [`KeyEvent`] for the idle loop to drain.
//!   EOI is always sent, even when no byte was waiting or the ring
//!   was full, so the controller never wedges.
//! * **Syscall** (vector 128 / `0x80`) — see [`crate::syscall`]: an
//!   interrupt gate at DPL 3 whose handler is an assembly stub that
//!   saves the trap's registers and dispatches on `rax` (libc's
//!   register convention). It is an interrupt gate, not a trap gate:
//!   the `yield` path locks the same scheduler mutex the timer IRQ
//!   takes, and a trap gate would let the timer land mid-`yield`
//!   and spin on that lock forever.
//!
//! Every CPU exception vector (0-21, plus 28-30) has a handler, so no
//! fault ever escalates silently to a double fault again: each prints
//! the vector name, the error code where the CPU pushes one, and the
//! stack frame to VGA and serial, then halts — **when the fault is in
//! the kernel (ring 0)**. A fault in a ring-3 (userspace) task kills
//! only that task instead (see [`kill_user_task`]): the handler
//! reads the faulting code segment's RPL from the frame, prints
//! vector + pid + RIP (+ the faulting address for a page fault),
//! terminates the task with exit code -1, and switches to the next
//! Ready task. The kernel keeps running. (Double fault still
//! panics: by the time it fires the first fault's state is gone,
//! and the handler itself may be what faulted.) (Breakpoint returns;
//! double fault panics.) All gates are built with the code selector
//! returned by [`gdt::init()`] — never the live `CS` the x86_64 crate
//! would otherwise read at build time.
//!
//! Ring-3 tasks ([`add_user_task`]) own a kernel stack of their own:
//! `TSS.rsp[0]` points at its top whenever the task runs, so
//! interrupts and syscalls from ring 3 land on it. `rsp[0]` is
//! updated on every switch ([`set_kernel_stack_for`]), before the
//! register switch, while interrupts are off — a timer IRQ landing
//! after the switch but before the update would push its frame onto
//! the previous task's kernel stack.
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
use crate::tasks;
use crate::{gdt, hlt_loop, println, serial_println, syscall};
use crate::syscall::SYSCALL_VECTOR;
use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use lazy_static::lazy_static;
use rustos_interrupts::apic::{IoApic, LocalApic, VolatileRegWindow, LAPIC_BASE, IOAPIC_BASE};
use rustos_interrupts::idt::{KEYBOARD_VECTOR, PIC_IRQ_BASE, TIMER_VECTOR};
use rustos_interrupts::pic::{Pic, IRQ_KEYBOARD, IRQ_TIMER};
use rustos_interrupts::{
    init_apic, switch_to_apic, ControllerMode, KeyEvent, KeyboardIrq, RealPorts, TimerIrq,
};
use rustos_interrupts::{pit};
use rustos_scheduler::{
    enter, prepare_ring3_stack, switch, Context, Ring3Frame, Scheduler, Task, TaskId, TickOutcome,
};
use spin::Mutex;
use x86_64::structures::gdt::SegmentSelector;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use x86_64::PrivilegeLevel;
use x86_64::VirtAddr;
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
// Lock-free mirror of the total PIT tick count, incremented once
// per timer IRQ by `handle_timer_irq` and read by `timer_ticks()`.
// The idle loop reads the tick count with interrupts enabled, so
// it must never take the `TIMER_IRQ` lock to do it: a tick landing
// while the loop held the lock would spin in the handler —
// interrupts are off inside the handler, so the loop could never
// release it, wedging the timer (and the uptime line) forever.
static TOTAL_TICKS: AtomicU64 = AtomicU64::new(0);
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
        // Syscall gate (vector 0x80): an INTERRUPT gate, not a trap
        // gate (see crate::syscall for why — the yield path locks the
        // same SCHEDULER mutex the timer IRQ takes), DPL 3 so
        // userspace can invoke it. The handler is the assembly stub
        // in crate::syscall, not a Rust x86-interrupt function: the
        // stub needs the trap's general-purpose registers, which the
        // InterruptStackFrame does not carry.
        idt[SYSCALL_VECTOR]
            .set_handler_addr(syscall::syscall_entry_addr())
            .set_code_selector(cs)
            .set_privilege_level(PrivilegeLevel::Ring3);
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

/// Print the loaded IDT's base/limit and the timer + keyboard +
/// syscall gates (handler address, code selector, present bit) to
/// serial.
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
    serial_println!("idt[128] syscall: {:?}", IDT[SYSCALL_VECTOR]);
}

/// Which controller currently delivers hardware IRQs.
pub fn mode() -> ControllerMode {
    if MODE.load(Ordering::SeqCst) == 1 {
        ControllerMode::Apic
    } else {
        ControllerMode::Pic
    }
}

/// Ticks between uptime reports: 500 ticks = 5 s at 100 Hz.
pub const UPTIME_INTERVAL_TICKS: u64 = 500;

/// True when an uptime line is due: `ticks` has advanced at least
/// [`UPTIME_INTERVAL_TICKS`] past the last report. Pure, so host
/// tests can pin the boundary.
///
/// (On the 2026-11-22 live boot no `uptime:` line appeared in ~26 s
/// — not because this never fired, but because the line was
/// VGA-only and the user runs with `-display none`. The print side
/// is fixed in `crate::idle_loop`, which now mirrors the line to
/// serial; this helper only owns the firing condition.)
pub fn uptime_due(ticks: u64, last_printed: u64) -> bool {
    ticks >= last_printed.saturating_add(UPTIME_INTERVAL_TICKS)
}

/// Total PIT ticks handled since boot (100 Hz in PIC mode).
///
/// Lock-free: a plain atomic load, never the `TIMER_IRQ` lock.
/// The idle loop calls this with interrupts enabled (see
/// `crate::idle_loop`); taking the lock there would deadlock the
/// moment a tick landed in the window — the handler spins on the
/// lock with interrupts off, so the holder could never release
/// it. The handler mirrors each tick into `TOTAL_TICKS` (see
/// `handle_timer_irq`); `Relaxed` is enough for a monotonic
/// uptime counter.
pub fn timer_ticks() -> u64 {
    TOTAL_TICKS.load(Ordering::Relaxed)
}

/// The currently-running task, if any (`None` before [`init_tasks`]).
pub fn current_task_id() -> Option<TaskId> {
    SCHEDULER.lock().current_id()
}

/// True when the currently-running task is a ring-3 (userspace)
/// task. The syscall dispatcher uses this to decide whether a
/// syscall buffer must be validated against the mapped user
/// regions (see `crate::syscall`): a userspace program that passes
/// a bad address gets `-EFAULT`, not a kernel `#PF`. False before
/// [`init_tasks`] and for every kernel task.
pub fn current_task_is_user() -> bool {
    let sched = SCHEDULER.lock();
    sched
        .current_id()
        .and_then(|id| sched.task(id))
        .map(|t| t.is_user)
        .unwrap_or(false)
}

/// Point `TSS.rsp[0]` at task `id`'s kernel stack top — the stack
/// the CPU switches to when it takes an interrupt or trap from
/// ring 3 while that task runs. (The idle task has no kernel stack
/// — `stack_top` 0 — so `rsp[0]` is set to 0 for it; idle never
/// runs in ring 3, so the value can never be used there.) The
/// `syscall`-instruction entry's per-CPU stack top (Batch 11) is
/// updated here too, in the same breath — a `syscall` landing
/// after the switch but before the update would run on the
/// previous task's kernel stack, exactly like an interrupt would.
///
/// Callers must hold the `SCHEDULER` lock already (this is called
/// from inside the locked region of the switch functions, right
/// after `switch_to(next)` and before the register switch) and run
/// with interrupts off, so no interrupt can land between the
/// `rsp[0]` update and the register switch: a timer IRQ landing
/// after the switch but before the update would push its frame
/// onto the *previous* task's kernel stack.
fn set_kernel_stack_for(scheduler: &Scheduler, id: TaskId) {
    let top = scheduler.task(id).map(|t| t.stack_top).unwrap_or(0);
    gdt::set_kernel_stack_top(VirtAddr::new(top));
    // The `syscall`-instruction entry reads its stack top from the
    // per-CPU SyscallGs (Batch 11), not the TSS — the two must move
    // together, on every switch, while interrupts are off.
    crate::fast_syscall::set_kernel_stack_top(VirtAddr::new(top));
}

/// Switch to task `next` now, from IRQ or syscall context.
///
/// The register switch itself is [`switch`] (rustos-scheduler): it
/// saves the current task's callee-saved registers + `rsp` into its
/// `Context` and resumes `next` right after *its* `switch()` call
/// (or in its entry point, on first run). The bookkeeping — the
/// outgoing task back to `Ready`, the incoming one to `Running` —
/// is `Scheduler::switch_to`, and it runs *after* the pointers are
/// taken and *before* the register switch: `context_switch` only
/// hands out pointers for `Ready` tasks, so the order is
/// `context_switch` first, `switch_to` second (the scheduler's own
/// tests pin this ordering).
///
/// The `SCHEDULER` lock is **dropped before the register switch**.
/// Holding it across the switch would deadlock: the task we switch
/// to runs with the timer IRQ live, and the next IRQ (or a `yield`
/// syscall from the new task) takes the same lock — it would spin
/// forever on a lock held by a suspended task. (The old
/// `maybe_reschedule` design — switch from the idle loop with the
/// lock held — could never work for exactly this reason.) The raw
/// context pointers stay valid across the unlock: single CPU, and
/// this path runs with interrupts off (interrupt gate / IRQ
/// context), so nothing can reallocate the scheduler's task list
/// between the unlock and the switch.
fn switch_to_task(next: TaskId) {
    let pair = {
        let mut scheduler = SCHEDULER.lock();
        let Some(pair) = (unsafe { scheduler.context_switch(next) }) else {
            return;
        };
        if pair.0.is_null() {
            // No current task to save (tick/yield never ask to switch
            // without one) — leave the bookkeeping alone: a null
            // `from` would store `rsp` at address 0.
            return;
        }
        scheduler.switch_to(next);
        // The next task's kernel stack is where ring-3 interrupts
        // land while it runs (TSS.rsp[0]) — update it BEFORE the
        // register switch, while interrupts are still off.
        set_kernel_stack_for(&scheduler, next);
        pair
    };
    let (from, to) = pair;
    unsafe {
        switch(from, to);
    }
}

/// Voluntary yield (syscall `SYS_YIELD`): switch to the next Ready
/// task now, from syscall context.
///
/// Called from the syscall handler only, never from IRQ context:
/// the vector-0x80 gate is an interrupt gate, so interrupts are
/// already off here and the timer IRQ cannot preempt the handler
/// while it holds the `SCHEDULER` lock (the timer handler takes the
/// same lock — see `crate::syscall`). Like [`switch_to_task`], the
/// lock is dropped before the register switch, for the same
/// deadlock reason. When nothing else is Ready the caller stays on
/// the current task with a fresh quantum; with no tasks at all
/// (before [`init_tasks`]) it is a no-op.
pub fn scheduler_yield() {
    let pair = {
        let mut scheduler = SCHEDULER.lock();
        let Some(current) = scheduler.current_id() else {
            return;
        };
        // The current task is Running, so pick_next() (Ready only)
        // already excludes it — the same pick yield_now() makes.
        // Nothing Ready: reset the quantum, stay put.
        let Some(next) = scheduler.pick_next() else {
            if let Some(t) = scheduler.task_mut(current) {
                t.reset_quantum();
            }
            return;
        };
        let Some(pair) = (unsafe { scheduler.context_switch(next) }) else {
            return;
        };
        if pair.0.is_null() {
            return;
        }
        scheduler.switch_to(next);
        // TSS.rsp[0] follows the task (see switch_to_task).
        set_kernel_stack_for(&scheduler, next);
        pair
    };
    let (from, to) = pair;
    unsafe {
        switch(from, to);
    }
}

/// Terminate the current task with `code` and switch to the next
/// Ready task. Never returns to the caller: the exiting task is
/// `Terminated` and its stack is abandoned — the switch uses
/// [`enter`] (load the next task's context, no save) instead of
/// [`switch`], because there is no context worth saving. The
/// exiting task's stack memory is not freed (no reaper yet); it
/// stays allocated until reboot.
///
/// Returns only when there is no Ready task to switch to (or no
/// task is current) — the caller must then halt, because the
/// current task is already dead. The idle task never reaches here:
/// exiting pid 0 halts in the syscall handler — the kernel itself
/// has nowhere to switch to.
pub fn exit_current_task(code: i32) {
    let to = {
        let mut scheduler = SCHEDULER.lock();
        let Some(current) = scheduler.current_id() else {
            return;
        };
        if !scheduler.exit(current, code) {
            return;
        }
        let Some(next) = scheduler.pick_next() else {
            return;
        };
        let Some(task) = scheduler.task(next) else {
            return;
        };
        // Take the target's context pointer before switch_to(): the
        // bookkeeping does not reallocate the task list, so it stays
        // valid, and the lock is dropped before enter() below.
        let to = &task.context as *const Context;
        scheduler.switch_to(next);
        // TSS.rsp[0] follows the task (see switch_to_task).
        set_kernel_stack_for(&scheduler, next);
        to
    };
    unsafe {
        enter(to);
    }
}

/// Guard that disables interrupts for a critical section and
/// restores the previous IF state on drop — the `spin_lock_irqsave`
/// pattern. Interrupts are only re-enabled if they were enabled
/// when the guard was created, so nested guards and use from
/// interrupt context (where IF is already off) are both safe: the
/// inner drop restores IF to off, the outer drop restores it to on.
///
/// Why this exists (Batch 8 wedge, 2026-11-22): a `spin::Mutex`
/// taken by preemptible ring-0 code and by non-preemptible code is
/// a deadlock. The worker task prints with interrupts on, holding
/// `SERIAL1` for ~2 ms per line (each byte spins on the UART
/// transmitter); a timer tick can preempt it mid-print and switch
/// to the ring-3 task, whose first `write` syscall runs with IF off
/// (interrupt gate) and spins on `SERIAL1` forever — the timer can
/// never fire to resume the worker, so the machine goes silently
/// dead with no fault. Every console-lock acquisition must therefore
/// hold this guard, so no ring-0 task can be preempted while holding
/// a console lock. The same hazard class already existed for the
/// `SCHEDULER` lock (see `init_tasks`' doc comment); the console
/// locks were the unguarded instance of it.
pub struct IrqGuard {
    was_enabled: bool,
}

impl IrqGuard {
    /// Disable interrupts, remembering whether they were on.
    ///
    /// On the host (unit tests) there are no interrupts to guard
    /// against — and `cli` would fault in ring 3 — so the guard is
    /// a no-op there (Batch 10: the `fbcon` init path is
    /// host-tested, and it holds this guard).
    pub fn hold() -> Self {
        #[cfg(test)]
        return IrqGuard { was_enabled: false };
        #[cfg(not(test))]
        {
            let was_enabled = x86_64::instructions::interrupts::are_enabled();
            x86_64::instructions::interrupts::disable();
            IrqGuard { was_enabled }
        }
    }
}

impl Drop for IrqGuard {
    fn drop(&mut self) {
        if self.was_enabled {
            x86_64::instructions::interrupts::enable();
        }
    }
}

/// Priority of the idle task (pid 0, `kernel_main`'s continuation).
/// MUST equal the worker / userspace priority: the scheduler picks
/// the highest-priority `Ready` task, and the worker plus any
/// userspace task are always `Ready` (they never block — `yield`
/// just switches), so a lower idle priority starves it forever.
/// That is what Batch 8's second live boot showed: with idle at 0
/// and worker/hello at 5, the boot stopped after
/// `task: pid 2 name=hello ...` with no
/// `userspace: hello task pid 2 created`, no
/// `RustOS kernel ready`, and no uptime, because `kernel_main`
/// never ran again. The scheduler documents the starvation
/// limitation ("no aging"); the kernel's own idle task must not be
/// a victim of it.
pub const IDLE_PRIORITY: u8 = 5;

/// Priority of the worker task (pid 1). Kept as a named constant
/// next to [`IDLE_PRIORITY`] so the equality the scheduler needs
/// is visible at the definition site, not a coincidence of two
/// literals.
pub const WORKER_PRIORITY: u8 = 5;

/// Create the kernel's tasks: the idle task (pid 0 — the kernel
/// itself, running on the boot stack) and one worker task (pid 1,
/// its own 16 KiB kernel stack). Call once from `kernel_main`,
/// after the heap is up and the syscall self-test has run. Prints
/// the task list to serial and VGA before returning — the first
/// switch happens on the next timer tick, ~10 ms later.
///
/// Interrupts are disabled while the scheduler is populated AND
/// while the task list prints, and restored to their previous state
/// only after the prints: the timer IRQ takes the same `SCHEDULER`
/// lock, and a tick landing mid-setup would spin on it while the
/// holder sits suspended underneath the handler. (Any path that
/// locks `SCHEDULER` with interrupts on has this hazard; the IRQ and
/// syscall paths run with interrupts off, so only this setup path
/// needs the explicit guard.) Keeping interrupts off through the
/// prints also makes the boot log deterministic: on the first Batch 7
/// boot a tick that was pending from the disabled setup window fired
/// the moment interrupts came back, the first switch preempted idle
/// before it printed anything, and the task list never appeared in
/// the log at all. With the prints inside the disabled region the
/// list always lands before the first switch can happen.
pub fn init_tasks() {
    use alloc::boxed::Box;

    let were_enabled = x86_64::instructions::interrupts::are_enabled();
    x86_64::instructions::interrupts::disable();

    let snapshot = {
        let mut scheduler = SCHEDULER.lock();
        // Idle: the kernel's own boot context. Its Context starts
        // empty — switch() fills in rsp the first time the timer
        // preempts it, and that saved rsp is what resumes it. The
        // entry is never called (see tasks::idle_entry); the record
        // exists so the scheduler can switch back to it. TaskId(0)
        // is the kernel's pid (getpid returns 0 for it); `current`
        // is Option<TaskId>, so Some(0) is unambiguous against "no
        // task".
        let idle = Task::new(TaskId(0), "idle", IDLE_PRIORITY, 0, 0, 0, tasks::idle_entry);
        scheduler.add_existing(idle);
        // Worker: its own stack, prepared so the first switch() to
        // it rets into worker_entry.
        let stack = Box::leak(Box::new([0u8; tasks::WORKER_STACK_SIZE]));
        let rsp = unsafe {
            Task::prepare_stack(stack.as_mut_ptr(), stack.len(), tasks::worker_entry)
        }
        .expect("worker stack too small");
        let top = stack.as_ptr() as u64 + stack.len() as u64;
        let _ = scheduler.add_task("worker", WORKER_PRIORITY, rsp, top, stack.len(), tasks::worker_entry);
        // The kernel itself is the current task from here on.
        scheduler.switch_to(TaskId(0));
        scheduler
            .tasks()
            .iter()
            .map(|t| (t.id, t.name, t.state, t.stack_bottom(), t.stack_top))
            .collect::<alloc::vec::Vec<_>>()
    };

    // Prints happen with interrupts still off (see the doc comment):
    // the first switch must not preempt idle before the task list
    // has landed in the log.
    serial_println!("tasks: {} created", snapshot.len());
    println!("tasks: {} created", snapshot.len());
    for (id, name, state, bottom, top) in snapshot {
        if id.0 == 0 {
            serial_println!(
                "task: pid {} name={} state={:?} stack boot (untracked)",
                id.0, name, state
            );
            println!(
                "task: pid {} name={} state={:?} stack boot (untracked)",
                id.0, name, state
            );
        } else {
            serial_println!(
                "task: pid {} name={} state={:?} stack {:#x}-{:#x}",
                id.0, name, state, bottom, top
            );
            println!(
                "task: pid {} name={} state={:?} stack {:#x}-{:#x}",
                id.0, name, state, bottom, top
            );
        }
    }

    if were_enabled {
        x86_64::instructions::interrupts::enable();
    }
}

/// Create a ring-3 (userspace) task for a loaded program and add it
/// to the scheduler: allocate its 16 KiB kernel stack, lay out the
/// iretq frame on it (so the first switch `ret`s into the iretq
/// stub and drops to ring 3 at `entry_rip` with `rsp = user_rsp`),
/// and register the task. Returns the assigned pid.
///
/// `entry_rip` is the program's ELF entry point and `user_rsp` its
/// initial user stack pointer (both from
/// [`crate::userspace::load_user_program`]). The kernel stack is
/// the task's privilege-transition stack: `TSS.rsp[0]` points at
/// its top whenever the task runs, so interrupts and syscalls from
/// ring 3 land on it (updated on every switch — see
/// [`set_kernel_stack_for`]). The stack is `Box::leak`ed and never
/// freed (no reaper yet), like the worker's.
///
/// Interrupts are disabled while the scheduler is populated, for
/// the same reason as in [`init_tasks`]: the timer IRQ takes the
/// same `SCHEDULER` lock, and a tick landing mid-setup would spin
/// on it while the holder sits suspended underneath the handler.
/// Prints the task line (marked `(user)`) to serial and VGA.
pub fn add_user_task(
    name: &'static str,
    priority: u8,
    entry_rip: u64,
    user_rsp: u64,
) -> TaskId {
    use alloc::boxed::Box;

    let were_enabled = x86_64::instructions::interrupts::are_enabled();
    x86_64::instructions::interrupts::disable();

    let id = {
        let mut scheduler = SCHEDULER.lock();
        let stack = Box::leak(Box::new([0u8; tasks::USER_KERNEL_STACK_SIZE]));
        let frame = Ring3Frame::new(
            entry_rip,
            user_rsp,
            gdt::USER_CODE_INDEX,
            gdt::USER_DATA_INDEX,
        );
        let rsp = prepare_ring3_stack(stack, frame).expect("user kernel stack too small");
        let top = stack.as_ptr() as u64 + stack.len() as u64;
        scheduler.add_user_task(name, priority, rsp, top, stack.len(), entry_rip)
    };

    serial_println!(
        "task: pid {} name={} state=Ready (user) entry={:#x} user_rsp={:#x}",
        id.0, name, entry_rip, user_rsp
    );
    println!(
        "task: pid {} name={} state=Ready (user) entry={:#x} user_rsp={:#x}",
        id.0, name, entry_rip, user_rsp
    );

    if were_enabled {
        x86_64::instructions::interrupts::enable();
    }
    id
}

/// Pop the oldest queued key event, or `None` when the ring is empty.
pub fn pop_key_event() -> Option<KeyEvent> {
    KEYBOARD_IRQ.lock().pop_event()
}

/// Iterator over decoded keyboard characters, for `read(fd 0)`:
/// pops key events and yields their characters, skipping
/// modifier/release events (which carry no character). Bounded:
/// the ring holds at most 64 events, so `next()` always terminates
/// with `None` once the ring is empty.
pub struct KeyCharIter;

impl Iterator for KeyCharIter {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        loop {
            match pop_key_event() {
                Some(event) => match event.ch {
                    Some(ch) => return Some(ch),
                    None => continue,
                },
                None => return None,
            }
        }
    }
}

/// True when at least one ring-3 (userspace) task is still alive
/// (any state but `Terminated`). The idle loop uses this to decide
/// who owns the keyboard: while a userspace program exists, its
/// `read(fd 0)` is the only consumer of stdin — the idle loop must
/// not drain or echo keystrokes itself, or it would steal bytes the
/// program is waiting for and echo them twice (Batch 9). Once every
/// user task has exited, the idle loop resumes echoing so the
/// console stays live. Kernel tasks never count: the worker does
/// not read stdin, so it must not silence the echo.
pub fn has_live_user_task() -> bool {
    SCHEDULER.lock().has_live_user_task()
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

/// Handle one timer IRQ (vector 32): advance the PIT tick counter
/// and run the scheduler tick at each quantum boundary. If the tick
/// says `Preempt(next)`, switch to `next` **here, inside the
/// handler**: the worker never runs the idle loop, so deferring the
/// switch there (the old `NEED_RESCHED` design, removed in Batch 7)
/// could never preempt it back to idle. EOI goes out *before* any
/// switch: the PIC holds IRQ0 in service until EOI, and the handler
/// suspends mid-flight when it switches — an EOI after the switch
/// would not run until the task resumes, wedging the timer.
fn handle_timer_irq() {
    let outcome = {
        let mut timer = TIMER_IRQ.lock();
        let mut scheduler = SCHEDULER.lock();
        let outcome = timer_irq_step(&mut timer, &mut scheduler);
        // One IRQ handled == one tick counted. Mirror it into the
        // lock-free counter here, next to the `PitTimer` tick, so
        // the idle loop can read it without ever taking this lock.
        TOTAL_TICKS.fetch_add(1, Ordering::Relaxed);
        outcome
    };
    send_eoi(TIMER_VECTOR);
    if let Some(TickOutcome::Preempt(next)) = outcome {
        switch_to_task(next);
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

extern "x86-interrupt" fn timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
    handle_timer_irq();
}

extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    handle_keyboard_irq();
    send_eoi(KEYBOARD_VECTOR);
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
}

/// True when the code segment in `frame` has RPL 3 — the fault
/// happened in a ring-3 (userspace) task. The CPU pushes the
/// faulting CS into the frame, so this is the one reliable
/// discriminator: a ring-0 fault has CS 0x08 (RPL 0), a ring-3
/// fault has CS 0x2B (RPL 3). Pure over the selector, so host tests
/// can pin it without building a whole frame.
fn is_ring3_selector(cs: SegmentSelector) -> bool {
    cs.rpl() == PrivilegeLevel::Ring3
}

/// A fault from a ring-3 task: report it (vector, pid, RIP, and
/// the faulting address / error code where they exist) and
/// terminate ONLY that task (exit code -1), then switch to the
/// next Ready task. The kernel keeps running — a userspace program
/// that touches a bad address or executes a bad instruction must
/// never halt the kernel, let alone triple-fault it.
///
/// Never returns to the fault handler: the faulting task is dead
/// and its frame is abandoned on its kernel stack (the same
/// abandonment as the normal `exit` path — see
/// [`exit_current_task`]). Returns only via `hlt_loop` when there
/// is nothing left to switch to: `exit_current_task` already
/// terminated the task, so there is nothing to return to.
fn kill_user_task(
    name: &str,
    frame: &InterruptStackFrame,
    fault_addr: Option<u64>,
    error_code: Option<u64>,
) -> ! {
    let pid = current_task_id().map(|id| id.0).unwrap_or(u64::MAX);
    println!(
        "EXCEPTION: {} (user task pid {} rip {:#x}) — killing task",
        name,
        pid,
        frame.instruction_pointer.as_u64()
    );
    serial_println!(
        "EXCEPTION: {} (user task pid {} rip {:#x}) — killing task",
        name,
        pid,
        frame.instruction_pointer.as_u64()
    );
    if let Some(addr) = fault_addr {
        println!("Accessed Address: {:#x}", addr);
        serial_println!("Accessed Address: {:#x}", addr);
    }
    if let Some(code) = error_code {
        println!("Error Code: {:#x}", code);
        serial_println!("Error Code: {:#x}", code);
    }
    exit_current_task(-1);
    // No Ready task left to switch to (or no current task): the
    // faulting task is already terminated, so halt rather than
    // iretq-ing into a dead task.
    println!("FAULT: no tasks left after killing pid {} — halting", pid);
    serial_println!("FAULT: no tasks left after killing pid {} — halting", pid);
    hlt_loop();
}

/// Report an exception on VGA and serial, then halt. The `hlt_loop`
/// at the end never returns, so this is the body for every fault
/// handler except breakpoint (which returns) and double fault (which
/// panics with the frame instead).
fn report_exception(name: &str, error_code: Option<u64>, stack_frame: &InterruptStackFrame) -> ! {
    // A fault in a userspace program kills only that program (see
    // kill_user_task). A fault in the kernel still halts: a ring-0
    // fault means the kernel itself is broken, and there is nothing
    // safe to resume.
    if is_ring3_selector(stack_frame.code_segment) {
        kill_user_task(name, stack_frame, None, error_code);
    }
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
    // A #GP in a userspace program kills only that program (the
    // error code names the bad selector — it is printed by the kill
    // path). A #GP in the kernel still halts, with the decoded
    // error code below.
    if is_ring3_selector(stack_frame.code_segment) {
        kill_user_task(
            "GENERAL PROTECTION FAULT",
            &stack_frame,
            None,
            Some(error_code),
        );
    }
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

    // A #PF in a userspace program kills only that program (the
    // faulting address is CR2 — read_raw, not read: the faulting
    // address can be non-canonical, which is exactly why it
    // faulted, and read() would Err on it). A #PF in the kernel
    // still halts.
    if is_ring3_selector(stack_frame.code_segment) {
        kill_user_task(
            "PAGE FAULT",
            &stack_frame,
            Some(Cr2::read_raw()),
            Some(error_code.bits()),
        );
    }
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
    use rustos_scheduler::{Scheduler, Task, TaskId, TaskState};

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
    fn kernel_task_priorities_schedule_the_idle_task() {
        // The kernel creates idle at IDLE_PRIORITY, worker at
        // WORKER_PRIORITY, and userspace tasks at 5 (the priority
        // passed to `add_user_task`). With those priorities the
        // round-robin must visit idle inside one full rotation: if
        // IDLE_PRIORITY ever drops below 5 again, this test fails —
        // that is the Batch 8 wedge, where idle at 0 starved while
        // worker and hello (both 5, always Ready) ran, so
        // `kernel_main` never resumed and the boot log stopped after
        // `task: pid 2 name=hello ...`.
        let mut sched = Scheduler::new();
        let mut ids = Vec::new();
        for (name, prio) in [
            ("idle", IDLE_PRIORITY),
            ("worker", WORKER_PRIORITY),
            ("hello", 5u8),
        ] {
            let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 8192]));
            let rsp = unsafe { Task::prepare_stack(buf.as_mut_ptr(), buf.len(), idle_entry) }
                .expect("stack");
            let top = buf.as_ptr() as u64 + buf.len() as u64;
            ids.push(sched.add_task(name, prio, rsp, top, buf.len(), idle_entry));
        }
        let mut seen_idle = false;
        for _ in 0..3 {
            if sched.run_next() == Some(ids[0]) {
                seen_idle = true;
            }
        }
        assert!(seen_idle, "idle task never scheduled in one full rotation");
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
    fn yield_sequence_picks_next_and_switches_bookkeeping() {
        // The decision sequence scheduler_yield() performs, on a
        // local scheduler: pick_next (Ready only, so the Running
        // current task is excluded), context_switch BEFORE switch_to
        // (it only hands out pointers for Ready tasks), then
        // switch_to. The register switch itself cannot run on the
        // host (it would jump to a fabricated stack); the box proves
        // that half.
        let (mut sched, a, b) = sched_with_two_running();
        assert_eq!(sched.current_id(), Some(a));
        let next = sched.pick_next().expect("b is Ready");
        assert_eq!(next, b);
        let (from, to) = unsafe { sched.context_switch(next) }.expect("pointers");
        assert!(!from.is_null());
        assert!(!to.is_null());
        sched.switch_to(next);
        assert_eq!(sched.current_id(), Some(b));
        assert_eq!(sched.task(a).unwrap().state, TaskState::Ready);
        assert_eq!(sched.task(b).unwrap().state, TaskState::Running);
    }

    #[test]
    fn exit_sequence_terminates_current_and_picks_next() {
        // The decision sequence exit_current_task() performs: exit
        // the current task (Terminated, never scheduled again), pick
        // the next Ready task, switch_to it. The enter() jump itself
        // cannot run on the host.
        let (mut sched, a, b) = sched_with_two_running();
        assert!(sched.exit_current(0));
        assert_eq!(sched.task(a).unwrap().state, TaskState::Terminated);
        assert_eq!(sched.current_id(), None);
        let next = sched.pick_next().expect("b is Ready");
        assert_eq!(next, b);
        sched.switch_to(next);
        assert_eq!(sched.current_id(), Some(b));
        // The terminated task is never picked again.
        assert_eq!(sched.pick_next(), None);
    }

    #[test]
    fn timer_and_keyboard_state_start_empty() {
        assert_eq!(timer_ticks(), 0);
        assert_eq!(pop_key_event(), None);
        assert_eq!(mode(), ControllerMode::Pic);
    }

    #[test]
    fn timer_ticks_reads_the_lock_free_counter() {
        // The idle loop reads this with interrupts enabled, so it
        // must be the atomic, never the `TIMER_IRQ` lock: a tick
        // landing while the loop held the lock would spin in the
        // handler (interrupts off) forever. This pins the wiring —
        // it holds by construction for the atomic read, whatever
        // order the step tests ran in (they drive local timers,
        // never the global counter).
        assert_eq!(timer_ticks(), TOTAL_TICKS.load(Ordering::Relaxed));
    }

    #[test]
    fn uptime_due_fires_every_500_ticks() {
        // The idle loop's uptime condition, pinned at the boundary:
        // quiet before 500 ticks, fires at exactly 500, and fires
        // again 500 after the last report. A long stall still fires
        // once (it does not spam one line per missed interval —
        // the loop updates `last_printed` to the current tick).
        assert!(!uptime_due(0, 0));
        assert!(!uptime_due(499, 0));
        assert!(uptime_due(500, 0));
        assert!(uptime_due(501, 0));
        assert!(!uptime_due(999, 500));
        assert!(uptime_due(1000, 500));
        assert!(uptime_due(2500, 500));
        assert_eq!(UPTIME_INTERVAL_TICKS, 500);
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
            format!("{:?}", idt[SYSCALL_VECTOR]),
        ];
        assert_eq!(gates.len(), 26);
        for gate in &gates {
            assert!(gate.contains("index: 1"), "gate lost its code selector: {gate}");
            assert!(gate.contains("present: true"), "gate not present: {gate}");
            assert!(
                !gate.contains("handler_addr: 0x0"),
                "gate has no handler: {gate}"
            );
        }
        // The syscall gate is the only user-callable one: DPL 3, and
        // an interrupt gate (type 0b1110 — interrupts cleared on
        // entry), so the yield path cannot deadlock against the timer
        // IRQ on the scheduler lock.
        let syscall_gate = gates.last().unwrap();
        assert!(
            syscall_gate.contains("privilege_level: Ring3"),
            "syscall gate is not DPL 3: {syscall_gate}"
        );
        assert!(
            syscall_gate.contains("type: 0b1110"),
            "syscall gate is not an interrupt gate: {syscall_gate}"
        );
        // The double-fault handler keeps its dedicated IST stack.
        assert!(
            gates[8].contains("stack_index: Some(0)"),
            "double fault lost its IST stack: {}",
            gates[8]
        );
    }

    #[test]
    fn ring3_selector_check_reads_the_rpl() {
        // The fault discriminator: a ring-3 fault's CS carries RPL 3
        // (0x2B), a ring-0 fault's CS carries RPL 0 (0x08). The
        // index does not matter — only the low two bits.
        use x86_64::structures::gdt::SegmentSelector;
        assert!(is_ring3_selector(SegmentSelector::new(5, PrivilegeLevel::Ring3)));
        assert!(is_ring3_selector(SegmentSelector::new(6, PrivilegeLevel::Ring3)));
        assert!(!is_ring3_selector(SegmentSelector::new(1, PrivilegeLevel::Ring0)));
        assert!(!is_ring3_selector(SegmentSelector::new(5, PrivilegeLevel::Ring0)));
        // The exact values the CPU pushes: user CS 0x2B, kernel CS 0x08.
        assert_eq!(SegmentSelector::new(5, PrivilegeLevel::Ring3).0, 0x2b);
        assert_eq!(SegmentSelector::new(1, PrivilegeLevel::Ring0).0, 0x08);
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
