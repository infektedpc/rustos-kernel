//! Kernel tasks: the idle task (pid 0) and the worker task (pid 1).
//!
//! The idle task is the kernel itself — `kernel_main`'s continuation
//! on the boot stack. It has a task record so the scheduler can
//! switch back to it, but its entry point is never called: the record
//! exists for the scheduler, not as a fresh start.
//!
//! The worker task is the first real task: its own 16 KiB kernel
//! stack, laid out by [`rustos_scheduler::Task::prepare_stack`] so
//! the first `switch()` to it `ret`s into [`worker_entry`]. It runs
//! in ring 0. Ring-3 tasks (Batch 8) run a userspace program in
//! ring 3 but still own a kernel stack of the same size: it is the
//! stack `TSS.rsp[0]` points at while they run, so interrupts and
//! syscalls from ring 3 land on it (see
//! `crate::interrupts::add_user_task`).

// Only the real (non-test) worker entry prints or issues syscalls;
// the host-test stand-in below uses neither.
#[cfg(not(test))]
use crate::{println, serial_println, syscall};

/// Size of the worker task's kernel stack, in bytes.
pub const WORKER_STACK_SIZE: usize = 16 * 1024;

/// Size of a ring-3 task's kernel stack, in bytes — the stack
/// `TSS.rsp[0]` points at while the task runs in ring 3 (see
/// `crate::interrupts::add_user_task`). Same size as the worker's:
/// the stack only ever holds interrupt/syscall frames, never the
/// program's own data.
pub const USER_KERNEL_STACK_SIZE: usize = 16 * 1024;

/// How often the worker prints its counter (every N runs). At 100 Hz
/// with a 1-tick quantum and a yield per run, one run is ~20 ms
/// (one worker quantum + one idle quantum), so 100 runs is ~2 s —
/// often enough to prove alternation, rare enough not to flood the
/// log.
pub const WORKER_PRINT_EVERY: u64 = 100;

/// Entry point of the idle task. Never entered as a fresh task: the
/// idle task IS `kernel_main`'s continuation, so its entry only
/// exists as a placeholder in the task record. If it were ever
/// entered, halting forever is the safe behaviour — returning would
/// `ret` into whatever sits above the (empty) stack. (Declared
/// `fn()`, not `fn() -> !`, because the scheduler's `TaskEntry` is
/// `fn()`; the loop below never returns.)
pub fn idle_entry() {
    loop {
        x86_64::instructions::hlt();
    }
}

/// Entry point of the worker task (pid 1).
///
/// First entry runs with interrupts OFF: the timer IRQ's interrupt
/// gate cleared `IF`, and the switch into this task has no interrupt
/// frame to restore `IF` from — so `sti` is the first thing it does.
/// (A resumed worker returns through the IRQ handler's `iretq`,
/// which restores `IF` from its own saved `RFLAGS`, so only the
/// first entry needs the explicit `sti`.)
///
/// After that it loops: count runs, print every 100th run (pid from
/// the `getpid` syscall, proving the syscall returns the running
/// task's real pid), then yield through the `yield` syscall (proving
/// `yield` switches to the next task). It never returns: a task entry
/// that returned would `ret` into the padding slot above its stack
/// (address 0), so the loop is infinite by construction and the only
/// way out is the `exit` syscall. (Declared `fn()`, like the idle
/// entry, because the scheduler's `TaskEntry` is `fn()`.)
#[cfg(not(test))]
pub fn worker_entry() {
    x86_64::instructions::interrupts::enable();
    serial_println!("worker: started (pid 1)");
    println!("worker: started (pid 1)");
    let mut runs: u64 = 0;
    loop {
        runs += 1;
        if runs % WORKER_PRINT_EVERY == 0 {
            let pid = syscall::invoke_syscall(syscall::SYS_GETPID, 0, 0, 0);
            serial_println!("worker: run {} (pid {})", runs, pid);
            println!("worker: run {} (pid {})", runs, pid);
        }
        let _ = syscall::invoke_syscall(syscall::SYS_YIELD, 0, 0, 0);
    }
}

/// Host-test stand-in: host tests never run tasks, but the module
/// must still compile for the host target, where `int 0x80` would
/// hit the Linux 32-bit compat entry instead of this kernel.
#[cfg(test)]
pub fn worker_entry() {
    loop {
        core::hint::spin_loop();
    }
}
