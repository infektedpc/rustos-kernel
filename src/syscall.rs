//! Syscalls into the kernel: vector `0x80`, the `rustos-libc` convention.
//!
//! ## Numbers and registers
//!
//! The syscall numbers are **not** defined here: they are re-exported
//! from `rustos_process::syscall`, whose numbers were verified
//! firsthand to be identical to `rustos-libc`'s (`read` 0, `write` 1,
//! `open` 2, `close` 3, `yield` 24, `getpid` 39, `spawn` 57, `exit` 60,
//! `wait` 61, `uptime` 1000 — the last is RustOS-private, added in
//! Batch 9 for the shell). The kernel therefore cannot drift from libc — the
//! numbers come from the same constants the userspace wrappers use.
//!
//! The register convention is libc's, not Linux's `int 0x80` one:
//! `rax` = number, `rdi` = a0, `rsi` = a1, `rdx` = a2, result in `rax`,
//! negative errno on error. (libc's `RealBackend` issues the `syscall`
//! *instruction* with the same registers — the fast entry is
//! [`crate::fast_syscall`]; this module is the `int 0x80` gate
//! into the same dispatcher.)
//!
//! ## The entry path
//!
//! `int 0x80` lands in the [`syscall_entry`] assembly stub (registered
//! as the IDT gate for vector `0x80` in `crate::interrupts`). The stub
//! saves all general-purpose registers, moves the trap's register
//! arguments into the C ABI order, calls [`rustos_syscall_entry`]
//! (the Rust dispatcher below), writes the return value into the
//! saved `rax` slot, restores the registers, and `iretq`s. The stub
//! works for callers in any ring: from ring 0 the CPU pushes only
//! `RFLAGS`/`CS`/`RIP`; from ring 3 it also switches stacks and
//! pushes `SS`/`RSP` — the stub only touches the registers above
//! whatever the CPU pushed.
//!
//! The gate is an **interrupt gate** (interrupts cleared on entry),
//! not a trap gate, even though the task asked for a trap gate: the
//! `yield` path locks the same `SCHEDULER` mutex the timer IRQ takes,
//! and a trap gate would let a timer IRQ land mid-`yield` and spin
//! forever on that lock. With an interrupt gate the syscall runs
//! atomically against IRQs and `iretq` restores `IF` on the way out.
//! Linux's own `int 0x80` gate is an interrupt gate for the same
//! reason. DPL is 3, so userspace can invoke it directly once it
//! exists — but see the ring-3 note below.
//!
//! ## What is implemented (Batch 6)
//!
//! * `write` (1): fd 1 -> serial + VGA, fd 2 -> serial only.
//!   Other fds are `-EBADF`. The whole buffer is written (no short
//!   writes yet); the return is the byte count.
//! * `read` (0): fd 0 -> stdin: serial input (COM1) first, then
//!   the PS/2 key ring (see [`fill_stdin_buffer`]). **Non-blocking**:
//!   it returns the bytes already waiting (up to `len`), or
//!   `-EAGAIN` when both sources are empty — it never blocks, because
//!   there are no tasks to sleep yet. Other fds are `-EBADF`. This
//!   deliberately differs from `rustos-process`'s model dispatch,
//!   which returns 0 (EOF) for stdin: the real kernel has live
//!   serial input, so EOF would be a lie.
//! * `uptime` (1000): milliseconds since boot (PIT ticks * 10 at
//!   100 Hz). RustOS-private number — Linux has no uptime syscall
//!   (see `rustos-process`). Added in Batch 9 for the shell's
//!   `uptime` builtin.
//! * `exit` (60): terminates the *current task* with the given
//!   code: the task is marked `Terminated` (never scheduled again)
//!   and the kernel switches to the next Ready task. With no tasks
//!   yet the exiting context *is* the kernel, so there is nowhere
//!   to switch to; the halt is the honest behaviour, not a stub —
//!   and exiting pid 0 (the kernel itself) always halts, even once
//!   tasks exist.
//! * `getpid` (39): returns the running task's real pid — 0 for the
//!   idle task (the kernel itself), 1 for the worker, and so on.
//!   Before any task exists it is 0, as before.
//! * `yield` (24): switches to the next Ready task now (a real
//!   context switch, not just a tick). With no tasks the tick just
//!   counts and returns `Continue`, as before; with nothing else
//!   Ready the caller keeps the CPU with a fresh quantum.
//!
//! The other defined numbers — `open` (2), `close` (3), `spawn` (57),
//! `wait` (61) — return `-ENOSYS`. That is deliberate: there is no
//! filesystem mounted for `open`, no open files for `close`, and no
//! user processes for `spawn`/`wait`. Faking success for any of them
//! would be a stub; `-ENOSYS` says plainly "not implemented yet".
//! Unknown numbers also return `-ENOSYS`, never a fault.
//!
//! Pointer validation: a null buffer is `-EFAULT`, a length over 1
//! MiB is `-EINVAL`, and — since Batch 8 — a buffer from a ring-3
//! caller is checked against the mapped user regions (see
//! [`crate::userspace::is_user_buffer`]): a userspace program that
//! passes an address outside its own mapped segments, stack, or
//! heap gets `-EFAULT`, not a kernel `#PF` (the dispatcher would
//! fault in ring 0 dereferencing it, and a ring-0 fault halts the
//! whole kernel). Kernel callers (the self-test, the worker) pass
//! kernel pointers, which are never checked this way.
//!
//! ## Ring 3 (Batch 8 — done)
//!
//! The gate's DPL is 3, and a `int 0x80` from ring 3 now works: the
//! CPU switches to the kernel stack from `TSS.rsp[0]` on a
//! privilege change, and `rsp[0]` is set to the running task's
//! kernel stack top on every switch (see
//! `crate::interrupts::set_kernel_stack_for`). The stub works for
//! callers in any ring (see the entry-path note above), and the
//! dispatcher validates user buffers as described above.
//!
//! The `syscall` instruction is wired since Batch 11 (see
//! [`crate::fast_syscall`]): the MSR entry (`EFER.SCE`, `LSTAR`,
//! `STAR`, `SFMASK`) plus a `swapgs` stack switch is programmed
//! at boot, and `rustos-libc`'s `RealBackend` is the default
//! backend now. `int 0x80` stays as the fallback: the gate is
//! still DPL 3, the dispatcher is shared, and the boot self-test
//! exercises both paths (the `int 0x80` results directly, the
//! fast path by reading its MSRs back — a `syscall` issued from
//! ring 0 would `sysret` back into ring 3, so kernel code must
//! never issue it).

use x86_64::VirtAddr;

/// The syscall trap vector: `int 0x80`.
pub const SYSCALL_VECTOR: u8 = 0x80;

/// Syscall numbers, re-exported from `rustos-process` so they can
/// never drift from `rustos-libc` (verified identical firsthand).
pub use rustos_process::syscall::{
    FD_STDERR, FD_STDIN, FD_STDOUT, SYS_CLOSE, SYS_EXIT, SYS_GETPID, SYS_OPEN, SYS_READ,
    SYS_SPAWN, SYS_UPTIME, SYS_WAIT, SYS_WRITE, SYS_YIELD,
};
/// Errnos, as the negative values the kernel actually returns.
/// (`rustos-process` defines only the four it needs; `EFAULT` comes
/// from `rustos-libc`'s `Error::FAULT` = 14.)
pub use rustos_process::syscall::{ERR_AGAIN, ERR_BADF, ERR_INVAL, ERR_NOSYS};

/// EFAULT: bad address (null buffer). `-14`, matching libc.
pub const ERR_FAULT: i64 = -14;

/// Maximum buffer length for `read`/`write`. Larger is `-EINVAL`
/// (same 1 MiB cap as `rustos-process`'s model dispatch).
pub const MAX_SYSCALL_LEN: usize = 1 << 20;

/// Pid of the kernel itself (the idle process in the process table).
/// `getpid` returns this until the first user process exists.
pub const KERNEL_PID: u32 = 0;

/// What a syscall did. `Returned` carries the value that goes back
/// in the caller's `rax`; `Exited` means the calling context is
/// finished and must never be returned to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyscallOutcome {
    Returned(i64),
    Exited(i32),
}

/// The devices a syscall touches, behind a trait so the dispatcher
/// is host-testable: the real kernel uses [`KernelDevices`], the
/// tests use a fake.
pub trait SyscallDevices {
    /// Write `bytes` to stdout (fd 1); returns bytes written.
    fn write_stdout(&mut self, bytes: &[u8]) -> usize;
    /// Write `bytes` to stderr (fd 2); returns bytes written.
    fn write_stderr(&mut self, bytes: &[u8]) -> usize;
    /// Read up to `buf.len()` bytes from stdin (fd 0), non-blocking:
    /// `Ok(n)` with `n` bytes read, or `Err(-EAGAIN)` when nothing is
    /// waiting. Never blocks.
    fn read_stdin(&mut self, buf: &mut [u8]) -> Result<usize, i64>;
    /// Milliseconds since boot (PIT ticks * 10 at 100 Hz).
    fn uptime_ms(&self) -> u64;
    /// Pid of the calling context.
    fn current_pid(&self) -> u32;
    /// True when the calling context is a ring-3 (userspace) task:
    /// its buffers are user pointers and must be validated against
    /// the mapped user regions (see [`checked_buf`]). False for
    /// kernel callers, whose buffers are kernel pointers.
    fn is_user_caller(&self) -> bool;
    /// Voluntary yield: switch to the next Ready task now. May not
    /// return until the task is scheduled again (the switch jumps
    /// away and the caller resumes later, in the new task's wake).
    /// With no tasks, or nothing else Ready, just returns.
    fn yield_now(&mut self);
}

/// Fill `buf` from the two stdin sources, in order: serial bytes
/// first (the live path — QEMU `-serial stdio` delivers keystrokes
/// to COM1), then decoded PS/2 key-ring characters (the path when
/// the machine has a display). Pure over its sources, so host tests
/// can drive it with a fake serial closure and a peekable char
/// iterator; the real kernel passes `try_read_byte` and the key
/// ring's [`crate::interrupts::KeyCharIter`].
///
/// Returns `Ok(n)` with `n` bytes filled (at least 1), or
/// `Err(-EAGAIN)` when both sources are empty. Never blocks. A
/// multi-byte PS/2 character is only taken when its whole UTF-8
/// encoding fits — the key source is peeked, never consumed
/// speculatively, so a character that does not fit stays queued for
/// the next call instead of being dropped half-written.
pub fn fill_stdin_buffer<I>(
    buf: &mut [u8],
    mut serial_byte: impl FnMut() -> Option<u8>,
    key_chars: &mut core::iter::Peekable<I>,
) -> Result<usize, i64>
where
    I: Iterator<Item = char>,
{
    let mut n = 0;
    for slot in buf.iter_mut() {
        match serial_byte() {
            Some(b) => {
                *slot = b;
                n += 1;
            }
            None => break,
        }
    }
    // Serial empty (or the buffer full): top up from the key ring.
    while n < buf.len() {
        let Some(&ch) = key_chars.peek() else { break };
        let mut tmp = [0u8; 4];
        let encoded = ch.encode_utf8(&mut tmp);
        if n + encoded.len() > buf.len() {
            break;
        }
        key_chars.next();
        buf[n..n + encoded.len()].copy_from_slice(encoded.as_bytes());
        n += encoded.len();
    }
    if n == 0 {
        Err(ERR_AGAIN)
    } else {
        Ok(n)
    }
}

/// Validate a syscall length: over the cap is `-EINVAL`.
fn checked_len(len: u64) -> Result<usize, i64> {
    if len > MAX_SYSCALL_LEN as u64 {
        Err(ERR_INVAL)
    } else {
        Ok(len as usize)
    }
}

/// Validate a syscall input pointer: null is `-EFAULT`; for a
/// ring-3 caller (`user` is true) the buffer must also lie wholly
/// inside one mapped user region (see
/// [`crate::userspace::is_user_buffer`]) — otherwise `-EFAULT`,
/// before anything is dereferenced. Kernel callers pass kernel
/// pointers, which are never region-checked.
fn checked_buf(ptr: u64, len: usize, user: bool) -> Result<&'static [u8], i64> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr == 0 {
        return Err(ERR_FAULT);
    }
    if user && !crate::userspace::is_user_buffer(ptr, len) {
        return Err(ERR_FAULT);
    }
    Ok(unsafe { core::slice::from_raw_parts(ptr as *const u8, len) })
}

/// Validate a syscall output pointer: null is `-EFAULT`; for a
/// ring-3 caller the buffer must also lie inside a mapped user
/// region (same rule as [`checked_buf`]).
fn checked_buf_mut(ptr: u64, len: usize, user: bool) -> Result<&'static mut [u8], i64> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr == 0 {
        return Err(ERR_FAULT);
    }
    if user && !crate::userspace::is_user_buffer(ptr, len) {
        return Err(ERR_FAULT);
    }
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr as *mut u8, len) })
}

/// Dispatch one syscall: `number` with register arguments `a0`, `a1`,
/// `a2`, all effects through `devices`. Pure except for the device
/// calls, so host tests drive it with a fake device set.
///
/// Return-value convention: `Returned(n)` with `n >= 0` on success,
/// `Returned(-errno)` on error, `Exited(code)` for `exit`.
pub fn dispatch(
    number: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    devices: &mut impl SyscallDevices,
) -> SyscallOutcome {
    use SyscallOutcome::*;
    let user = devices.is_user_caller();
    match number {
        SYS_READ => {
            if a0 != FD_STDIN {
                return Returned(ERR_BADF);
            }
            let len = match checked_len(a2) {
                Ok(l) => l,
                Err(e) => return Returned(e),
            };
            if len == 0 {
                return Returned(0);
            }
            let buf = match checked_buf_mut(a1, len, user) {
                Ok(b) => b,
                Err(e) => return Returned(e),
            };
            match devices.read_stdin(buf) {
                Ok(n) => Returned(n as i64),
                Err(e) => Returned(e),
            }
        }
        SYS_WRITE => {
            let to_stdout = match a0 {
                FD_STDOUT => true,
                FD_STDERR => false,
                _ => return Returned(ERR_BADF),
            };
            let len = match checked_len(a2) {
                Ok(l) => l,
                Err(e) => return Returned(e),
            };
            if len == 0 {
                return Returned(0);
            }
            let bytes = match checked_buf(a1, len, user) {
                Ok(b) => b,
                Err(e) => return Returned(e),
            };
            let n = if to_stdout {
                devices.write_stdout(bytes)
            } else {
                devices.write_stderr(bytes)
            };
            Returned(n as i64)
        }
        SYS_YIELD => {
            devices.yield_now();
            Returned(0)
        }
        SYS_GETPID => Returned(devices.current_pid() as i64),
        SYS_UPTIME => Returned(devices.uptime_ms() as i64),
        SYS_EXIT => Exited(a0 as i32),
        // Defined in the ABI but not implemented by this kernel yet:
        // open (no filesystem mounted), close (no open files), spawn
        // and wait (no user processes yet). -ENOSYS, never fake success.
        SYS_OPEN | SYS_CLOSE | SYS_SPAWN | SYS_WAIT => Returned(ERR_NOSYS),
        _ => Returned(ERR_NOSYS),
    }
}

/// The real device set: COM1 serial, the VGA text buffer, and the
/// scheduler. Unit struct — all state lives in the crate's statics.
pub struct KernelDevices;

impl KernelDevices {
    /// Write `bytes` to serial, and to VGA as well when `to_vga`.
    /// Lock order is serial first, then VGA, then the framebuffer
    /// TTY — no other path in the kernel nests the locks the other
    /// way round. Each lock is taken under
    /// [`crate::interrupts::IrqGuard`]: the syscall handler runs
    /// with interrupts off (interrupt gate), so no IRQ can
    /// interleave between the two locks — but the hazard is the
    /// other direction. A ring-0 task printing with interrupts on
    /// can be preempted mid-print holding a console lock, and the
    /// ring-3 task the tick switches to then spins on that lock
    /// with interrupts off, forever (Batch 8 wedge). The guard
    /// makes every console-lock holder non-preemptible, so a
    /// syscall handler can never find a console lock held by a
    /// suspended task.
    fn write_console(bytes: &[u8], to_vga: bool) -> usize {
        {
            let _irq = crate::interrupts::IrqGuard::hold();
            let mut port = crate::serial::SERIAL1.lock();
            for &b in bytes {
                port.send(b);
            }
        }
        if to_vga {
            let _irq = crate::interrupts::IrqGuard::hold();
            if crate::vga::vga_available() {
                let mut writer = crate::vga::WRITER.lock();
                for &b in bytes {
                    writer.write_byte(b);
                }
            }
            // Framebuffer console mirrors stdout (Batch 10): the
            // shell's and hello's output renders through the TTY
            // layer too, so a display boot shows it. (fd 2/stderr
            // stays serial-only, as before.) The TTY interprets
            // escape sequences; VGA just draws them as blocks.
            crate::fbcon::write_bytes(bytes);
        }
        bytes.len()
    }
}

impl SyscallDevices for KernelDevices {
    fn write_stdout(&mut self, bytes: &[u8]) -> usize {
        Self::write_console(bytes, true)
    }

    fn write_stderr(&mut self, bytes: &[u8]) -> usize {
        Self::write_console(bytes, false)
    }

    fn read_stdin(&mut self, buf: &mut [u8]) -> Result<usize, i64> {
        // Non-blocking: serial bytes first, then PS/2 key-ring
        // characters (see `fill_stdin_buffer`). The idle loop stops
        // draining both once a userspace task exists (see
        // `has_live_user_task`), so this is the only consumer — a
        // byte is read once, never echoed twice.
        let mut keys = crate::interrupts::KeyCharIter.peekable();
        fill_stdin_buffer(buf, crate::serial::try_read_byte, &mut keys)
    }

    fn uptime_ms(&self) -> u64 {
        crate::interrupts::timer_ticks() * 10
    }

    fn current_pid(&self) -> u32 {
        crate::interrupts::current_task_id()
            .map(|id| id.0 as u32)
            .unwrap_or(KERNEL_PID)
    }

    fn is_user_caller(&self) -> bool {
        crate::interrupts::current_task_is_user()
    }

    fn yield_now(&mut self) {
        crate::interrupts::scheduler_yield();
    }
}

/// The Rust half of the vector-0x80 handler, called from the
/// [`syscall_entry`] assembly stub with the trap's register arguments.
/// Returns the value that goes back in the caller's `rax`.
///
/// Never returns for `exit`: the calling task is terminated and the
/// kernel switches to the next Ready task (which never returns here
/// either — see `exit_current_task`). Exiting pid 0 (the kernel
/// itself, or no task at all) has nowhere to switch to, so it prints
/// the exit and halts, as in Batch 6.
#[no_mangle]
pub extern "C" fn rustos_syscall_entry(number: u64, a0: u64, a1: u64, a2: u64) -> i64 {
    let mut devices = KernelDevices;
    match dispatch(number, a0, a1, a2, &mut devices) {
        SyscallOutcome::Returned(ret) => ret,
        SyscallOutcome::Exited(code) => {
            let pid = KernelDevices.current_pid();
            crate::serial_println!("syscall: pid {} exited with code {}", pid, code);
            crate::println!("syscall: pid {} exited with code {}", pid, code);
            if pid == KERNEL_PID {
                crate::hlt_loop();
            }
            crate::interrupts::exit_current_task(code);
            // exit_current_task only returns when no task is left to
            // run (it switches away otherwise, never returning).
            crate::hlt_loop();
        }
    }
}

// The vector-0x80 entry stub: save every general-purpose register,
// call [`rustos_syscall_entry`] with the trap's arguments, put the
// return value in the saved `rax` slot, restore the registers, `iretq`.
//
// Saved-register offsets from `rbp` (which points at the saved
// `r15` after the pushes): r15 0, r14 8, r13 16, r12 24, r11 32,
// r10 40, r9 48, r8 56, rbp 64, rbx 72, rdi 80, rsi 88, rdx 96,
// rcx 104, rax 112.
//
// The stack is forced 16-byte aligned before the `call` (`and rsp,
// -16`): the stub cannot know the caller's alignment, and the Rust
// dispatcher may use SSE. The saved registers sit above `rbp` and
// are untouched by the alignment or the call's return address.
core::arch::global_asm!(
    ".text",
    ".globl syscall_entry",
    "syscall_entry:",
    "push rax",
    "push rcx",
    "push rdx",
    "push rsi",
    "push rdi",
    "push rbx",
    "push rbp",
    "push r8",
    "push r9",
    "push r10",
    "push r11",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "mov rbp, rsp",
    // Syscall args, per the libc convention (rax = number, rdi = a0,
    // rsi = a1, rdx = a2), into C ABI order for the Rust dispatcher
    // (rdi = number, rsi = a0, rdx = a1, rcx = a2).
    "mov rdi, [rbp + 112]",
    "mov rsi, [rbp + 80]",
    "mov rdx, [rbp + 88]",
    "mov rcx, [rbp + 96]",
    "and rsp, -16",
    "call rustos_syscall_entry",
    // Return value back into the saved rax slot.
    "mov [rbp + 112], rax",
    "mov rsp, rbp",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop r11",
    "pop r10",
    "pop r9",
    "pop r8",
    "pop rbp",
    "pop rbx",
    "pop rdi",
    "pop rsi",
    "pop rdx",
    "pop rcx",
    "pop rax",
    "iretq",
);

extern "C" {
    fn syscall_entry();
}

/// Address of the vector-0x80 assembly stub, for the IDT gate.
pub fn syscall_entry_addr() -> VirtAddr {
    VirtAddr::new(syscall_entry as *const () as u64)
}

/// Issue a real `int 0x80` syscall from kernel code, with libc's
/// register convention (`rax` = number, `rdi`/`rsi`/`rdx` = args,
/// result in `rax`). Only compiled into the kernel binary, never
/// into host tests: on the host an `int 0x80` would hit the Linux
/// 32-bit compat entry instead of this kernel.
#[cfg(not(test))]
pub fn invoke_syscall(number: u64, a0: u64, a1: u64, a2: u64) -> i64 {
    let ret: i64;
    unsafe {
        core::arch::asm!(
            "int 0x80",
            inlateout("rax") number as i64 => ret,
            in("rdi") a0,
            in("rsi") a1,
            in("rdx") a2,
            options(nostack, preserves_flags),
        );
    }
    ret
}

/// Boot self-test: issue real `int 0x80` syscalls from kernel code
/// and print what came back, on serial and VGA. This is the on-box
/// proof that the syscall path works. Runs after interrupts are up;
/// it must not break the boot — a fault here would land in the
/// kernel's own exception handlers, not a triple fault.
///
/// Expected output on a working boot:
/// ```text
/// syscall test: write ok
/// syscall self-test: write(fd=1, 23 bytes) -> 23
/// syscall self-test: getpid() -> 0
/// syscall self-test: yield() -> 0
/// syscall self-test: read(fd=0, 16) -> -11 (EAGAIN=-11 when idle)
/// syscall self-test: fast MSRs: SCE=1 LSTAR=0x... STAR=0x... SFMASK=0x47700
/// syscall self-test: fast MSRs ok
/// syscall self-test: done
/// ```
///
/// The `syscall`-instruction entry itself is NOT executed here:
/// this runs in ring 0, and `sysret` always returns to ring 3 —
/// a `syscall` from kernel code would return into ring 3 with the
/// kernel still executing. Kernel code must never issue it. What
/// the self-test checks instead is the MSR programming: the
/// registers are read back and compared against the values
/// [`crate::fast_syscall::init`] programmed. The live proof of the
/// fast path is the userspace programs: they run on `RealBackend`
/// now, so hello's first line only prints if the `syscall`
/// instruction works, and the first fast trap prints
/// `syscall: fast entry live` on serial.
#[cfg(not(test))]
pub fn self_test() {
    let msg = b"syscall test: write ok\n";
    let ret = invoke_syscall(SYS_WRITE, FD_STDOUT, msg.as_ptr() as u64, msg.len() as u64);
    crate::serial_println!(
        "syscall self-test: write(fd=1, {} bytes) -> {}",
        msg.len(),
        ret
    );
    crate::println!(
        "syscall self-test: write(fd=1, {} bytes) -> {}",
        msg.len(),
        ret
    );
    let pid = invoke_syscall(SYS_GETPID, 0, 0, 0);
    crate::serial_println!("syscall self-test: getpid() -> {}", pid);
    crate::println!("syscall self-test: getpid() -> {}", pid);
    let y = invoke_syscall(SYS_YIELD, 0, 0, 0);
    crate::serial_println!("syscall self-test: yield() -> {}", y);
    crate::println!("syscall self-test: yield() -> {}", y);
    let mut buf = [0u8; 16];
    let r = invoke_syscall(SYS_READ, FD_STDIN, buf.as_mut_ptr() as u64, buf.len() as u64);
    crate::serial_println!(
        "syscall self-test: read(fd=0, 16) -> {} (EAGAIN=-11 when idle)",
        r
    );
    crate::println!(
        "syscall self-test: read(fd=0, 16) -> {} (EAGAIN=-11 when idle)",
        r
    );
    // Fast-path MSRs, read back and checked (see the doc comment
    // above): SCE must be set, LSTAR must be the fast stub, STAR
    // must be the value the GDT encodes, SFMASK must be the Linux
    // mask, and the GS bases must be live 0 / kernel struct.
    let msrs = crate::fast_syscall::read_back();
    let lstar_ok = msrs.lstar == crate::fast_syscall::syscall_entry_fast_addr();
    let star_ok = msrs.star_raw
        == crate::fast_syscall::star_value(
            crate::gdt::user_data_selector().0 - 8,
            crate::gdt::code_selector().0,
        );
    let mask_ok = msrs.sfmask_raw == crate::fast_syscall::SFMASK_VALUE;
    let gs_ok = msrs.gs_base.as_u64() == 0
        && msrs.kernel_gs_base == crate::fast_syscall::syscall_gs_addr();
    let ok = msrs.sce_enabled() && lstar_ok && star_ok && mask_ok && gs_ok;
    crate::serial_println!(
        "syscall self-test: fast MSRs: SCE={} LSTAR={:#x} STAR={:#x} SFMASK={:#x}",
        msrs.sce_enabled() as u8,
        msrs.lstar,
        msrs.star_raw,
        msrs.sfmask_raw
    );
    crate::println!(
        "syscall self-test: fast MSRs: SCE={} LSTAR={:#x} STAR={:#x} SFMASK={:#x}",
        msrs.sce_enabled() as u8,
        msrs.lstar,
        msrs.star_raw,
        msrs.sfmask_raw
    );
    crate::serial_println!(
        "syscall self-test: fast MSRs {}",
        if ok { "ok" } else { "MISMATCH" }
    );
    crate::println!(
        "syscall self-test: fast MSRs {}",
        if ok { "ok" } else { "MISMATCH" }
    );
    crate::serial_println!("syscall self-test: done");
    crate::println!("syscall self-test: done");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake devices for host tests: buffers instead of hardware.
    struct TestDevices {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        stdin: Vec<u8>,
        pid: u32,
        yielded: bool,
        /// True when this fake stands in for a ring-3 (userspace)
        /// caller: its buffers must lie in a registered user region.
        user: bool,
        /// Milliseconds since boot, for `SYS_UPTIME`.
        uptime_ms: u64,
    }

    impl TestDevices {
        fn new(pid: u32) -> Self {
            Self::with_user(pid, false)
        }

        /// A ring-3 caller: buffers are validated against the
        /// mapped user regions.
        fn new_user(pid: u32) -> Self {
            Self::with_user(pid, true)
        }

        fn with_user(pid: u32, user: bool) -> Self {
            Self {
                stdout: Vec::new(),
                stderr: Vec::new(),
                stdin: Vec::new(),
                pid,
                yielded: false,
                user,
                uptime_ms: 12340,
            }
        }
    }

    impl SyscallDevices for TestDevices {
        fn write_stdout(&mut self, bytes: &[u8]) -> usize {
            self.stdout.extend_from_slice(bytes);
            bytes.len()
        }

        fn write_stderr(&mut self, bytes: &[u8]) -> usize {
            self.stderr.extend_from_slice(bytes);
            bytes.len()
        }

        fn read_stdin(&mut self, buf: &mut [u8]) -> Result<usize, i64> {
            if self.stdin.is_empty() {
                return Err(ERR_AGAIN);
            }
            let n = buf.len().min(self.stdin.len());
            buf[..n].copy_from_slice(&self.stdin[..n]);
            self.stdin.drain(..n);
            Ok(n)
        }

        fn uptime_ms(&self) -> u64 {
            self.uptime_ms
        }

        fn current_pid(&self) -> u32 {
            self.pid
        }

        fn is_user_caller(&self) -> bool {
            self.user
        }

        fn yield_now(&mut self) {
            self.yielded = true;
        }
    }

    #[test]
    fn numbers_match_libc_and_process_exactly() {
        // The values rustos-libc documents and rustos-process
        // dispatches on. If any of these change, every userspace
        // program breaks silently — pin them here.
        assert_eq!(SYS_READ, 0);
        assert_eq!(SYS_WRITE, 1);
        assert_eq!(SYS_OPEN, 2);
        assert_eq!(SYS_CLOSE, 3);
        assert_eq!(SYS_YIELD, 24);
        assert_eq!(SYS_GETPID, 39);
        assert_eq!(SYS_SPAWN, 57);
        assert_eq!(SYS_EXIT, 60);
        assert_eq!(SYS_WAIT, 61);
        // RustOS-private: uptime is the first slot in the 1000 range.
        assert_eq!(SYS_UPTIME, 1000);
        assert_eq!(FD_STDIN, 0);
        assert_eq!(FD_STDOUT, 1);
        assert_eq!(FD_STDERR, 2);
        // And they are the same constants rustos-process exports
        // (this crate re-exports them, so this is belt-and-braces).
        assert_eq!(SYS_WRITE, rustos_process::syscall::SYS_WRITE);
        assert_eq!(SYS_EXIT, rustos_process::syscall::SYS_EXIT);
        // Errnos are the negative values the kernel returns.
        assert_eq!(ERR_BADF, -9);
        assert_eq!(ERR_AGAIN, -11);
        assert_eq!(ERR_INVAL, -22);
        assert_eq!(ERR_NOSYS, -38);
        assert_eq!(ERR_FAULT, -14);
    }

    #[test]
    fn dispatch_write_goes_to_stdout_and_stderr() {
        let mut dev = TestDevices::new(0);
        let msg = b"hello";
        assert_eq!(
            dispatch(SYS_WRITE, FD_STDOUT, msg.as_ptr() as u64, msg.len() as u64, &mut dev),
            SyscallOutcome::Returned(5)
        );
        assert_eq!(dev.stdout, b"hello");
        assert_eq!(
            dispatch(SYS_WRITE, FD_STDERR, msg.as_ptr() as u64, msg.len() as u64, &mut dev),
            SyscallOutcome::Returned(5)
        );
        assert_eq!(dev.stderr, b"hello");
        // Zero-length write is a no-op success.
        assert_eq!(
            dispatch(SYS_WRITE, FD_STDOUT, msg.as_ptr() as u64, 0, &mut dev),
            SyscallOutcome::Returned(0)
        );
        assert_eq!(dev.stdout.len(), 5);
    }

    #[test]
    fn dispatch_write_bad_fd_null_and_oversize_are_errors() {
        let mut dev = TestDevices::new(0);
        let msg = b"hi";
        assert_eq!(
            dispatch(SYS_WRITE, 9, msg.as_ptr() as u64, msg.len() as u64, &mut dev),
            SyscallOutcome::Returned(ERR_BADF)
        );
        assert_eq!(
            dispatch(SYS_WRITE, FD_STDOUT, 0, 4, &mut dev),
            SyscallOutcome::Returned(ERR_FAULT)
        );
        assert_eq!(
            dispatch(
                SYS_WRITE,
                FD_STDOUT,
                msg.as_ptr() as u64,
                (MAX_SYSCALL_LEN + 1) as u64,
                &mut dev
            ),
            SyscallOutcome::Returned(ERR_INVAL)
        );
        assert!(dev.stdout.is_empty());
    }

    #[test]
    fn dispatch_read_is_nonblocking_and_consumes_input() {
        let mut dev = TestDevices::new(0);
        // Nothing waiting: -EAGAIN, not EOF (0) and not a block.
        let mut buf = [0u8; 16];
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, buf.as_mut_ptr() as u64, 4, &mut dev),
            SyscallOutcome::Returned(ERR_AGAIN)
        );
        // Bytes waiting: they are delivered, up to len.
        dev.stdin.extend_from_slice(b"abc");
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, buf.as_mut_ptr() as u64, 2, &mut dev),
            SyscallOutcome::Returned(2)
        );
        assert_eq!(&buf[..2], b"ab");
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, buf.as_mut_ptr() as u64, 16, &mut dev),
            SyscallOutcome::Returned(1)
        );
        assert_eq!(buf[0], b'c');
        // Drained: -EAGAIN again.
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, buf.as_mut_ptr() as u64, 16, &mut dev),
            SyscallOutcome::Returned(ERR_AGAIN)
        );
        // Other fds are -EBADF; zero-length read is a no-op success.
        assert_eq!(
            dispatch(SYS_READ, 5, buf.as_mut_ptr() as u64, 4, &mut dev),
            SyscallOutcome::Returned(ERR_BADF)
        );
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, buf.as_mut_ptr() as u64, 0, &mut dev),
            SyscallOutcome::Returned(0)
        );
        // Null buffer with a real length is -EFAULT.
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, 0, 4, &mut dev),
            SyscallOutcome::Returned(ERR_FAULT)
        );
    }

    #[test]
    fn dispatch_getpid_yield_and_exit() {
        let mut dev = TestDevices::new(0);
        assert_eq!(
            dispatch(SYS_GETPID, 0, 0, 0, &mut dev),
            SyscallOutcome::Returned(0)
        );
        // Yield goes through the device's yield_now (a real switch
        // on the box; here just the flag — the switch itself cannot
        // run on the host).
        assert_eq!(
            dispatch(SYS_YIELD, 0, 0, 0, &mut dev),
            SyscallOutcome::Returned(0)
        );
        assert!(dev.yielded);
        // getpid reports the device's pid: the kernel's real
        // implementation reads the running task from the scheduler.
        let mut dev = TestDevices::new(7);
        assert_eq!(
            dispatch(SYS_GETPID, 0, 0, 0, &mut dev),
            SyscallOutcome::Returned(7)
        );
        assert_eq!(
            dispatch(SYS_EXIT, 3, 0, 0, &mut dev),
            SyscallOutcome::Exited(3)
        );
    }

    #[test]
    fn dispatch_uptime_returns_milliseconds_since_boot() {
        let mut dev = TestDevices::new(0);
        dev.uptime_ms = 12340;
        assert_eq!(
            dispatch(SYS_UPTIME, 0, 0, 0, &mut dev),
            SyscallOutcome::Returned(12340)
        );
    }

    #[test]
    fn fill_stdin_buffer_merges_serial_then_key_ring() {
        // Serial bytes come first, in order; the key ring tops the
        // buffer up once serial is empty.
        let mut serial = [b'a', b'b'].into_iter();
        let mut keys = ['c', 'd'].into_iter().peekable();
        let mut buf = [0u8; 4];
        assert_eq!(
            fill_stdin_buffer(&mut buf, || serial.next(), &mut keys),
            Ok(4)
        );
        assert_eq!(&buf, b"abcd");
        // Serial only, short of the buffer: the key ring fills the
        // rest; a second call drains the remainder.
        let mut serial = [b'x'].into_iter();
        let mut keys = ['y', 'z'].into_iter().peekable();
        let mut buf = [0u8; 2];
        assert_eq!(
            fill_stdin_buffer(&mut buf, || serial.next(), &mut keys),
            Ok(2)
        );
        assert_eq!(&buf, b"xy");
        assert_eq!(
            fill_stdin_buffer(&mut buf, || serial.next(), &mut keys),
            Ok(1)
        );
        assert_eq!(buf[0], b'z');
        // Both empty: -EAGAIN, never a block, never EOF (0).
        let mut serial = [].into_iter();
        let mut keys = [].into_iter().peekable();
        assert_eq!(
            fill_stdin_buffer(&mut buf, || serial.next(), &mut keys),
            Err(ERR_AGAIN)
        );
    }

    #[test]
    fn fill_stdin_buffer_never_splits_a_key_character() {
        // A 2-byte character at the end of a 1-byte buffer is left
        // for the next call (which has room), not written half.
        let mut serial = [].into_iter();
        let mut keys = ['é'].into_iter().peekable();
        let mut buf = [0u8; 1];
        assert_eq!(
            fill_stdin_buffer(&mut buf, || serial.next(), &mut keys),
            Err(ERR_AGAIN)
        );
        // The character was not consumed: a bigger buffer gets it
        // whole on the next call.
        let mut buf = [0u8; 2];
        assert_eq!(
            fill_stdin_buffer(&mut buf, || serial.next(), &mut keys),
            Ok(2)
        );
        assert_eq!(&buf, "é".as_bytes());
    }

    #[test]
    fn dispatch_user_write_validates_buffers_against_user_regions() {
        // A ring-3 caller whose buffer lies in a registered user
        // region writes fine; one whose buffer lies nowhere mapped
        // gets -EFAULT and nothing is written (the buffer is never
        // dereferenced). A kernel caller with the same unmapped
        // pointer writes fine — it is not region-checked.
        let msg = b"hello from userspace";
        let mut dev = TestDevices::new_user(2);
        // Register the real host buffer as a user region, so the
        // positive case has somewhere mapped to point at.
        let ptr = msg.as_ptr() as u64;
        crate::userspace::register_user_region(ptr, ptr + msg.len() as u64);
        assert_eq!(
            dispatch(SYS_WRITE, FD_STDOUT, ptr, msg.len() as u64, &mut dev),
            SyscallOutcome::Returned(msg.len() as i64)
        );
        assert_eq!(dev.stdout, msg);
        // Same bytes, unmapped address: -EFAULT, nothing written.
        let mut dev = TestDevices::new_user(2);
        assert_eq!(
            dispatch(SYS_WRITE, FD_STDOUT, 0x1234, msg.len() as u64, &mut dev),
            SyscallOutcome::Returned(ERR_FAULT)
        );
        assert!(dev.stdout.is_empty());
        // A kernel caller with an unregistered (but real) buffer is
        // not region-checked: it writes fine.
        let other = b"kernel";
        let mut dev = TestDevices::new(0);
        assert_eq!(
            dispatch(SYS_WRITE, FD_STDOUT, other.as_ptr() as u64, other.len() as u64, &mut dev),
            SyscallOutcome::Returned(other.len() as i64)
        );
        assert_eq!(dev.stdout, other);
    }

    #[test]
    fn dispatch_user_read_with_unmapped_buffer_is_efault() {
        let mut dev = TestDevices::new_user(2);
        dev.stdin.extend_from_slice(b"abc");
        let mut buf = [0u8; 16];
        // Unmapped buffer: -EFAULT before any stdin is consumed.
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, 0x1234, 4, &mut dev),
            SyscallOutcome::Returned(ERR_FAULT)
        );
        assert_eq!(dev.stdin, b"abc");
        // The same buffer, registered as a user region, reads fine.
        let ptr = buf.as_mut_ptr() as u64;
        crate::userspace::register_user_region(ptr, ptr + buf.len() as u64);
        assert_eq!(
            dispatch(SYS_READ, FD_STDIN, ptr, 2, &mut dev),
            SyscallOutcome::Returned(2)
        );
        assert_eq!(&buf[..2], b"ab");
    }

    #[test]
    fn dispatch_unimplemented_and_unknown_are_enosys_never_fake_success() {
        let mut dev = TestDevices::new(0);
        for n in [SYS_OPEN, SYS_CLOSE, SYS_SPAWN, SYS_WAIT, 4, 999] {
            assert_eq!(
                dispatch(n, 0, 0, 0, &mut dev),
                SyscallOutcome::Returned(ERR_NOSYS),
                "syscall {n} must be -ENOSYS"
            );
        }
    }
}
