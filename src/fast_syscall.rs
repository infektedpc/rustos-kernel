//! The `syscall`-instruction entry (Batch 11): the fast path into the kernel.
//!
//! ## Why this exists
//!
//! Since Batch 6 the kernel has served syscalls through `int 0x80`
//! (vector 0x80, DPL 3 interrupt gate). That works, but every trap
//! goes through the IDT, pushes an interrupt frame, and `iretq`s
//! back. The x86_64 `syscall`/`sysret` pair is the fast path Linux
//! uses: `syscall` jumps straight to an address in a register
//! (`LSTAR`), `sysret` jumps straight back — no IDT, no frame. This
//! module wires it up, and `rustos-libc`'s [`RealBackend`] (the
//! default backend since this batch) issues it.
//!
//! ## The moving parts
//!
//! * `EFER.SCE` (bit 0 of `IA32_EFER`, 0xC0000080): enables the
//!   `syscall`/`sysret` instructions at all. Without it a ring-3
//!   `syscall` is `#UD`.
//! * `LSTAR` (0xC0000082): the RIP `syscall` jumps to —
//!   [`syscall_entry_fast`], the assembly stub below.
//! * `STAR` (0xC0000081): the segment selectors. Bits 47:32 are the
//!   kernel CS (`syscall` loads CS from there, SS = CS + 8); bits
//!   63:48 are the *base* for `sysret` (CS = base + 16, SS = base +
//!   8, RPL forced to 3). That derivation is why the GDT puts the
//!   user DATA segment one entry before the user CODE segment (see
//!   `crate::gdt`): with data at index 5 (0x2B) and code at index 6
//!   (0x33) the base is 0x23, giving CS = 0x33 and SS = 0x2B.
//! * `SFMASK` (0xC0000084): the RFLAGS bits `syscall` clears on
//!   entry. IF is the load-bearing one — the stub runs with
//!   interrupts off, exactly like the `int 0x80` interrupt gate
//!   (the `yield` path takes the same `SCHEDULER` lock the timer
//!   IRQ takes, so an IRQ mid-syscall would deadlock the same way
//!   a trap gate would). TF/DF/IOPL/AC/NT are masked too, matching
//!   Linux's mask (0x47700).
//! * `swapgs` + the per-CPU [`SyscallGs`]: unlike an interrupt,
//!   `syscall` does NOT switch stacks and does NOT save `rsp`.
//!   The stub starts with `swapgs`, which swaps the live GS base
//!   with `IA32_KERNEL_GS_BASE` — the kernel's per-CPU struct —
//!   then saves the user `rsp` in the struct's scratch slot and
//!   loads the running task's kernel stack top from it.
//!   `KernelGsBase` is programmed once at [`init`]; the live GS
//!   base is 0 while a user task runs (userspace never sets GS),
//!   so the swap always lands on the struct. The second `swapgs`
//!   (before `sysret`) restores the user GS base (0).
//!
//! ## The stub's contract
//!
//! `syscall` itself saves only the user RIP (into `rcx`) and
//! RFLAGS (into `r11`) and clobbers both; `sysret` restores from
//! exactly those two registers. The stub therefore saves and
//! restores `rcx`/`r11` around the dispatcher call, along with the
//! other registers the caller expects back (`rdi`/`rsi`/`rdx` —
//! the syscall arguments, which the C-ABI shuffle overwrites — and
//! `r8`/`r9`/`r10`, the extra arguments). `rbx`/`rbp`/`r12`–`r15`
//! are callee-saved: the Rust dispatcher preserves them, so the
//! stub does not push them. `rax` carries the syscall number in
//! and the return value out.
//!
//! The dispatcher is the SAME [`crate::syscall::rustos_syscall_entry`]
//! the `int 0x80` stub calls — one table, one implementation, two
//! entry paths. The only difference is the thin wrapper
//! [`rustos_syscall_entry_fast`], which counts fast-path traps and
//! prints one line the first time the path goes live.
//!
//! ## What the boot self-test can and cannot prove
//!
//! The kernel's self-test runs in ring 0, and `sysret` ALWAYS
//! returns to ring 3 — so a `syscall` issued from ring 0 would
//! `sysret` back into ring 3 at the saved RIP, with the kernel
//! still executing, which is catastrophic. Kernel code must NEVER
//! issue `syscall`; the self-test therefore verifies the fast path
//! by reading the MSRs back (SCE set, LSTAR/STAR/SFMASK as
//! programmed) and printing the verdict, and the live proof is the
//! userspace programs themselves: hello and sh now run on
//! [`RealBackend`], so their first output only happens if the
//! `syscall` instruction works, and the first fast trap prints
//! `syscall: fast entry live` on serial.
//!
//! ## Stack-top bookkeeping
//!
//! [`SyscallGs::kernel_stack_top`] mirrors `TSS.rsp[0]` (see
//! `crate::gdt::set_kernel_stack_top`): it is updated on every
//! task switch, in [`crate::interrupts::set_kernel_stack_for`],
//! before the register switch, while interrupts are off — the
//! same moment the TSS is updated, for the same reason (a
//! `syscall` landing after the switch but before the update would
//! run on the previous task's kernel stack).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::registers::model_specific::EferFlags;
use x86_64::registers::rflags::RFlags;
use x86_64::VirtAddr;

// The MSR types are only used by init()/read_back(), which are
// ring-0-only (#[cfg(not(test))]) — importing them unconditionally
// warns as unused in host test builds.
#[cfg(not(test))]
use x86_64::registers::model_specific::{Efer, GsBase, KernelGsBase, LStar, SFMask, Star};

/// `SFMASK`: the RFLAGS bits `syscall` clears on entry.
///
/// TF (single-step) | DF (direction) | IF (interrupts) |
/// IOPL (both bits) | NT (nested task) | AC (alignment check).
/// IF is load-bearing (the handler runs with interrupts off, like
/// the `int 0x80` interrupt gate); the rest match Linux's mask.
pub const SFMASK_VALUE: u64 = 0x47700;

/// The STAR field (bits 63:48) for this kernel's GDT: the user
/// data selector (0x2B) minus 8 — `sysret` derives CS = base + 16
/// (0x33, user code) and SS = base + 8 (0x2B, user data).
/// Pinned by test against the GDT constants, not hardcoded into
/// the programming path ([`init`] derives it from the live GDT).
pub const SYSRET_BASE: u16 = 0x23;

/// The STAR field (bits 47:32) for `syscall`: the kernel code
/// selector (0x08); the kernel SS is CS + 8 (0x10, kernel data).
pub const SYSCALL_CS: u16 = 0x08;

/// Compute a raw STAR value from its two 16-bit fields:
/// `sysret_base` in bits 63:48, `syscall_cs` in bits 47:32.
/// Pure, so host tests can pin the encoding.
pub const fn star_value(sysret_base: u16, syscall_cs: u16) -> u64 {
    ((sysret_base as u64) << 48) | ((syscall_cs as u64) << 32)
}

/// The per-CPU `syscall`-entry state, reached through GS after
/// `swapgs`. Single CPU, so one static is the whole table.
///
/// `#[repr(C)]` pins the layout the assembly stub relies on:
/// `kernel_stack_top` at offset 0, `user_rsp` (scratch) at
/// offset 8 — both pinned by test.
///
/// The fields are [`UnsafeCell`]s on purpose: a plain `static`
/// with plain `u64` fields is emitted into `.rodata` by the
/// compiler (safe Rust can never write it), and the first raw
/// write through the pointer would `#PF` — the Batch 11 test
/// binary segfaulted on exactly that. `UnsafeCell` keeps the
/// static in writable memory. Access is sound for the same
/// reason as the TSS (see `crate::gdt`): single CPU, writers run
/// with interrupts off, and the stub only touches the scratch
/// slot while interrupts are off.
#[repr(C)]
pub struct SyscallGs {
    /// Top of the running task's kernel stack: the `rsp` the stub
    /// switches to on entry. Mirrors `TSS.rsp[0]`; updated on
    /// every task switch (see [`set_kernel_stack_top`]).
    pub kernel_stack_top: UnsafeCell<u64>,
    /// Scratch: the user `rsp`, saved by the stub before it
    /// switches stacks and pushed onto the kernel stack
    /// immediately after — the slot is dead from that point on,
    /// so a later trap overwriting it is harmless.
    pub user_rsp: UnsafeCell<u64>,
}

// SAFETY: single CPU; the fields are written only with interrupts
// off (the task switch, and the stub itself), and read only from
// the stub (after `swapgs`, interrupts off) or from the same
// interrupt-off paths. No two contexts can touch them at once.
unsafe impl Sync for SyscallGs {}

impl SyscallGs {
    const fn new() -> Self {
        Self {
            kernel_stack_top: UnsafeCell::new(0),
            user_rsp: UnsafeCell::new(0),
        }
    }
}

static SYSCALL_GS: SyscallGs = SyscallGs::new();

/// Address of the per-CPU [`SyscallGs`], for the `KernelGsBase`
/// MSR and for read-back verification.
pub fn syscall_gs_addr() -> VirtAddr {
    VirtAddr::new(&raw const SYSCALL_GS as u64)
}

/// Set the kernel stack the `syscall` stub switches to: the
/// running task's kernel stack top. Called on every task switch
/// (see [`crate::interrupts::set_kernel_stack_for`]), next to the
/// `TSS.rsp[0]` update. Same soundness story as
/// [`crate::gdt::set_kernel_stack_top`]: the struct lives at a
/// fixed address for the whole run, this is single-CPU, and every
/// caller runs with interrupts off — and the stub only reads the
/// field after `swapgs`, which cannot interleave with this write.
pub fn set_kernel_stack_top(top: VirtAddr) {
    unsafe {
        *SYSCALL_GS.kernel_stack_top.get() = top.as_u64();
    }
}

/// The `kernel_stack_top` value right now, for tests and
/// diagnostics. (Only meaningful after a task switch has set it;
/// before that it is 0 — the same as `TSS.rsp[0]` for the idle
/// task, which never runs in ring 3.)
pub fn kernel_stack_top() -> VirtAddr {
    VirtAddr::new(unsafe { *SYSCALL_GS.kernel_stack_top.get() })
}

/// Fast-path traps taken so far (every `syscall`-instruction entry
/// through [`rustos_syscall_entry_fast`]).
static FAST_SYSCALLS: AtomicU64 = AtomicU64::new(0);

/// How many `syscall`-instruction traps have been taken.
pub fn fast_syscall_count() -> u64 {
    FAST_SYSCALLS.load(Ordering::Relaxed)
}

/// The MSR state of the fast entry, read back after programming
/// (see [`init`] / [`read_back`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastEntryInfo {
    /// Raw `IA32_EFER` value.
    pub efer_raw: u64,
    /// `LSTAR`: the stub address `syscall` jumps to.
    pub lstar: VirtAddr,
    /// Raw `STAR` value.
    pub star_raw: u64,
    /// Raw `SFMASK` value.
    pub sfmask_raw: u64,
    /// Live GS base (0 while a user task runs).
    pub gs_base: VirtAddr,
    /// `IA32_KERNEL_GS_BASE`: the [`SyscallGs`] address.
    pub kernel_gs_base: VirtAddr,
}

impl FastEntryInfo {
    /// True when `EFER.SCE` (bit 0) is set: the `syscall`
    /// instruction is enabled at all.
    pub fn sce_enabled(&self) -> bool {
        self.efer_raw & 1 == 1
    }
}

/// Program the `syscall`-instruction MSRs and read them back.
///
/// Must be called once, after [`crate::gdt::init`] (the STAR
/// selectors come from the loaded GDT) and before the first user
/// task runs. A wrong STAR (bad selectors) or a faulting MSR write
/// panics here, at boot, rather than failing silently on the
/// first ring-3 `syscall` — a `#GP` on `wrmsr` would land in the
/// kernel's own exception handlers, not a triple fault.
///
/// Only compiled into the kernel binary, never into host tests:
/// `wrmsr`/`rdmsr` from ring 3 on the host would `#GP` the test
/// process.
#[cfg(not(test))]
pub fn init() -> FastEntryInfo {
    // EFER.SCE: enable `syscall`/`sysret`. `update` preserves the
    // other EFER bits (LME, NXE, ...) and the reserved fields.
    unsafe {
        Efer::update(|flags| {
            flags.insert(EferFlags::SYSTEM_CALL_EXTENSIONS);
        });
    }
    // LSTAR: the fast stub (see below).
    LStar::write(syscall_entry_fast_addr());
    // STAR: validated by the crate — the GDT order (user data
    // index 5, user code index 6) is what makes this succeed.
    Star::write(
        crate::gdt::user_code_selector(),
        crate::gdt::user_data_selector(),
        crate::gdt::code_selector(),
        crate::gdt::data_selector(),
    )
    .expect("STAR selectors invalid — GDT user segment order wrong?");
    // SFMASK: the RFLAGS bits `syscall` clears (see SFMASK_VALUE).
    SFMask::write(
        RFlags::from_bits(SFMASK_VALUE).expect("SFMASK bits are all known RFlags"),
    );
    // GS bases: live GS = 0 (userspace never sets GS), kernel GS =
    // the per-CPU struct — `swapgs` in the stub swaps between them.
    GsBase::write(VirtAddr::new(0));
    KernelGsBase::write(syscall_gs_addr());
    read_back()
}

/// Read the fast-entry MSRs back, for the boot log and the
/// self-test verdict. Ring 0 only (see [`init`]).
#[cfg(not(test))]
pub fn read_back() -> FastEntryInfo {
    FastEntryInfo {
        efer_raw: Efer::read_raw(),
        lstar: LStar::read(),
        star_raw: unsafe { Star::MSR.read() },
        sfmask_raw: unsafe { SFMask::MSR.read() },
        gs_base: GsBase::read(),
        kernel_gs_base: KernelGsBase::read(),
    }
}

/// The Rust half of the `syscall`-instruction entry, called from
/// the [`syscall_entry_fast`] assembly stub with the trap's
/// register arguments (C ABI order — the stub does the shuffle).
///
/// It is the SAME dispatcher the `int 0x80` path calls
/// ([`crate::syscall::rustos_syscall_entry`]): one table, one
/// implementation, two entry paths. The only additions are the
/// trap counter and the one-shot "fast entry live" line — the
/// first fast trap is the on-box proof the path works (hello's
/// first `write` is it, so the line lands right before
/// `hello from userspace (pid 2)`).
///
/// Never returns for `exit`, exactly like the `int 0x80` path:
/// the task is terminated and the kernel switches away.
#[no_mangle]
pub extern "C" fn rustos_syscall_entry_fast(number: u64, a0: u64, a1: u64, a2: u64) -> i64 {
    let n = FAST_SYSCALLS.fetch_add(1, Ordering::Relaxed);
    if n == 0 {
        crate::serial_println!("syscall: fast entry live (first `syscall`-instruction trap)");
    }
    crate::syscall::rustos_syscall_entry(number, a0, a1, a2)
}

// The `syscall`-instruction entry stub: `swapgs` onto the per-CPU
// struct, switch to the running task's kernel stack, save the
// registers the dispatcher may clobber and the user expects back,
// call [`rustos_syscall_entry_fast`] with the trap's arguments,
// restore, `sysret` back to ring 3.
//
// Saved-register offsets from `rbp` (which points at the saved
// `r11` after the pushes): r11 0, rcx 8, r10 16, r9 24, r8 32,
// rdx 40, rsi 48, rdi 56, user_rsp 64.
//
// The stub runs with interrupts off (`SFMASK` masks IF on entry),
// so no IRQ can land between the `swapgs` and the stack switch —
// and no task switch can interleave either, so the per-CPU
// `user_rsp` scratch slot is live only between the two
// instructions that use it.
//
// The stack is forced 16-byte aligned before the `call` (`and
// rsp, -16`): the stub cannot know the caller's alignment, and
// the Rust dispatcher may use SSE. The saved registers sit above
// `rbp` and are untouched by the alignment or the call's return
// address — `rbp` itself survives the call (callee-saved), so
// `mov rsp, rbp` restores the frame afterwards.
//
// Ring 3 ONLY: `sysret` always returns to ring 3, so kernel code
// must never issue `syscall` (see the module docs). The stub
// assumes the caller came from ring 3.
core::arch::global_asm!(
    ".text",
    ".globl syscall_entry_fast",
    "syscall_entry_fast:",
    // GS base <-> KernelGsBase: GS now points at SyscallGs.
    "swapgs",
    // Save the user rsp in the per-CPU scratch slot, then switch
    // to the running task's kernel stack top.
    "mov gs:[8], rsp",
    "mov rsp, gs:[0]",
    // The user rsp, onto the kernel stack (restored before sysret).
    "push qword ptr gs:[8]",
    // Save what the dispatcher may clobber and the user expects
    // back: the syscall arguments (rdi/rsi/rdx — the C-ABI shuffle
    // below overwrites them), the extra arguments (r8/r9/r10), and
    // rcx/r11, which hold the user RIP/RFLAGS for `sysret`.
    // rbx/rbp/r12-r15 are callee-saved: the Rust fn preserves them.
    // rax carries the syscall number in and the return value out.
    "push rdi",
    "push rsi",
    "push rdx",
    "push r8",
    "push r9",
    "push r10",
    "push rcx",
    "push r11",
    "mov rbp, rsp",
    // Syscall args, per the libc convention (rax = number, rdi =
    // a0, rsi = a1, rdx = a2), into C ABI order for the Rust
    // dispatcher (rdi = number, rsi = a0, rdx = a1, rcx = a2).
    // r10/r8/r9 (the 4th–6th Linux-convention args) are saved
    // above but not passed on: no RustOS syscall takes more than
    // three arguments, so the dispatcher never reads them.
    "mov rdi, rax",
    "mov rsi, [rbp + 56]",
    "mov rdx, [rbp + 48]",
    "mov rcx, [rbp + 40]",
    "and rsp, -16",
    "call rustos_syscall_entry_fast",
    // Return value back into the saved rax slot is NOT needed:
    // rax already holds it, and the pops below restore every
    // other register. Just restore the frame.
    "mov rsp, rbp",
    "pop r11",
    "pop rcx",
    "pop r10",
    "pop r9",
    "pop r8",
    "pop rdx",
    "pop rsi",
    "pop rdi",
    // rsp now points at the saved user-rsp slot on the kernel
    // stack: restore the user rsp, swap GS back, return to ring 3
    // (RIP from rcx, RFLAGS from r11 — both restored above).
    "mov rsp, [rsp]",
    "swapgs",
    "sysret",
);

extern "C" {
    fn syscall_entry_fast();
}

/// Address of the `syscall`-instruction assembly stub, for LSTAR.
pub fn syscall_entry_fast_addr() -> VirtAddr {
    VirtAddr::new(syscall_entry_fast as *const () as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdt;
    use core::mem::{offset_of, size_of};

    #[test]
    fn star_value_encodes_both_fields() {
        // The Batch 11 GDT: sysret base 0x23 (user data 0x2B - 8),
        // syscall CS 0x08 (kernel code). Raw STAR:
        // 0x0023_0008_0000_0000.
        assert_eq!(star_value(0x23, 0x08), 0x0023_0008_0000_0000);
        assert_eq!(star_value(0, 0), 0);
        // Fields do not overlap: each half survives on its own.
        assert_eq!(star_value(0x23, 0) >> 48, 0x23);
        assert_eq!((star_value(0, 0x08) >> 32) & 0xffff, 0x08);
    }

    #[test]
    fn sysret_base_matches_the_gdt_user_selectors() {
        // SYSRET derives CS = base + 16, SS = base + 8. With user
        // data at GDT index 5 (selector 0x2B) and user code at
        // index 6 (0x33), the base must be 0x23 — and it must agree
        // with both derivations at once.
        assert_eq!(gdt::USER_DATA_INDEX, 5);
        assert_eq!(gdt::USER_CODE_INDEX, 6);
        assert_eq!(gdt::user_data_selector().0, 0x2b);
        assert_eq!(gdt::user_code_selector().0, 0x33);
        assert_eq!(SYSRET_BASE, 0x23);
        assert_eq!(SYSRET_BASE + 16, gdt::user_code_selector().0);
        assert_eq!(SYSRET_BASE + 8, gdt::user_data_selector().0);
        assert_eq!(SYSCALL_CS, gdt::code_selector().0);
        assert_eq!(SYSCALL_CS + 8, gdt::data_selector().0);
    }

    #[test]
    fn sfmask_value_is_the_linux_mask() {
        // TF | DF | IF | IOPL (both bits) | NT | AC — the same mask
        // Linux programs into IA32_FMASK. IF is the load-bearing
        // bit: the fast handler runs with interrupts off, like the
        // int 0x80 interrupt gate.
        let expect = RFlags::TRAP_FLAG
            | RFlags::DIRECTION_FLAG
            | RFlags::INTERRUPT_FLAG
            | RFlags::IOPL_HIGH
            | RFlags::IOPL_LOW
            | RFlags::NESTED_TASK
            | RFlags::ALIGNMENT_CHECK;
        assert_eq!(SFMASK_VALUE, expect.bits());
        assert_eq!(SFMASK_VALUE, 0x47700);
        assert_ne!(SFMASK_VALUE & (1 << 9), 0); // IF masked
        assert_eq!(RFlags::from_bits(SFMASK_VALUE).unwrap().bits(), SFMASK_VALUE);
    }

    #[test]
    fn syscall_gs_layout_matches_the_stub_offsets() {
        // The stub reads kernel_stack_top at gs:[0] and the user
        // rsp scratch at gs:[8]. If this struct ever grows a field
        // in front of either, the stub breaks silently — pin it.
        assert_eq!(offset_of!(SyscallGs, kernel_stack_top), 0);
        assert_eq!(offset_of!(SyscallGs, user_rsp), 8);
        assert_eq!(size_of::<SyscallGs>(), 16);
    }

    #[test]
    fn set_kernel_stack_top_round_trips() {
        // The setter the task switch calls writes the same slot
        // the stub reads — host-testable because it is plain
        // memory, no MSR involved.
        set_kernel_stack_top(VirtAddr::new(0x4444_4444_42c0));
        assert_eq!(kernel_stack_top(), VirtAddr::new(0x4444_4444_42c0));
        assert_eq!(syscall_gs_addr().as_u64(), &raw const SYSCALL_GS as u64);
        set_kernel_stack_top(VirtAddr::new(0));
        assert_eq!(kernel_stack_top(), VirtAddr::new(0));
    }

    #[test]
    fn fast_syscall_counter_starts_at_zero() {
        // The counter only moves on a real trap (the wrapper is
        // never called on the host); it must start at zero so the
        // first-trap line fires exactly once, on the first trap.
        assert_eq!(fast_syscall_count(), 0);
    }

    #[test]
    fn efer_sce_is_bit_zero() {
        // EferFlags::SYSTEM_CALL_EXTENSIONS is the SCE bit; the
        // self-test checks it in the read-back EFER value.
        assert_eq!(EferFlags::SYSTEM_CALL_EXTENSIONS.bits(), 1);
        let info = FastEntryInfo {
            efer_raw: 0xD01,
            lstar: VirtAddr::new(0),
            star_raw: 0,
            sfmask_raw: 0,
            gs_base: VirtAddr::new(0),
            kernel_gs_base: VirtAddr::new(0),
        };
        assert!(info.sce_enabled());
        let info = FastEntryInfo { efer_raw: 0xD00, ..info };
        assert!(!info.sce_enabled());
    }
}
