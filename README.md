# rustos-kernel

A `no_std` x86-64 kernel for RustOS, using `bootloader_api` 0.11
(`entry_point!` + `BootInfo`), `x86_64` 0.15, VGA text output, serial
output, GDT + TSS, IDT, paging, and a heap.

## What it does at boot

1. Receives `BootInfo` from the bootloader (`entry_point!`).
2. Prints a banner on serial COM1 (port I/O cannot fault, so this
   is safe before the IDT is up — and it proves the kernel started
   even if everything after it goes wrong).
3. Loads the GDT (kernel code segment + TSS with a dedicated
   double-fault stack) and the IDT (breakpoint, double-fault,
   page-fault, timer, and keyboard handlers) **before any VGA
   access**: the bootloader leaves its own IDT behind, unmapped in
   the kernel's page tables, so a fault before this point
   triple-faults.
4. Builds an `OffsetPageTable` from the bootloader's physical-memory
   mapping and a frame allocator from the `Usable` memory regions,
   then maps the VGA text buffer at `0xb8000` (identity-mapped,
   PRESENT | WRITABLE | NO_CACHE) before the first VGA print. If the
   mapping cannot be established, VGA output stays off and the
   kernel runs serial-only — it never prints to VGA on a hope. Only
   then does it print the banner on VGA.
5. Maps a 100 KiB heap and proves it with `Box`, `Vec`, and `Rc`.
6. Initialises interrupts (`interrupts::init_interrupts`): PIC
   remap (master -> 32, slave -> 40), PIT at 100 Hz, APIC MMIO
   pages mapped + LAPIC enabled (IOAPIC entries for IRQ0/IRQ1
   programmed but masked — delivery stays with the PIC), IRQ0/IRQ1
   unmasked, interrupts enabled (`sti`).
7. Triggers a breakpoint (`int3`) to prove the IDT works, prints that it
   survived, and enters the idle loop: `hlt` until the next
   interrupt, drain queued keystrokes (echoing printable
   characters), act on timer preempt requests, print uptime every
   5 s. The kernel no longer halts after booting — it idles on the
   timer interrupt.

## Build (needs nightly)

The kernel needs a **nightly** toolchain: the IDT handlers use the
`extern "x86-interrupt"` ABI, which stable rustc 1.99.1 rejects
(E0658, rust-lang/rust#40180). `rust-toolchain.toml` in this repo pins
nightly, and `.cargo/config.toml` selects the bare-metal target, so a
plain build just works once nightly is installed:

```sh
rustup toolchain install nightly
rustup target add --toolchain nightly x86_64-unknown-none
cargo build
```

That produces `target/x86_64-unknown-none/debug/rustos-kernel`.
No custom target JSON file is needed: `x86_64-unknown-none` ships with
rustc (`rustup target add x86_64-unknown-none`), so `build-std` and a
hand-written target spec are both unnecessary.

## Tests (run on the host)

```sh
cargo test --target x86_64-unknown-linux-gnu
```

The kernel is split into a library (`src/lib.rs`, `no_std` on target,
`std` under `#[cfg(test)]`) and a thin binary (`src/main.rs`, `test =
false` so `cargo test` never tries to link the freestanding binary for
the host). Tests cover:

- VGA colour-byte encoding and colour values,
- the VGA writer over a fake in-memory buffer (ordering, newlines,
  non-printable bytes, row wrap, scrolling when full, `fmt::Write`),
- the heap allocator (allocate into a scratch heap, write, free, and
  reallocate — the same `linked_list_allocator::Heap` the kernel uses,
  exercised as a local heap so the test harness keeps its own allocator).
- the interrupt handler logic, factored so host tests can drive it:
  the timer step calls the scheduler tick exactly at quantum
  boundaries (first tick at 100 Hz, tenth at a 10-tick quantum) and
  preempts to the other task; the keyboard step decodes scancodes
  into queued events; EOI for vector 33 goes to the PIC; a preempt
  request with no running task is dropped without panicking.

Plain `cargo test` (no `--target`) does **not** work: the default build
target is `x86_64-unknown-none`, which has no OS to run tests under.
The `--target` flag is the documented command.

## Boot it

Build a disk image with `rustos-bootloader`, then boot under QEMU
(QEMU is not installed in this sandbox — boot on a machine that has it):

```sh
../rustos-bootloader/target/release/rustos-bootloader \
  --kernel target/x86_64-unknown-none/debug/rustos-kernel \
  --out disk.img
qemu-system-x86_64 -drive file=disk.img,format=raw \
  -bios /usr/share/OVMF/OVMF_CODE.fd -serial stdio
```

## Layout

- `src/main.rs` — entry point, `BootloaderConfig` (requests the
  physical-memory mapping), panic handler.
- `src/lib.rs` — `no_std`/`std` split, `print!`/`println!`,
  `serial_print!`/`serial_println!`, `hlt_loop`.
- `src/vga.rs` — VGA text writer (0xb8000, 80x25).
- `src/serial.rs` — COM1 serial writer (0x3F8) + polled serial input:
  the idle loop drains waiting bytes and echoes printable ones back
  over serial, so typing works in a `-serial stdio` terminal too.
- `src/gdt.rs` — GDT + TSS + double-fault IST stack.
- `src/interrupts.rs` — IDT (breakpoint, double-fault, page-fault,
  timer, keyboard) + controller setup: PIC remap, PIT 100 Hz, APIC
  init, IRQ0/IRQ1 unmask, `sti`; timer -> scheduler tick,
  keyboard -> 64-entry event ring; `maybe_reschedule` for the idle
  loop; `switch_to_apic_mode` (not called at boot — see below).
- `src/memory.rs` — page-table init, `BootInfoFrameAllocator`, heap,
  `map_mmio_page` for device register pages (LAPIC/IOAPIC).

## Known limitations

- **VGA text output is mapped by the kernel itself at boot.**
  The bootloader never maps `0xb8000` under UEFI/GOP, and the first
  VGA print used to page-fault on it — with the bootloader's IDT
  still live, that faulted again reading the #DF gate and
  triple-faulted (2026-11-21 reset loop on the user's box: RIP in
  `Writer::write_byte`, CR2 = IDT base + 0x80). The kernel now loads
  its own GDT + IDT before any VGA access and maps one page at
  `0xb8000` before the first VGA print; if the mapping fails it runs
  serial-only. A fault after that lands in the kernel's own #PF
  handler (mirrored to serial) instead of a reset. **Open question:**
  the first live boot (2026-11-21 ~13:32, old kernel) printed
  uptime on VGA fine, so `0xb8000` was evidently usable then; what
  changed is unknown — possibly OVMF memory-map variance between
  boots, or the rebuilt bootloader's layout. The fix holds either
  way: VGA is mapped by the kernel or not used at all.
- **Timer/keyboard interrupts are in, but not live-boot tested.**
  The IDT has timer (vector 32) and keyboard (vector 33) handlers,
  the PIC is remapped, the PIT runs at 100 Hz, and the LAPIC is
  enabled with IOAPIC entries programmed (masked — delivery stays
  with the PIC). All of it is verified by host unit tests against
  mock ports/registers only. The PIC -> APIC mode switch
  (`switch_to_apic_mode`) is implemented but not called at boot and
  not verified on hardware. The APIC bases are defaults
  (0xFEE00000 / 0xFEC00000), not discovered from ACPI/MADT.
- **Typing works two ways.** PS/2 keyboard input (IRQ1) echoes on
  VGA — type in the QEMU display window. Serial input (COM1, polled
  in the idle loop, no interrupt involved) echoes back over serial —
  type in the terminal when booting with `-serial stdio -display
  none`. Only printable ASCII echoes; CR/LF echo as CRLF; other
  bytes are ignored. The echo policy is host unit-tested; the
  actual port reads are not — the live boot is the proof.
- **No tasks in the kernel yet.** The scheduler exists and the timer
  IRQ calls `tick()` on every tick, but the kernel creates no tasks,
  so `tick()` always returns `Continue` and `maybe_reschedule`
  has nothing to switch to. Adding a first task (shell) is the next
  step.
- **No filesystem or userspace yet.** The heap, paging,
  and exception handling in this repo are the foundation those build on.
- **Not live-boot tested here.** QEMU is not installed in this sandbox,
  so the kernel is verified by `cargo build` for the target plus the
  host unit tests — not by an actual boot. The first real boot is the
  proof that matters.
