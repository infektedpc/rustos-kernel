//! `rustos-kernel` binary: the freestanding kernel entry point.
//!
//! Thin shell over the `rustos_kernel` library: it receives `BootInfo`
//! from the bootloader, initialises GDT/IDT, paging, and the heap, then
//! proves the heap and the breakpoint exception work before halting.
//!
//! Build with `cargo build` (the `.cargo/config.toml` in this repo
//! already selects the `x86_64-unknown-none` target, and
//! `rust-toolchain.toml` selects nightly).

#![no_std]
#![no_main]

extern crate alloc;

use alloc::{boxed::Box, rc::Rc, vec};
use bootloader_api::config::Mapping;
use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
use core::panic::PanicInfo;
use rustos_kernel::memory::BootInfoFrameAllocator;
use rustos_kernel::{gdt, interrupts, memory, print, println, serial_println, vga, hlt_loop};
use x86_64::VirtAddr;

/// Bootloader configuration for this kernel.
///
/// The default configuration does **not** map physical memory, which
/// would leave `BootInfo::physical_memory_offset` as `None` and make
/// `memory::init` impossible. Request a dynamic physical-memory mapping
/// here so the kernel can walk its own page tables.
pub static BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut config = BootloaderConfig::new_default();
    config.mappings.physical_memory = Some(Mapping::Dynamic);
    config
};

entry_point!(kernel_main, config = &BOOTLOADER_CONFIG);

fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // Serial first: port I/O cannot fault, so this is safe before the
    // IDT is up — and it proves the kernel started even if everything
    // after it goes wrong.
    serial_println!("RustOS kernel: booting");

    // GDT + IDT BEFORE anything that can fault. The bootloader leaves
    // its own IDT behind, and it is unmapped in the kernel's page
    // tables: a fault before this point cannot even read the #PF gate,
    // so it faults again delivering #DF and triple-faults (this is
    // exactly the 2026-11-21 reset loop: the first VGA print faulted
    // on the unmapped 0xb8000 with the bootloader's IDT still live).
    // From here a fault lands in the kernel's own #PF handler and its
    // panic message instead of a reset.
    gdt::init();
    interrupts::init_idt();

    // `syscall`-instruction entry (Batch 11): program EFER.SCE,
    // LSTAR, STAR, SFMASK and the GS bases so ring-3 programs can
    // trap with the `syscall` instruction instead of `int 0x80`.
    // Needs the GDT (STAR takes its selectors) and nothing else —
    // no heap, no paging — so it goes here, before memory init. A
    // bad STAR panics here, at boot, instead of failing on the
    // first userspace `syscall`.
    let fast = rustos_kernel::fast_syscall::init();
    serial_println!(
        "syscall: fast entry ready: EFER.SCE={} LSTAR={:#x} STAR={:#x} SFMASK={:#x}",
        fast.sce_enabled() as u8,
        fast.lstar,
        fast.star_raw,
        fast.sfmask_raw
    );

    let phys_mem_offset = VirtAddr::new(
        boot_info
            .physical_memory_offset
            .into_option()
            .expect("physical memory mapping was not set up by the bootloader"),
    );
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator =
        unsafe { BootInfoFrameAllocator::init(&boot_info.memory_regions) };

    // Map the VGA text buffer before the first VGA write. The
    // bootloader never maps 0xb8000 under UEFI/GOP, so the first
    // `println!` used to page-fault right here. If the mapping cannot
    // be established, VGA output stays off and the kernel runs
    // serial-only — it never prints to VGA on a hope.
    if vga::init(&mut mapper, &mut frame_allocator) {
        serial_println!("vga: text buffer at 0xb8000 mapped");
    } else {
        serial_println!("vga: text buffer at 0xb8000 could not be mapped; serial-only output");
    }
    println!("RustOS kernel: booting");

    memory::init_heap(&mut mapper, &mut frame_allocator).expect("heap initialization failed");

    // Framebuffer console (Batch 10): the bootloader hands us a GOP
    // framebuffer (1280x800 Bgr on QEMU); render the kernel console
    // into it through the TTY layer (rustos-tty over rustos-fb)
    // while serial and VGA keep working exactly as before. The fb
    // is additive: if it is missing or an unsupported format, the
    // kernel runs serial+VGA-only and says so on serial. Init needs
    // the heap (the TTY's shadow cell buffer is a Vec), so it goes
    // here — the first few boot lines never reach the framebuffer.
    match boot_info.framebuffer.as_mut() {
        Some(fb) => match rustos_kernel::fbcon::init(fb) {
            Some(init) => {
                serial_println!(
                    "fb: framebuffer {}x{} {} 32bpp stride {} at {:#x}, {}x{} cells",
                    init.width, init.height, init.format, init.stride, init.addr, init.cols, init.rows
                );
                // Live VT self-test, fb-only (raw escapes must not
                // pollute the serial log): a green line proves SGR
                // parsing works on the real console.
                {
                    let _irq = interrupts::IrqGuard::hold();
                    rustos_kernel::fbcon::write_str("\x1b[32mfb: console ready (SGR ok)\x1b[0m\n");
                }
            }
            None => {
                serial_println!(
                    "fb: framebuffer reported but unusable (unsupported format); serial+VGA only"
                );
            }
        },
        None => {
            serial_println!("fb: no framebuffer reported by the bootloader; serial+VGA only");
        }
    }

    // Prove the heap works: Box, Vec, and Rc all allocate from it.
    let heap_value = Box::new(41);
    let mut vec = vec![1, 2, 3];
    vec.push(*heap_value);
    let rc = Rc::new(vec);
    println!("heap test: Box={} Vec len={} Rc strong={}", heap_value, rc.len(), Rc::strong_count(&rc));
    serial_println!("heap test: ok (Box, Vec, Rc)");

    // Bring up hardware interrupts: PIC remap, PIT at 100 Hz,
    // IRQ0/IRQ1 unmasked, sti. PIC-only by design — the LAPIC stays
    // off until switch_to_apic_mode is actually called (enabling it
    // while the PIC owned delivery #GP'd the first timer IRQ on
    // 2026-11-21: the interrupt arrived with no valid vector).
    // From here the timer fires and keys queue into the ring buffer.
    interrupts::init_interrupts();
    serial_println!("interrupts: PIC remap + PIT 100 Hz done, sti");

    // Prove the breakpoint exception works: int3 should print the stack
    // frame via the IDT handler and return here.
    println!("triggering breakpoint exception...");
    x86_64::instructions::interrupts::int3();
    println!("breakpoint handled, back in kernel_main.");

    // Prove the syscall path works: real `int 0x80` traps through
    // the vector-0x80 gate and back, with the results printed on
    // serial and VGA. This is the on-box proof for Batch 6.
    rustos_kernel::syscall::self_test();

    // Create the kernel's tasks (Batch 7): the idle task (pid 0, the
    // kernel itself) and the worker task (pid 1). The first switch
    // happens on the next timer tick, ~10 ms later, inside the timer
    // IRQ handler.
    interrupts::init_tasks();

    // Load the userspace programs (Batch 8 hello, Batch 9 shell)
    // and run each as a ring-3 task. The loader maps each program's
    // ELF segments, its stack, and its heap into user-accessible
    // pages in its own slot; add_user_task gives it a kernel stack
    // (TSS.rsp[0]) and an iretq frame, and the timer preempts it
    // like any other task. A missing program (not built before the
    // kernel) is a loud warning, not a failed boot: the kernel keeps
    // running without it. Hello exits after ~40 s; the shell keeps
    // going until `exit` — neither disturbs the other (separate
    // tasks, separate slots, separate heaps).
    load_userspace_program(
        &mut mapper,
        &mut frame_allocator,
        "hello",
        rustos_kernel::userspace::HELLO_ELF,
        &rustos_kernel::userspace::HELLO_LAYOUT,
    );
    load_userspace_program(
        &mut mapper,
        &mut frame_allocator,
        "sh",
        rustos_kernel::userspace::SH_ELF,
        &rustos_kernel::userspace::SH_LAYOUT,
    );

    idle_loop();
}

/// Load one userspace program and register it as a ring-3 task,
/// printing the loaded/created lines (serial + VGA) or a loud
/// warning when the program was not built before the kernel.
fn load_userspace_program(
    mapper: &mut x86_64::structures::paging::OffsetPageTable<'static>,
    frame_allocator: &mut BootInfoFrameAllocator,
    name: &'static str,
    elf: &[u8],
    layout: &rustos_kernel::userspace::UserLayout,
) {
    match rustos_kernel::userspace::load_user_program(mapper, frame_allocator, elf, layout) {
        Ok(loaded) => {
            serial_println!(
                "userspace: {} loaded: entry={:#x} user_rsp={:#x} segments={} heap={:#x}+{:#x}",
                name,
                loaded.entry,
                loaded.user_rsp,
                loaded.segments,
                loaded.heap_base,
                loaded.heap_size
            );
            println!(
                "userspace: {} loaded: entry={:#x} user_rsp={:#x} segments={} heap={:#x}+{:#x}",
                name,
                loaded.entry,
                loaded.user_rsp,
                loaded.segments,
                loaded.heap_base,
                loaded.heap_size
            );
            let pid = interrupts::add_user_task(name, 5, loaded.entry, loaded.user_rsp);
            serial_println!("userspace: {} task pid {} created", name, pid.0);
            println!("userspace: {} task pid {} created", name, pid.0);
        }
        Err(e) => {
            serial_println!(
                "userspace: WARNING — no {} program loaded ({:?}); booting without it (build rustos-{}, then rebuild the kernel)",
                name, e, name
            );
            println!(
                "userspace: WARNING — no {} program loaded ({:?}); booting without it",
                name, e
            );
        }
    }
}

/// Idle loop: halt until the next interrupt, then drain any queued
/// keystrokes (echoing printable characters), poll COM1 for serial
/// input (echoing it back over serial), and report uptime every 500
/// ticks (5 s at 100 Hz). The PIT fires at 100 Hz, so this wakes
/// ~100 times a second even with no keys pressed. Scheduling needs
/// nothing from here anymore: the timer IRQ switches tasks itself
/// (see interrupts::handle_timer_irq), so this loop is just the idle
/// task's body — it runs whenever no other task is Ready.
fn idle_loop() -> ! {
    println!("RustOS kernel ready. Timer + keyboard interrupts live; type something.");
    serial_println!("RustOS kernel ready. Timer + keyboard interrupts live; type something.");
    // Framebuffer live-proof, part 1: right after the ready line,
    // the fb holds every boot line so far (plus the shell prompt
    // once sh has run). Logged to serial so a `-display none` boot
    // can still prove the console renders — see `log_fb_stats`.
    log_fb_stats("fb: stats");
    let mut last_printed_ticks = 0u64;
    let mut fb_stats_5s_logged = false;
    loop {
        x86_64::instructions::hlt();
        // Keyboard input belongs to userspace while any user task
        // is alive: its `read(fd 0)` is the only consumer (see
        // `has_live_user_task`). Draining here as well would steal
        // bytes the shell is waiting for and echo them twice — once
        // here, once by the shell. With no user task left, the idle
        // loop resumes echoing so the console stays live.
        if !interrupts::has_live_user_task() {
            while let Some(event) = interrupts::pop_key_event() {
                if let Some(ch) = event.ch {
                    print!("{}", ch);
                }
            }
            rustos_kernel::serial::drain_serial_input();
        }
        let ticks = interrupts::timer_ticks();
        if interrupts::uptime_due(ticks, last_printed_ticks) {
            // Both outputs: `println!` is VGA-only, and the user
            // runs with `-display none`, so a VGA-only uptime line
            // is invisible — the 2026-11-22 live boot showed zero
            // `uptime:` lines in ~26 s for exactly this reason,
            // while the line was printing to VGA all along. Every
            // other boot line is mirrored to serial; this one is now
            // too.
            println!("uptime: {} ticks ({} ms)", ticks, ticks * 10);
            serial_println!("uptime: {} ticks ({} ms)", ticks, ticks * 10);
            // Framebuffer live-proof, part 2: at the first uptime
            // tick (~5 s in) hello and the shell prompt have
            // rendered, so this snapshot's glyph count and checksum
            // should both be larger than the ready-time one — a
            // changing, non-zero checksum proves live rendering.
            if !fb_stats_5s_logged {
                log_fb_stats("fb: stats@5s");
                fb_stats_5s_logged = true;
            }
            last_printed_ticks = ticks;
        }
    }
}

/// Log the framebuffer console's live-proof snapshot to serial:
/// glyphs drawn, lit pixels, and a checksum of the pixel buffer.
/// A no-op when the console is uninitialised. Called once right
/// after the ready line and once at the first uptime tick (~5 s in,
/// by which time hello and the shell prompt have rendered).
fn log_fb_stats(tag: &str) {
    let _irq = interrupts::IrqGuard::hold();
    if let Some(s) = rustos_kernel::fbcon::stats() {
        serial_println!(
            "{} glyphs={} lit_pixels={} checksum={:#x}",
            tag, s.glyphs_drawn, s.lit_pixels, s.checksum
        );
    }
}

/// Panic handler: report on VGA and serial, then halt.
/// (Panic = abort is set in Cargo.toml, so no unwinding is involved.)
/// The VGA print is a no-op while the text buffer is unmapped (see
/// `vga::VGA_AVAILABLE`), so a panic before VGA is mapped still
/// reaches the serial log instead of faulting again.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("{}", info);
    serial_println!("{}", info);
    hlt_loop();
}
