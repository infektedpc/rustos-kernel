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

    idle_loop();
}

/// Idle loop: halt until the next interrupt, then drain any queued
/// keystrokes (echoing printable characters), poll COM1 for serial
/// input (echoing it back over serial), act on a pending preempt
/// request from the timer IRQ, and report uptime every 500 ticks
/// (5 s at 100 Hz). The PIT fires at 100 Hz, so this wakes ~100
/// times a second even with no keys pressed.
fn idle_loop() -> ! {
    println!("RustOS kernel ready. Timer + keyboard interrupts live; type something.");
    serial_println!("RustOS kernel ready. Timer + keyboard interrupts live; type something.");
    let mut last_printed_ticks = 0u64;
    loop {
        x86_64::instructions::hlt();
        interrupts::maybe_reschedule();
        while let Some(event) = interrupts::pop_key_event() {
            if let Some(ch) = event.ch {
                print!("{}", ch);
            }
        }
        rustos_kernel::serial::drain_serial_input();
        let ticks = interrupts::timer_ticks();
        if ticks >= last_printed_ticks + 500 {
            println!("uptime: {} ticks ({} ms)", ticks, ticks * 10);
            last_printed_ticks = ticks;
        }
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
