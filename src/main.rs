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
use rustos_kernel::{gdt, interrupts, memory, println, serial_println, hlt_loop};
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
    serial_println!("RustOS kernel: booting");
    println!("RustOS kernel: booting");

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

    memory::init_heap(&mut mapper, &mut frame_allocator).expect("heap initialization failed");

    // Prove the heap works: Box, Vec, and Rc all allocate from it.
    let heap_value = Box::new(41);
    let mut vec = vec![1, 2, 3];
    vec.push(*heap_value);
    let rc = Rc::new(vec);
    println!("heap test: Box={} Vec len={} Rc strong={}", heap_value, rc.len(), Rc::strong_count(&rc));
    serial_println!("heap test: ok (Box, Vec, Rc)");

    // Prove the breakpoint exception works: int3 should print the stack
    // frame via the IDT handler and return here.
    println!("triggering breakpoint exception...");
    x86_64::instructions::interrupts::int3();
    println!("breakpoint handled, back in kernel_main.");

    println!("RustOS kernel ready. Halting.");
    serial_println!("RustOS kernel ready. Halting.");
    hlt_loop();
}

/// Panic handler: report on VGA and serial, then halt.
/// (Panic = abort is set in Cargo.toml, so no unwinding is involved.)
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("{}", info);
    serial_println!("{}", info);
    hlt_loop();
}
