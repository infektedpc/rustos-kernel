//! `rustos-kernel` library: the testable guts of the kernel.
//!
//! This crate is `no_std` when built for the kernel target, and a normal
//! `std` crate when compiled for host unit tests
//! (`cargo test --target x86_64-unknown-linux-gnu`). That split is what
//! makes the VGA writer, colour encoding, and heap logic unit-testable
//! without booting anything: the tests exercise the same code the kernel
//! runs, against a fake in-memory VGA buffer instead of `0xb8000`.
//!
//! The freestanding entry point lives in `src/main.rs` (the `rustos-kernel`
//! binary); it is a thin shell over this library.

#![cfg_attr(not(test), no_std)]
#![feature(abi_x86_interrupt)]

extern crate alloc;

pub mod fbcon;
pub mod fast_syscall;
pub mod gdt;
pub mod interrupts;
pub mod memory;
pub mod serial;
pub mod syscall;
pub mod tasks;
pub mod userspace;
pub mod vga;

/// Halt the CPU forever. Used by the panic handler and by fatal
/// exception handlers: a fault here has no safe context to resume.
pub fn hlt_loop() -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}

/// Print to the VGA text buffer.
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::vga::_print(format_args!($($arg)*)));
}

/// Print to the VGA text buffer, with a newline.
#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

/// Print to serial port COM1 (0x3F8).
#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => ($crate::serial::_print(format_args!($($arg)*)));
}

/// Print to serial port COM1 (0x3F8), with a newline.
#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($($arg:tt)*) => ($crate::serial_print!("{}\n", format_args!($($arg)*)));
}
