//! Serial output on COM1 (I/O port `0x3F8`, 16550 UART).
//!
//! Serial is the kernel's reliable log: unlike the VGA text buffer it
//! does not depend on the firmware leaving a text mode mapped, so it
//! works on both BIOS and UEFI boots. Run QEMU with `-serial stdio` to
//! see it on the host terminal.
//!
//! The port is initialised lazily on first use; host unit tests never
//! call into this module, so they never touch the hardware port.

use core::fmt;
use lazy_static::lazy_static;
use spin::Mutex;
use uart_16550::SerialPort;

/// I/O port of COM1.
pub const SERIAL_PORT_COM1: u16 = 0x3F8;

lazy_static! {
    /// The kernel's global serial port.
    pub static ref SERIAL1: Mutex<SerialPort> = {
        let mut serial_port = unsafe { SerialPort::new(SERIAL_PORT_COM1) };
        serial_port.init();
        Mutex::new(serial_port)
    };
}

/// Write formatted arguments to COM1.
/// Used by the `serial_print!` / `serial_println!` macros.
#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    SERIAL1
        .lock()
        .write_fmt(args)
        .expect("printing to serial failed");
}
