//! Serial output on COM1 (I/O port `0x3F8`, 16550 UART).
//!
//! Serial is the kernel's reliable log: unlike the VGA text buffer it
//! does not depend on the firmware leaving a text mode mapped, so it
//! works on both BIOS and UEFI boots. Run QEMU with `-serial stdio` to
//! see it on the host terminal.
//!
//! COM1 is also an input: the idle loop polls it (see
//! [`drain_serial_input`]) so keystrokes typed in the serial terminal
//! echo back there. The port is initialised lazily on first use; host
//! unit tests only exercise the pure echo policy ([`echo_for`]) and
//! never touch the hardware port.

use core::fmt;
use lazy_static::lazy_static;
use spin::Mutex;
use uart_16550::SerialPort;

/// I/O port of COM1.
pub const SERIAL_PORT_COM1: u16 = 0x3F8;

/// Maximum bytes drained from COM1 per idle-loop pass, so a flood of
/// serial input cannot starve the uptime print or the PS/2 drain.
const MAX_DRAIN_PER_PASS: usize = 64;

/// What to write back to the terminal for one byte received on COM1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Echo {
    /// Ignore the byte: nothing is echoed.
    None,
    /// Echo this byte as-is.
    Byte(u8),
    /// Echo a carriage return + line feed (what a terminal expects
    /// for a newline).
    Crlf,
}

/// Decide what to echo for one byte received on COM1.
///
/// Printable ASCII echoes as-is; carriage return and line feed echo
/// as CRLF; everything else (control bytes, escape sequences, high
/// bytes) is ignored so it is never reflected back at the terminal.
/// Pure function, so it is unit-tested on the host.
pub fn echo_for(byte: u8) -> Echo {
    match byte {
        b'\r' | b'\n' => Echo::Crlf,
        0x20..=0x7e => Echo::Byte(byte),
        _ => Echo::None,
    }
}

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

/// Read one byte from COM1 if one is waiting, without blocking.
/// Returns `None` when the receive buffer is empty.
pub fn try_read_byte() -> Option<u8> {
    SERIAL1.lock().try_receive().ok()
}

/// Drain any bytes waiting on COM1 and echo them back over serial.
///
/// Non-blocking: stops as soon as the receive buffer is empty (or
/// after [`MAX_DRAIN_PER_PASS`] bytes), so the idle loop keeps halting
/// and printing uptime. Called from the idle loop on every wake, which
/// is ~100x/sec from the PIT. The hardware access itself cannot be
/// host-tested; the echo policy is covered by [`echo_for`] tests.
pub fn drain_serial_input() {
    let mut port = SERIAL1.lock();
    for _ in 0..MAX_DRAIN_PER_PASS {
        let byte = match port.try_receive() {
            Ok(byte) => byte,
            Err(_) => break,
        };
        match echo_for(byte) {
            Echo::None => {}
            Echo::Byte(b) => port.send(b),
            Echo::Crlf => {
                port.send(b'\r');
                port.send(b'\n');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printable_ascii_echoes_as_is() {
        assert_eq!(echo_for(b'a'), Echo::Byte(b'a'));
        assert_eq!(echo_for(b' '), Echo::Byte(b' '));
        assert_eq!(echo_for(b'~'), Echo::Byte(b'~'));
        assert_eq!(echo_for(b'0'), Echo::Byte(b'0'));
    }

    #[test]
    fn carriage_return_and_line_feed_echo_as_crlf() {
        assert_eq!(echo_for(b'\r'), Echo::Crlf);
        assert_eq!(echo_for(b'\n'), Echo::Crlf);
    }

    #[test]
    fn control_and_high_bytes_are_ignored() {
        assert_eq!(echo_for(0x00), Echo::None);
        assert_eq!(echo_for(0x01), Echo::None);
        assert_eq!(echo_for(0x1b), Echo::None); // ESC
        assert_eq!(echo_for(0x7f), Echo::None); // DEL
        assert_eq!(echo_for(0x80), Echo::None);
        assert_eq!(echo_for(0xff), Echo::None);
    }
}
