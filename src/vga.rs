//! VGA text-mode writer (memory at `0xb8000`, 80x25, 16 colours).
//!
//! The writer is built over an explicit buffer, so the *same* code runs
//! in the kernel (over the real VGA buffer) and in host unit tests (over
//! a fake buffer on the stack). The global [`WRITER`] is only ever
//! constructed over the real hardware address; tests must use
//! [`Writer::new`] with their own buffer and never touch the global.
//!
//! Volatile access is done directly with `core::ptr::read_volatile` /
//! `write_volatile` instead of the `volatile` crate: the crate's API
//! changed across 0.4/0.5/0.6 (reference-wrapping `Volatile` vs
//! `VolatileRef`), and a plain `[[ScreenChar; 80]; 25]` buffer plus two
//! pointer intrinsics is stable across all of them.

use core::fmt;
use lazy_static::lazy_static;
use spin::Mutex;

/// Number of character cells per VGA text row.
pub const BUFFER_WIDTH: usize = 80;
/// Number of rows in the VGA text buffer.
pub const BUFFER_HEIGHT: usize = 25;

/// VGA text-mode colours (the low nibble is the foreground, the high
/// nibble the background, in a colour byte).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    Blue = 1,
    Green = 2,
    Cyan = 3,
    Red = 4,
    Magenta = 5,
    Brown = 6,
    LightGray = 7,
    DarkGray = 8,
    LightBlue = 9,
    LightGreen = 10,
    LightCyan = 11,
    LightRed = 12,
    Pink = 13,
    Yellow = 14,
    White = 15,
}

/// Pack a foreground/background pair into one VGA colour byte:
/// background in the high nibble, foreground in the low nibble.
pub const fn color_code(foreground: Color, background: Color) -> u8 {
    (background as u8) << 4 | (foreground as u8)
}

/// One VGA text cell: an ASCII byte plus a colour byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ScreenChar {
    pub ascii_character: u8,
    pub color_code: u8,
}

impl ScreenChar {
    pub const fn new(ascii_character: u8, color_code: u8) -> Self {
        Self {
            ascii_character,
            color_code,
        }
    }
}

/// The VGA text buffer: 25 rows of 80 cells.
///
/// Cells are accessed with volatile reads/writes (see [`Writer`]) so the
/// compiler never optimises the memory-mapped I/O away.
pub type Buffer = [[ScreenChar; BUFFER_WIDTH]; BUFFER_HEIGHT];

/// The VGA text buffer in the kernel's address space.
///
/// Writing to this address only works where the firmware left the VGA
/// text buffer mapped — BIOS boot and VGA-compatible firmware. Under a
/// pure UEFI/GOP boot there is no text buffer, and output goes to the
/// serial port instead (see `crate::serial`).
pub fn vga_buffer() -> &'static mut Buffer {
    // SAFETY: 0xb8000 is the conventional VGA text buffer address.
    // The caller must be running on hardware/firmware where it is mapped.
    unsafe { &mut *(0xb8000 as *mut Buffer) }
}

/// A writer over a VGA text buffer.
pub struct Writer {
    column_position: usize,
    row_position: usize,
    color_code: u8,
    buffer: &'static mut Buffer,
}

impl Writer {
    /// Create a writer over `buffer` (real hardware in the kernel, a
    /// fake buffer in tests).
    pub fn new(buffer: &'static mut Buffer) -> Self {
        Writer {
            column_position: 0,
            row_position: 0,
            color_code: color_code(Color::White, Color::Black),
            buffer,
        }
    }

    /// Write one byte, handling `\n` and wrapping at the row edge.
    /// Non-printable bytes are rendered as `0xFE` (■).
    pub fn write_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.new_line(),
            byte => {
                if self.column_position >= BUFFER_WIDTH {
                    self.new_line();
                }
                let row = self.row_position;
                let col = self.column_position;
                let color_code = self.color_code;
                let character = ScreenChar {
                    ascii_character: match byte {
                        0x20..=0x7e => byte,
                        _ => 0xfe,
                    },
                    color_code,
                };
                // Volatile write: this is memory-mapped I/O, the
                // compiler must not elide or reorder it.
                unsafe {
                    core::ptr::write_volatile(&mut self.buffer[row][col], character);
                }
                self.column_position += 1;
            }
        }
    }

    /// Write a whole string, byte by byte.
    pub fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
    }

    /// Move to the next row, scrolling the buffer up when full.
    fn new_line(&mut self) {
        if self.row_position >= BUFFER_HEIGHT - 1 {
            // Scroll: copy every row up by one, with volatile I/O.
            for row in 1..BUFFER_HEIGHT {
                for col in 0..BUFFER_WIDTH {
                    let character =
                        unsafe { core::ptr::read_volatile(&self.buffer[row][col]) };
                    unsafe {
                        core::ptr::write_volatile(&mut self.buffer[row - 1][col], character);
                    }
                }
            }
            self.clear_row(BUFFER_HEIGHT - 1);
            self.column_position = 0;
        } else {
            self.row_position += 1;
            self.column_position = 0;
        }
    }

    /// Blank one row with the writer's colour.
    fn clear_row(&mut self, row: usize) {
        let blank = ScreenChar {
            ascii_character: b' ',
            color_code: self.color_code,
        };
        for col in 0..BUFFER_WIDTH {
            unsafe {
                core::ptr::write_volatile(&mut self.buffer[row][col], blank);
            }
        }
    }

    /// Read back one cell (tests and diagnostics).
    pub fn read_char(&self, row: usize, col: usize) -> ScreenChar {
        unsafe { core::ptr::read_volatile(&self.buffer[row][col]) }
    }
}

impl fmt::Write for Writer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_string(s);
        Ok(())
    }
}

lazy_static! {
    /// The kernel's global VGA writer, over the real hardware buffer.
    ///
    /// Only constructed on first use, and only in the kernel: host unit
    /// tests never call `print!`/`println!`, so they never touch
    /// `0xb8000`.
    pub static ref WRITER: Mutex<Writer> = Mutex::new(Writer::new(vga_buffer()));
}

/// Write formatted arguments to the global VGA writer.
/// Used by the `print!` / `println!` macros.
#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    WRITER.lock().write_fmt(args).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

    /// A fake VGA buffer that lives long enough to hand to `Writer::new`.
    fn fake_buffer() -> &'static mut Buffer {
        let blank = ScreenChar::new(b' ', color_code(Color::White, Color::Black));
        Box::leak(Box::new([[blank; BUFFER_WIDTH]; BUFFER_HEIGHT]))
    }

    #[test]
    fn colour_code_packs_background_high_foreground_low() {
        assert_eq!(color_code(Color::White, Color::Black), 0x0f);
        assert_eq!(color_code(Color::Black, Color::White), 0xf0);
        assert_eq!(color_code(Color::Yellow, Color::Blue), 0x1e);
        assert_eq!(color_code(Color::Red, Color::Red), 0x44);
    }

    #[test]
    fn colour_values_match_vga_spec() {
        assert_eq!(Color::Black as u8, 0);
        assert_eq!(Color::LightGray as u8, 7);
        assert_eq!(Color::White as u8, 15);
    }

    #[test]
    fn screen_char_stores_ascii_and_colour() {
        let ch = ScreenChar::new(b'A', 0x0f);
        assert_eq!(ch.ascii_character, b'A');
        assert_eq!(ch.color_code, 0x0f);
    }

    #[test]
    fn writer_places_characters_in_order() {
        let mut writer = Writer::new(fake_buffer());
        writer.write_string("hi");
        assert_eq!(writer.read_char(0, 0).ascii_character, b'h');
        assert_eq!(writer.read_char(0, 1).ascii_character, b'i');
        assert_eq!(writer.read_char(0, 2).ascii_character, b' ');
    }

    #[test]
    fn writer_handles_newlines() {
        let mut writer = Writer::new(fake_buffer());
        writer.write_string("ab\ncd");
        assert_eq!(writer.read_char(0, 0).ascii_character, b'a');
        assert_eq!(writer.read_char(0, 1).ascii_character, b'b');
        assert_eq!(writer.read_char(1, 0).ascii_character, b'c');
        assert_eq!(writer.read_char(1, 1).ascii_character, b'd');
    }

    #[test]
    fn writer_renders_non_printable_bytes_as_block() {
        let mut writer = Writer::new(fake_buffer());
        writer.write_byte(0x01);
        assert_eq!(writer.read_char(0, 0).ascii_character, 0xfe);
    }

    #[test]
    fn writer_wraps_at_end_of_row() {
        let mut writer = Writer::new(fake_buffer());
        writer.write_string(&"x".repeat(BUFFER_WIDTH));
        // The 80th 'x' filled row 0; the next byte wraps to row 1.
        writer.write_byte(b'y');
        assert_eq!(writer.read_char(0, BUFFER_WIDTH - 1).ascii_character, b'x');
        assert_eq!(writer.read_char(1, 0).ascii_character, b'y');
    }

    #[test]
    fn writer_scrolls_when_full() {
        let mut writer = Writer::new(fake_buffer());
        for row in 0..BUFFER_HEIGHT {
            writer.write_string(&format!("row{row:02}"));
            if row < BUFFER_HEIGHT - 1 {
                writer.write_byte(b'\n');
            }
        }
        // One more line forces a scroll: row 0 must now be the old row 1.
        writer.write_byte(b'\n');
        assert_eq!(writer.read_char(0, 0).ascii_character, b'r');
        assert_eq!(writer.read_char(0, 3).ascii_character, b'0');
        assert_eq!(writer.read_char(0, 4).ascii_character, b'1');
    }

    #[test]
    fn writer_implements_fmt_write() {
        use core::fmt::Write;
        let mut writer = Writer::new(fake_buffer());
        write!(writer, "n={}", 42).unwrap();
        assert_eq!(writer.read_char(0, 0).ascii_character, b'n');
        assert_eq!(writer.read_char(0, 2).ascii_character, b'4');
        assert_eq!(writer.read_char(0, 3).ascii_character, b'2');
    }
}
