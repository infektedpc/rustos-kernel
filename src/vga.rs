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
use core::sync::atomic::{AtomicBool, Ordering};
use lazy_static::lazy_static;
use spin::Mutex;
use x86_64::structures::paging::{FrameAllocator, Mapper, OffsetPageTable, Page, Size4KiB};
use x86_64::VirtAddr;

/// Physical address of the VGA text-mode buffer (80x25, 16 colours).
pub const VGA_TEXT_BUFFER_ADDR: u64 = 0xb8000;

/// Whether the VGA text buffer is safe to write to.
///
/// The bootloader never maps `0xb8000` under UEFI/GOP, so the buffer
/// is unmapped when the kernel starts: the first VGA write then
/// page-faults. [`init`] maps it (or finds it already mapped) and
/// sets this flag; [`_print`] drops all VGA output while the flag is
/// clear, so the kernel degrades to serial-only output instead of
/// faulting on a hope. Defaults to `false`: VGA stays off until the
/// mapping is proven, never the other way round.
static VGA_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Whether the VGA text buffer is mapped and safe to write to.
pub fn vga_available() -> bool {
    VGA_AVAILABLE.load(Ordering::SeqCst)
}

/// Map the VGA text buffer so [`_print`] can use it, and report
/// whether it is now safe to write to.
///
/// The page at `0xb8000` is mapped identity-mapped (virtual address
/// == physical address) with PRESENT | WRITABLE | NO_CACHE, via
/// [`crate::memory::map_mmio_page`]. If the bootloader or firmware
/// already mapped it (as on the first live boot, 2026-11-21), the
/// existing mapping is kept and this just records that VGA is usable.
///
/// Returns `true` when VGA output is safe, `false` when the mapping
/// could not be established — in that case the caller must not print
/// to VGA at all, and the kernel runs serial-only. The mapping itself
/// cannot be host-tested; see the module tests for what is covered.
///
/// # Safety
///
/// The caller must pass the kernel's live page-table mapper and a
/// frame allocator over genuinely free frames (only used for the
/// intermediate page-table frames `map_to` may need).
pub fn init(
    mapper: &mut OffsetPageTable,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> bool {
    let page: Page<Size4KiB> =
        Page::containing_address(VirtAddr::new(VGA_TEXT_BUFFER_ADDR));
    // Already mapped (firmware/bootloader left it)? Keep it and
    // record VGA as usable — mapping it again would fail with
    // `PageAlreadyMapped` and wrongly force serial-only mode.
    if mapper.translate_page(page).is_ok() {
        VGA_AVAILABLE.store(true, Ordering::SeqCst);
        return true;
    }
    match crate::memory::map_mmio_page(mapper, frame_allocator, VGA_TEXT_BUFFER_ADDR) {
        Ok(()) => {
            VGA_AVAILABLE.store(true, Ordering::SeqCst);
            true
        }
        Err(_) => false,
    }
}

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
/// Writing to this address only works after [`init`] has mapped it
/// (or found it already mapped). Before that, [`_print`] refuses to
/// write at all — see `VGA_AVAILABLE`. Callers must never touch this
/// buffer except through [`_print`] / the `print!` macros, which are
/// guarded by the flag.
pub fn vga_buffer() -> &'static mut Buffer {
    // SAFETY: 0xb8000 is the conventional VGA text buffer address.
    // The caller must have mapped it first (see `init`); the global
    // writer is only reached through `_print`, which checks that.
    unsafe { &mut *(VGA_TEXT_BUFFER_ADDR as *mut Buffer) }
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
///
/// Silently drops the VGA output while the VGA text buffer is
/// unmapped (see `VGA_AVAILABLE` / [`init`]): writing to `0xb8000`
/// before the kernel maps it page-faults, and a fault before the
/// IDT is up is a triple fault. Serial output is a separate path
/// (`crate::serial`) and is unaffected.
///
/// The framebuffer console is mirrored unconditionally (Batch 10):
/// the same arguments go to the TTY layer through
/// [`crate::fbcon::write_fmt`], so every kernel `println!` also
/// renders on the GOP framebuffer. That write is a no-op while the
/// fb console is uninitialised, so early prints (before
/// `fbcon::init`) and serial-only boots are unaffected.
///
/// Holds [`crate::interrupts::IrqGuard`] while the writer is locked:
/// a tick preempting a VGA print mid-line and switching to a ring-3
/// task would deadlock (the task's `write` syscall spins on this
/// lock with interrupts off) — see `IrqGuard` (Batch 8 wedge).
/// Lock order is VGA then TTY; no path takes them the other way
/// round.
#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    let _irq = crate::interrupts::IrqGuard::hold();
    if VGA_AVAILABLE.load(Ordering::SeqCst) {
        WRITER.lock().write_fmt(args).unwrap();
    }
    crate::fbcon::write_fmt(args);
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

    #[test]
    fn vga_is_unavailable_until_init_maps_it() {
        // The flag starts clear and only `init` (which needs a real
        // page-table mapper, so it cannot run on the host) can set it.
        assert!(!vga_available());
    }

    #[test]
    fn print_is_a_noop_while_vga_is_unavailable() {
        // This is the guard that stops the 2026-11-21 triple fault:
        // `_print` must never touch `0xb8000` while the buffer is
        // unmapped. On the host, touching that address would segfault
        // the test process, so merely returning from this call proves
        // the guard holds.
        assert!(!vga_available());
        _print(format_args!("this must never reach 0xb8000"));
    }
}
