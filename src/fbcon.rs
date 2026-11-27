//! Framebuffer console: the kernel's `Terminal` over the bootloader's
//! GOP framebuffer.
//!
//! The bootloader hands the kernel a framebuffer in `BootInfo`
//! (1280x800 Bgr on QEMU). [`init`] builds a
//! [`rustos_fb::Framebuffer`] over it and a [`rustos_tty::Terminal`]
//! over that, and stores the terminal in the global [`FBCON`]. From
//! then on every kernel console print is mirrored to the framebuffer:
//! [`vga::_print`] (the `print!`/`println!` macros) forwards its
//! formatted arguments here, and the syscall `write` path
//! (`KernelDevices::write_console`) forwards userspace stdout bytes
//! here — so the shell's prompt, echo, and output render through the
//! TTY layer too, with VT escapes interpreted.
//!
//! The framebuffer is purely additive: serial and VGA keep working
//! exactly as before, and if the framebuffer is missing or an
//! unsupported format [`init`] returns `None` and the kernel runs
//! serial+VGA-only. The terminal is only ever written while holding
//! [`crate::interrupts::IrqGuard`]: the lock discipline is serial
//! → VGA → TTY, never the reverse (see the Batch 8 wedge — a
//! console lock held by a task that gets preempted deadlocks the
//! ring-3 task the tick switches to). All functions in this module
//! that lock [`FBCON`] assume the caller already holds the guard,
//! except [`init`], which takes it itself (it runs before `sti`).
//!
//! Live proof for a `-display none` boot: the kernel logs the
//! framebuffer geometry at init and a [`FbStats`] snapshot
//! (glyphs drawn, lit pixels, FNV checksum of the pixel buffer) to
//! serial after the ready line and again ~5 s in. A changing,
//! non-zero checksum proves the console is really rendering.

use bootloader_api::info::{FrameBuffer, PixelFormat as BootPixelFormat};
use core::fmt;
use rustos_fb::{Framebuffer, FrameBufferInfo as FbInfo, PixelFormat as FbPixelFormat};
use rustos_tty::Terminal;
use spin::Mutex;

/// The kernel's framebuffer terminal, once [`init`] has run.
static FBCON: Mutex<Option<Terminal>> = Mutex::new(None);

/// What [`init`] established about the framebuffer.
#[derive(Debug, Clone, Copy)]
pub struct FbInit {
    /// Framebuffer width in pixels.
    pub width: usize,
    /// Framebuffer height in pixels.
    pub height: usize,
    /// Pixels per line (may exceed `width`).
    pub stride: usize,
    /// Virtual address of the pixel buffer.
    pub addr: usize,
    /// Pixel format, as a name for the serial log.
    pub format: &'static str,
    /// Text columns that fit.
    pub cols: usize,
    /// Text rows that fit.
    pub rows: usize,
}

/// Live-proof snapshot of the framebuffer console (see [`stats`]).
#[derive(Debug, Clone, Copy)]
pub struct FbStats {
    /// Framebuffer width in pixels.
    pub width: usize,
    /// Framebuffer height in pixels.
    pub height: usize,
    /// Text columns that fit.
    pub cols: usize,
    /// Text rows that fit.
    pub rows: usize,
    /// Glyphs drawn since the terminal was created.
    pub glyphs_drawn: u64,
    /// Pixels that differ from black.
    pub lit_pixels: u64,
    /// FNV-1a checksum of the pixel buffer.
    pub checksum: u64,
}

/// Whether the framebuffer console is initialised.
pub fn available() -> bool {
    FBCON.lock().is_some()
}

/// Initialise the framebuffer console from the bootloader's
/// framebuffer. Returns the geometry on success, `None` when the
/// framebuffer is unusable (unsupported format — anything but 32-bit
/// Rgb/Bgr — zero size, or a buffer too short to hold it). `None`
/// is a normal outcome: the caller logs it and keeps going
/// serial+VGA-only.
///
/// Needs the kernel heap (the terminal's shadow cell buffer is a
/// `Vec`), so the kernel calls this after `init_heap`.
pub fn init(fb: &mut FrameBuffer) -> Option<FbInit> {
    let _irq = crate::interrupts::IrqGuard::hold();
    let info = fb.info();
    let (format, name) = match info.pixel_format {
        BootPixelFormat::Rgb => (FbPixelFormat::Rgb, "Rgb"),
        BootPixelFormat::Bgr => (FbPixelFormat::Bgr, "Bgr"),
        // U8 (grayscale) and Unknown bit layouts: unsupported.
        _ => return None,
    };
    if info.bytes_per_pixel != 4 || info.width == 0 || info.height == 0 {
        return None;
    }
    let fb_info = FbInfo {
        byte_len: info.byte_len,
        width: info.width,
        height: info.height,
        pixel_format: format,
        bytes_per_pixel: 4,
        stride: info.stride,
    };
    let buffer = fb.buffer_mut();
    let addr = buffer.as_ptr() as usize;
    let framebuffer = Framebuffer::from_bytes(buffer, fb_info)?;
    let term = Terminal::new(framebuffer);
    let (cols, rows) = (term.cols(), term.rows());
    *FBCON.lock() = Some(term);
    Some(FbInit {
        width: info.width,
        height: info.height,
        stride: info.stride,
        addr,
        format: name,
        cols,
        rows,
    })
}

/// Write bytes to the framebuffer terminal, if initialised.
/// No-op while it is not. The caller must hold
/// [`crate::interrupts::IrqGuard`] (see the module docs).
pub fn write_bytes(bytes: &[u8]) {
    if let Some(term) = FBCON.lock().as_mut() {
        term.write_bytes(bytes);
    }
}

/// Write a string to the framebuffer terminal, if initialised.
/// Same guard rule as [`write_bytes`].
pub fn write_str(s: &str) {
    write_bytes(s.as_bytes());
}

/// Write formatted arguments to the framebuffer terminal, if
/// initialised. Same guard rule as [`write_bytes`]. Used by
/// [`crate::vga::_print`] so every kernel `println!` is mirrored.
pub fn write_fmt(args: fmt::Arguments) {
    if let Some(term) = FBCON.lock().as_mut() {
        use core::fmt::Write;
        let _ = term.write_fmt(args);
    }
}

/// Snapshot the console for the serial log: geometry, glyphs drawn,
/// lit pixels, and a checksum of the pixel buffer. `None` while the
/// console is uninitialised. The caller must hold
/// [`crate::interrupts::IrqGuard`] (see the module docs).
pub fn stats() -> Option<FbStats> {
    let term = FBCON.lock();
    let term = term.as_ref()?;
    Some(FbStats {
        width: term.fb_width(),
        height: term.fb_height(),
        cols: term.cols(),
        rows: term.rows(),
        glyphs_drawn: term.glyphs_drawn(),
        lit_pixels: term.lit_pixels(),
        checksum: term.checksum(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bootloader_api::info::{FrameBufferInfo as BootInfo, PixelFormat as BootFmt};

    /// A fake bootloader framebuffer over a `Vec<u32>`: 1280x800
    /// Bgr, the QEMU geometry. The `Vec` must stay alive while the
    /// `FrameBuffer` built from it is used.
    fn fake_boot_fb(
        buf: &mut [u32],
        format: BootFmt,
    ) -> FrameBuffer {
        let info = BootInfo {
            byte_len: buf.len() * 4,
            width: 1280,
            height: 800,
            pixel_format: format,
            bytes_per_pixel: 4,
            stride: 1280,
        };
        unsafe { FrameBuffer::new(buf.as_ptr() as u64, info) }
    }

    #[test]
    fn init_accepts_bgr_and_rejects_u8_then_writes() {
        let mut buf = alloc::vec![0u32; 1280 * 800];
        let mut fb = fake_boot_fb(&mut buf, BootFmt::Bgr);
        let info = init(&mut fb).expect("Bgr framebuffer must initialise");
        assert_eq!((info.width, info.height), (1280, 800));
        assert_eq!(info.format, "Bgr");
        assert_eq!((info.cols, info.rows), (160, 50));
        assert!(available());
        // The terminal is live: writing through the module draws
        // glyphs (the SGR self-test line the kernel writes at boot
        // exercises the parser the same way).
        {
            let _irq = crate::interrupts::IrqGuard::hold();
            write_str("\x1b[32mfb: console ready (SGR ok)\x1b[0m\n");
            write_bytes(b"hi");
        }
        let stats = stats().expect("stats after init");
        assert_eq!(stats.cols, 160);
        assert_eq!(stats.rows, 50);
        assert!(stats.glyphs_drawn > 0);
        assert!(stats.lit_pixels > 0);
        assert_ne!(stats.checksum, 0);

        // A grayscale framebuffer is rejected: init returns None.
        // (The global still holds the Bgr terminal from above —
        // init failure must not disturb it.)
        let mut buf2 = alloc::vec![0u32; 1280 * 800];
        let mut fb2 = fake_boot_fb(&mut buf2, BootFmt::U8);
        assert!(init(&mut fb2).is_none());
        assert!(available());
    }
}
