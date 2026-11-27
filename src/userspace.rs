//! First userspace program: loading and the ring-3 address layout.
//!
//! ## The user address space (single shared page table)
//!
//! There are no per-process page tables yet: the user program's
//! pages are mapped into the same level-4 table the kernel uses,
//! distinguished only by the `USER_ACCESSIBLE` bit. Ring 3 can
//! touch exactly these pages — the kernel's own pages (kernel
//! image, heap at `0x444444440000`, VGA) are supervisor-only, so a
//! userspace program cannot read or write them. The kernel, in
//! ring 0, can touch the user pages (SMAP is off), which is how the
//! loader copies the program in.
//!
//! Fixed layout per program slot (all user-accessible) — hello in
//! slot A, the shell in slot B (Batch 9); the two slots share one
//! page table, so their windows never overlap (see
//! [`UserLayout`] and [`layouts_overlap`]):
//!
//! | region | hello slot | shell slot | size | purpose |
//! |---|---|---|---|---|
//! | code/data (ELF segments) | `USER_CODE_BASE` (0x400000) | `SH_CODE_BASE` (0x800000) | ELF-defined | the program itself |
//! | user stack | `USER_STACK_TOP` − 16 KiB .. `USER_STACK_TOP` (0x800000) | `SH_STACK_TOP` − 16 KiB .. `SH_STACK_TOP` (0xC00000) | 16 KiB | the program's stack |
//! | user heap | `USER_HEAP_BASE` (0x1000000) | `SH_HEAP_BASE` (0x1100000) | 64 KiB | the program's allocator region |
//!
//! ## How the programs get in
//!
//! The kernel binary embeds the program ELFs (`rustos-hello` and
//! `rustos-sh`) — `build.rs` copies them into `OUT_DIR` and this
//! module `include_bytes!`s them. [`load_user_program`] parses one
//! ELF with the `rustos-process` ELF loader, maps each `PT_LOAD`
//! segment at its linked `vaddr` (each program is a static
//! `ET_EXEC` linked at its slot's code base, so the vaddrs are the
//! load addresses), maps the user stack and heap, writes the
//! [`StartInfo`] (heap base + size) onto the initial user stack,
//! and returns the entry point and initial user `rsp` for the
//! ring-3 task. Segments are loaded in two phases: pages are
//! mapped writable for the copy ([`SEGMENT_COPY_FLAGS`]) and only
//! then flipped to their final flags ([`protect_segment_pages`]) —
//! mapping a read-only code page first faults the copy itself,
//! because supervisor writes respect read-only pages (CR0.WP).
//!
//! ## The startup handshake
//!
//! The kernel sets the initial user `RSP` to
//! `USER_STACK_TOP - 24` (8 mod 16 — the System V entry
//! alignment) with the 16-byte [`StartInfo`] at `[rsp]` and 8
//! bytes of slack above it. The program's `_start` reads the
//! `StartInfo` from `[rsp]` and calls `rustos_libc::start`, which
//! installs the heap and runs `main`. Nothing else is handed over:
//! fds 0/1/2 are already open (they are the only fds that exist).
//!
//! ## Syscall buffer validation
//!
//! A userspace program that passes a bad address to a syscall must
//! get `-EFAULT`, not a kernel `#PF`: the dispatcher would fault
//! in ring 0 dereferencing it, and a ring-0 fault halts the whole
//! kernel. The loader records every mapped user interval in
//! [`USER_REGIONS`]; [`is_user_buffer`] checks a buffer against
//! them, and the syscall dispatcher calls it for ring-3 callers
//! (see `crate::syscall`).

use alloc::vec::Vec;
use rustos_process::elf::{parse_elf, ElfError, ProgramHeader};
use rustos_libc::StartInfo;
use spin::Mutex;
use x86_64::structures::paging::mapper::{FlagUpdateError, MapToError};
use x86_64::structures::paging::{
    FrameAllocator, Mapper, OffsetPageTable, Page, PageTableFlags, Size4KiB,
};
use x86_64::VirtAddr;

/// Base address of the user program's code/data (the ELF link base).
pub const USER_CODE_BASE: u64 = 0x400000;
/// One-past-the-end of the user image region: ELF segments must lie
/// below this (the user stack starts here).
pub const USER_IMAGE_END: u64 = 0x800000;
/// Top (one-past-the-end) of the user stack.
pub const USER_STACK_TOP: u64 = 0x800000;
/// Size of the user stack, in bytes (16 KiB).
pub const USER_STACK_SIZE: usize = 16 * 1024;
/// Base address of the user heap region.
pub const USER_HEAP_BASE: u64 = 0x100_0000;
/// Size of the user heap region, in bytes (64 KiB).
pub const USER_HEAP_SIZE: usize = 64 * 1024;
/// One-past-the-end of the whole user address space.
pub const USER_ADDRESS_SPACE_END: u64 = USER_HEAP_BASE + USER_HEAP_SIZE as u64;

/// Base address of the shell's code/data (its ELF link base).
/// The shell is the second program slot (Batch 9): its window sits
/// directly above hello's — code at 0x800000, where hello's stack
/// top ends.
pub const SH_CODE_BASE: u64 = 0x800000;
/// Top (one-past-the-end) of the shell's user stack.
pub const SH_STACK_TOP: u64 = 0xC00000;
/// Base address of the shell's heap region.
pub const SH_HEAP_BASE: u64 = 0x110_0000;
/// One-past-the-end of the shell's whole address space.
pub const SH_ADDRESS_SPACE_END: u64 = SH_HEAP_BASE + USER_HEAP_SIZE as u64;

/// The address layout of one userspace program slot: where its
/// code, stack, and heap live. Two programs share one page table
/// (there are no per-process tables yet), so each slot needs its
/// own non-overlapping window (see [`HELLO_LAYOUT`] and
/// [`SH_LAYOUT`], and [`layouts_overlap`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserLayout {
    /// Base address of the program's code/data (the ELF link base).
    pub code_base: u64,
    /// One-past-the-end of the program's image region (== the user
    /// stack top): ELF segments must lie below this.
    pub image_end: u64,
    /// Top (one-past-the-end) of the user stack.
    pub stack_top: u64,
    /// Size of the user stack, in bytes.
    pub stack_size: usize,
    /// Base address of the user heap region.
    pub heap_base: u64,
    /// Size of the user heap region, in bytes.
    pub heap_size: usize,
}

/// The hello slot (Batch 8): the original fixed layout.
pub const HELLO_LAYOUT: UserLayout = UserLayout {
    code_base: USER_CODE_BASE,
    image_end: USER_IMAGE_END,
    stack_top: USER_STACK_TOP,
    stack_size: USER_STACK_SIZE,
    heap_base: USER_HEAP_BASE,
    heap_size: USER_HEAP_SIZE,
};

/// The shell slot (Batch 9): the next window up — code at
/// 0x800000, stack top 0xC00000, heap 0x1100000. No interval
/// overlaps hello's (see [`layouts_overlap`]).
pub const SH_LAYOUT: UserLayout = UserLayout {
    code_base: SH_CODE_BASE,
    image_end: SH_STACK_TOP,
    stack_top: SH_STACK_TOP,
    stack_size: USER_STACK_SIZE,
    heap_base: SH_HEAP_BASE,
    heap_size: USER_HEAP_SIZE,
};

/// True when any interval of layout `a` (image region, stack,
/// heap) overlaps any interval of layout `b`. Pure, so host tests
/// can pin that the two program slots never share an address —
/// overlapping slots would map two programs' pages onto each
/// other in the shared page table.
pub fn layouts_overlap(a: &UserLayout, b: &UserLayout) -> bool {
    fn overlap(x: (u64, u64), y: (u64, u64)) -> bool {
        x.0 < y.1 && y.0 < x.1
    }
    let a_intervals = [
        (a.code_base, a.image_end),
        (a.stack_top - a.stack_size as u64, a.stack_top),
        (a.heap_base, a.heap_base + a.heap_size as u64),
    ];
    let b_intervals = [
        (b.code_base, b.image_end),
        (b.stack_top - b.stack_size as u64, b.stack_top),
        (b.heap_base, b.heap_base + b.heap_size as u64),
    ];
    a_intervals
        .iter()
        .any(|&x| b_intervals.iter().any(|&y| overlap(x, y)))
}

/// Offset of the [`StartInfo`] below the user stack top: the
/// initial user `rsp` is `USER_STACK_TOP - START_INFO_OFFSET`, so
/// `rsp` is 8 mod 16 (the System V entry alignment) and the
/// 16-byte `StartInfo` sits at `[rsp]` with 8 bytes of slack above
/// it (see the module docs).
pub const START_INFO_OFFSET: u64 = 24;

/// The embedded userspace program ELF (`rustos-hello`), placed in
/// `OUT_DIR` by `build.rs`. Empty when the program was not built
/// before the kernel — the loader then reports `MissingElf` and
/// the kernel boots without a userspace program (loudly, see
/// `kernel_main`).
#[cfg(not(test))]
pub static HELLO_ELF: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hello"));
#[cfg(test)]
pub static HELLO_ELF: &[u8] = &[];

/// The embedded userspace shell ELF (`rustos-sh`), placed in
/// `OUT_DIR` by `build.rs` (Batch 9). Same empty-placeholder rule
/// as [`HELLO_ELF`]: a missing shell boots without one, loudly.
#[cfg(not(test))]
pub static SH_ELF: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sh"));
#[cfg(test)]
pub static SH_ELF: &[u8] = &[];

/// Mapped user intervals, as (start, end) pairs, in the order the
/// loader mapped them: ELF segments, the user stack, the user heap.
/// [`is_user_buffer`] checks syscall buffers against these — a
/// buffer is only safe to dereference from the kernel if it lies
/// wholly inside one mapped interval.
static USER_REGIONS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Record one mapped user interval (start inclusive, end
/// exclusive). Called by the loader for every interval it maps;
/// host tests call it directly to set up fake regions.
pub fn register_user_region(start: u64, end: u64) {
    USER_REGIONS.lock().push((start, end));
}

/// All mapped user intervals, for diagnostics (the boot log prints
/// them when the program loads).
pub fn user_regions() -> Vec<(u64, u64)> {
    USER_REGIONS.lock().clone()
}

/// True when `[ptr, ptr+len)` lies wholly inside one mapped user
/// interval — i.e. the kernel may dereference it on behalf of a
/// ring-3 caller. Zero-length buffers are always safe (nothing is
/// dereferenced). An overflowing range is never safe.
pub fn is_user_buffer(ptr: u64, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    let end = match (ptr as u128).checked_add(len as u128) {
        Some(e) if e <= u64::MAX as u128 => e as u64,
        _ => return false,
    };
    USER_REGIONS
        .lock()
        .iter()
        .any(|&(s, e)| ptr >= s && end <= e)
}

/// What went wrong loading the userspace program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    /// No program ELF was embedded (rustos-hello was not built
    /// before the kernel).
    MissingElf,
    /// The embedded bytes are not a loadable ELF64.
    Elf(ElfError),
    /// A `PT_LOAD` segment lies outside the user image region
    /// (`USER_CODE_BASE`..`USER_IMAGE_END`) or is empty.
    SegmentOutsideUserRange,
    /// A page could not be mapped.
    MapFailed,
    /// The frame allocator ran out of usable frames.
    FrameAllocationFailed,
}

impl From<ElfError> for LoadError {
    fn from(e: ElfError) -> Self {
        LoadError::Elf(e)
    }
}

impl From<MapToError<Size4KiB>> for LoadError {
    fn from(e: MapToError<Size4KiB>) -> Self {
        match e {
            MapToError::FrameAllocationFailed => LoadError::FrameAllocationFailed,
            _ => LoadError::MapFailed,
        }
    }
}

impl From<FlagUpdateError> for LoadError {
    fn from(_e: FlagUpdateError) -> Self {
        LoadError::MapFailed
    }
}

/// Final flags for one user page, from the ELF segments covering
/// it — the union of their permissions: writable if any covering
/// segment is `PF_W`, executable unless no covering segment is
/// `PF_X`. ELF segments share pages (hello's rodata page is also
/// the tail of its code segment; its data page is shared by three
/// segments), so per-segment flags would be wrong for the shared
/// page — the union is what the page must end up with. Pure, so
/// host tests can pin it. A page no segment covers gets the safe
/// default: readable, never writable, never executable.
pub fn page_flags_for_segments(segments: &[ProgramHeader], page_start: u64) -> PageTableFlags {
    let page_end = page_start + 0x1000;
    let mut writable = false;
    let mut executable = false;
    for seg in segments {
        if seg.memsz == 0 {
            continue;
        }
        let seg_end = seg.vaddr + seg.memsz;
        if seg.vaddr < page_end && seg_end > page_start {
            if seg.is_writable() {
                writable = true;
            }
            if seg.is_executable() {
                executable = true;
            }
        }
    }
    let mut flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;
    if writable {
        flags |= PageTableFlags::WRITABLE;
    }
    if !executable {
        flags |= PageTableFlags::NO_EXECUTE;
    }
    flags
}

/// Apply the final flags to every page the segments were copied
/// into: each page gets the union of the permissions of the
/// segments covering it (see [`page_flags_for_segments`]), so code
/// ends up read-only + executable and data writable +
/// non-executable, even where segments share a page. Must run
/// after all segments are copied and before the program runs —
/// the copy phase leaves every segment page writable (see
/// [`SEGMENT_COPY_FLAGS`]).
fn protect_segment_pages(
    mapper: &mut OffsetPageTable<'static>,
    segments: &[ProgramHeader],
    mapped: &[u64],
) -> Result<(), LoadError> {
    for &page_start in mapped {
        let flags = page_flags_for_segments(segments, page_start);
        let page: Page<Size4KiB> = Page::containing_address(VirtAddr::new(page_start));
        unsafe {
            mapper.update_flags(page, flags)?.flush();
        }
    }
    Ok(())
}

/// A loaded userspace program, ready to run as a ring-3 task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadedUserProgram {
    /// Entry point (the ELF `e_entry`): the `RIP` of the ring-3
    /// iretq frame.
    pub entry: u64,
    /// Initial user stack pointer (`USER_STACK_TOP - 24`, with the
    /// [`StartInfo`] at `[rsp]`).
    pub user_rsp: u64,
    /// Base of the user heap region (for the boot log).
    pub heap_base: u64,
    /// Size of the user heap region.
    pub heap_size: usize,
    /// Number of ELF segments loaded.
    pub segments: usize,
}

/// Map flags used while copying a segment in: every segment page
/// is writable (and never executable) until its bytes are in.
/// The final flags are applied afterwards by
/// [`protect_segment_pages`].
///
/// This two-phase load is not optional: the CPU faults a
/// supervisor write to a read-only page (CR0.WP is set), so
/// mapping a read-only code segment first and then `memcpy`ing
/// the file bytes into it dies with a protection-violation
/// #PF at the first byte (Batch 8 live boot: fault at 0x400000,
/// RIP in `memcpy`). Copying through writable pages and then
/// flipping the flags keeps the final protection intact.
pub const SEGMENT_COPY_FLAGS: PageTableFlags = PageTableFlags::from_bits_truncate(
    PageTableFlags::PRESENT.bits()
        | PageTableFlags::USER_ACCESSIBLE.bits()
        | PageTableFlags::WRITABLE.bits()
        | PageTableFlags::NO_EXECUTE.bits(),
);
/// Map one 4 KiB user page at `page_start` (page-aligned), with the
/// given flags, allocating a fresh frame. `mapped` tracks the pages
/// already mapped (ELF segments can share a page) so a page is
/// never mapped twice — a second `map_to` on a live page fails.
fn map_user_page(
    mapper: &mut OffsetPageTable<'static>,
    allocator: &mut impl FrameAllocator<Size4KiB>,
    mapped: &mut Vec<u64>,
    page_start: u64,
    flags: PageTableFlags,
) -> Result<(), LoadError> {
    if mapped.contains(&page_start) {
        return Ok(());
    }
    let page = Page::containing_address(VirtAddr::new(page_start));
    let frame = allocator
        .allocate_frame()
        .ok_or(LoadError::FrameAllocationFailed)?;
    unsafe {
        mapper.map_to(page, frame, flags, allocator)?.flush();
    }
    mapped.push(page_start);
    Ok(())
}

/// Map the pages of one ELF segment and copy its bytes in.
///
/// `seg` must lie inside the user image region (checked by the
/// caller). Pages are mapped with [`SEGMENT_COPY_FLAGS`]
/// (writable, never executable) for the copy and left that way —
/// the caller applies the final per-page flags with
/// [`protect_segment_pages`] once every segment is copied, so a
/// page shared by two segments ends up with the union of their
/// permissions. File bytes are copied to their `vaddr` (the
/// kernel can write user pages — SMAP is off) and the
/// `memsz - filesz` BSS tail is zero-filled. The segment's
/// vaddr need not be page-aligned (the hello ELF's segments
/// start mid-page): the pages are the containing ones, and the
/// copy goes to the exact vaddr.
fn load_segment(
    mapper: &mut OffsetPageTable<'static>,
    allocator: &mut impl FrameAllocator<Size4KiB>,
    mapped: &mut Vec<u64>,
    seg: &ProgramHeader,
    file: &[u8],
) -> Result<(), LoadError> {
    // Map every page the segment touches with the writable copy
    // flags, then copy. Mapping final (possibly read-only) flags
    // first would fault the copy itself (see SEGMENT_COPY_FLAGS).
    let flags = SEGMENT_COPY_FLAGS;
    if seg.memsz == 0 {
        return Ok(());
    }
    let first = seg.vaddr & !0xFFF;
    let last = (seg.vaddr + seg.memsz - 1) & !0xFFF;
    let mut page = first;
    loop {
        map_user_page(mapper, allocator, mapped, page, flags)?;
        if page == last {
            break;
        }
        page += 0x1000;
    }
    // File bytes, then the zero-filled BSS tail.
    let file_start = seg.offset as usize;
    let dest = seg.vaddr as *mut u8;
    unsafe {
        core::ptr::copy_nonoverlapping(file.as_ptr().add(file_start), dest, seg.filesz as usize);
        core::ptr::write_bytes(dest.add(seg.filesz as usize), 0, seg.bss_size() as usize);
    }
    Ok(())
}

/// Map a whole user region (stack or heap) of `size` bytes ending at
/// `top` (one-past-the-end), with the given flags. Returns the
/// region's start address.
fn map_user_region(
    mapper: &mut OffsetPageTable<'static>,
    allocator: &mut impl FrameAllocator<Size4KiB>,
    mapped: &mut Vec<u64>,
    top: u64,
    size: usize,
    flags: PageTableFlags,
) -> Result<u64, LoadError> {
    let start = top - size as u64;
    let mut page = start & !0xFFF;
    let last = (top - 1) & !0xFFF;
    loop {
        map_user_page(mapper, allocator, mapped, page, flags)?;
        if page == last {
            break;
        }
        page += 0x1000;
    }
    Ok(start)
}

/// Load one embedded userspace program into its slot: parse its
/// ELF, map its segments, its stack, and its heap into
/// user-accessible pages, and write the [`StartInfo`] onto the
/// initial user stack.
///
/// `elf` is the program's ELF bytes (see [`HELLO_ELF`] /
/// [`SH_ELF`]) and `layout` is the slot it lives in (see
/// [`HELLO_LAYOUT`] / [`SH_LAYOUT`]) — the two programs share one
/// page table, so their layouts must not overlap (see
/// [`layouts_overlap`]). The caller (kernel_main) owns the mapper
/// and frame allocator; the loaded program is registered as a
/// ring-3 task by the caller (see `crate::interrupts::add_user_task`).
pub fn load_user_program(
    mapper: &mut OffsetPageTable<'static>,
    allocator: &mut impl FrameAllocator<Size4KiB>,
    elf: &[u8],
    layout: &UserLayout,
) -> Result<LoadedUserProgram, LoadError> {
    if elf.is_empty() {
        return Err(LoadError::MissingElf);
    }
    let parsed = parse_elf(elf)?;
    if parsed.segments.is_empty() {
        return Err(LoadError::Elf(ElfError::BadSegmentSize));
    }
    let mut mapped: Vec<u64> = Vec::new();
    // Segments first: each must lie inside the slot's image region.
    for seg in &parsed.segments {
        let seg_end = seg.vaddr.checked_add(seg.memsz).ok_or(LoadError::SegmentOutsideUserRange)?;
        if seg.vaddr < layout.code_base || seg_end > layout.image_end || seg.memsz == 0 {
            return Err(LoadError::SegmentOutsideUserRange);
        }
        // (The segment must be readable: a non-readable PT_LOAD is
        // nonsense for a program the CPU has to fetch from.)
        if !seg.is_readable() {
            return Err(LoadError::SegmentOutsideUserRange);
        }
        load_segment(mapper, allocator, &mut mapped, seg, elf)?;
        register_user_region(seg.vaddr, seg_end);
    }
    // Copy done: flip every segment page to its final flags
    // (union over the segments covering it) before the program
    // can run — the copy phase left them all writable.
    protect_segment_pages(mapper, &parsed.segments, &mapped)?;
    // User stack: 16 KiB, writable, never executable.
    let stack_flags = PageTableFlags::PRESENT
        | PageTableFlags::WRITABLE
        | PageTableFlags::USER_ACCESSIBLE
        | PageTableFlags::NO_EXECUTE;
    let stack_start = map_user_region(
        mapper,
        allocator,
        &mut mapped,
        layout.stack_top,
        layout.stack_size,
        stack_flags,
    )?;
    register_user_region(stack_start, layout.stack_top);
    // User heap: 64 KiB, writable, never executable.
    let heap_start = map_user_region(
        mapper,
        allocator,
        &mut mapped,
        layout.heap_base + layout.heap_size as u64,
        layout.heap_size,
        stack_flags,
    )?;
    register_user_region(heap_start, layout.heap_base + layout.heap_size as u64);
    // Startup handshake: the StartInfo (heap base + size) on the
    // initial user stack, at rsp = top - 24 (8 mod 16, see the
    // module docs). The program's _start reads it from [rsp].
    let user_rsp = layout.stack_top - START_INFO_OFFSET;
    let info = StartInfo::new(layout.heap_base as *mut u8, layout.heap_size);
    unsafe {
        core::ptr::write(user_rsp as *mut StartInfo, info);
    }
    Ok(LoadedUserProgram {
        entry: parsed.entry_point,
        user_rsp,
        heap_base: layout.heap_base,
        heap_size: layout.heap_size,
        segments: parsed.segments.len(),
    })
}

/// Map flags for a segment, from its ELF flags — pure, so host
/// tests can pin the mapping (R_X -> executable, R_W -> writable,
/// everything user-accessible, non-executable unless PF_X).
pub fn segment_flags(seg: &ProgramHeader) -> PageTableFlags {
    let mut flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;
    if seg.is_writable() {
        flags |= PageTableFlags::WRITABLE;
    }
    if !seg.is_executable() {
        flags |= PageTableFlags::NO_EXECUTE;
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustos_process::elf::{PF_R, PF_W, PF_X};

    fn fake_header(flags: u32) -> ProgramHeader {
        ProgramHeader {
            p_type: 1,
            flags,
            offset: 0x1000,
            vaddr: 0x400000,
            paddr: 0x400000,
            filesz: 0x100,
            memsz: 0x100,
            align: 0x1000,
        }
    }

    #[test]
    fn segment_flags_match_elf_permissions() {
        // R_X: executable, not writable.
        let f = segment_flags(&fake_header(PF_R | PF_X));
        assert!(f.contains(PageTableFlags::PRESENT));
        assert!(f.contains(PageTableFlags::USER_ACCESSIBLE));
        assert!(!f.contains(PageTableFlags::WRITABLE));
        assert!(!f.contains(PageTableFlags::NO_EXECUTE));
        // R_W: writable, never executable.
        let f = segment_flags(&fake_header(PF_R | PF_W));
        assert!(f.contains(PageTableFlags::WRITABLE));
        assert!(f.contains(PageTableFlags::NO_EXECUTE));
        // R only: neither writable nor executable.
        let f = segment_flags(&fake_header(PF_R));
        assert!(!f.contains(PageTableFlags::WRITABLE));
        assert!(f.contains(PageTableFlags::NO_EXECUTE));
    }

    #[test]
    fn segment_copy_flags_are_writable_and_nonexecutable() {
        // The copy phase must always be able to write (even into
        // read-only code pages) and must never execute: the fault
        // this guards is the Batch 8 live-boot #PF at 0x400000.
        assert!(SEGMENT_COPY_FLAGS.contains(PageTableFlags::PRESENT));
        assert!(SEGMENT_COPY_FLAGS.contains(PageTableFlags::USER_ACCESSIBLE));
        assert!(SEGMENT_COPY_FLAGS.contains(PageTableFlags::WRITABLE));
        assert!(SEGMENT_COPY_FLAGS.contains(PageTableFlags::NO_EXECUTE));
    }

    fn fake_segment(vaddr: u64, memsz: u64, flags: u32) -> ProgramHeader {
        ProgramHeader {
            p_type: 1,
            flags,
            offset: 0,
            vaddr,
            paddr: vaddr,
            filesz: memsz,
            memsz,
            align: 0x1000,
        }
    }

    #[test]
    fn page_flags_for_segments_unions_shared_pages() {
        // The real hello layout (readelf): code R E at 0x400000
        // (0x2dff), rodata R at 0x402e00 (0x8a4), .got RW at
        // 0x4036a8 (0x1e0), data+bss RW at 0x403888 (0x280).
        let segments = [
            fake_segment(0x400000, 0x2dff, PF_R | PF_X),
            fake_segment(0x402e00, 0x8a4, PF_R),
            fake_segment(0x4036a8, 0x1e0, PF_R | PF_W),
            fake_segment(0x403888, 0x280, PF_R | PF_W),
        ];
        // Code-only page: executable, NOT writable (the page the
        // Batch 8 boot faulted copying into).
        let f = page_flags_for_segments(&segments, 0x400000);
        assert!(!f.contains(PageTableFlags::WRITABLE));
        assert!(!f.contains(PageTableFlags::NO_EXECUTE));
        // Page shared by code (R E) and rodata (R): still
        // executable, still not writable.
        let f = page_flags_for_segments(&segments, 0x402000);
        assert!(!f.contains(PageTableFlags::WRITABLE));
        assert!(!f.contains(PageTableFlags::NO_EXECUTE));
        // Page shared by rodata (R), .got (RW) and data (RW):
        // writable (union) and NOT executable (no covering
        // segment is PF_X).
        let f = page_flags_for_segments(&segments, 0x403000);
        assert!(f.contains(PageTableFlags::WRITABLE));
        assert!(f.contains(PageTableFlags::NO_EXECUTE));
        // A page no segment covers: readable only, never
        // writable, never executable.
        let f = page_flags_for_segments(&segments, 0x404000);
        assert!(!f.contains(PageTableFlags::WRITABLE));
        assert!(f.contains(PageTableFlags::NO_EXECUTE));
        // Every result stays user-accessible and present.
        for page in [0x400000, 0x402000, 0x403000, 0x404000] {
            let f = page_flags_for_segments(&segments, page);
            assert!(f.contains(PageTableFlags::PRESENT));
            assert!(f.contains(PageTableFlags::USER_ACCESSIBLE));
        }
    }

    #[test]
    fn user_layout_constants_are_sane() {
        assert_eq!(USER_CODE_BASE, 0x400000);
        assert_eq!(USER_STACK_TOP, 0x800000);
        assert_eq!(USER_HEAP_BASE, 0x100_0000);
        assert_eq!(USER_ADDRESS_SPACE_END, 0x101_0000);
        // Stack and heap do not overlap the image region or each other.
        assert!(USER_STACK_TOP - USER_STACK_SIZE as u64 >= USER_CODE_BASE);
        assert!(USER_HEAP_BASE >= USER_STACK_TOP);
        // The initial rsp is 8 mod 16 with room for the StartInfo
        // (16 bytes) below the stack top.
        let rsp = USER_STACK_TOP - START_INFO_OFFSET;
        assert_eq!(rsp % 16, 8);
        assert_eq!(START_INFO_OFFSET, 24);
        assert!(rsp + 16 <= USER_STACK_TOP);
    }

    #[test]
    fn shell_slot_layout_is_sane_and_does_not_overlap_hello() {
        // The shell slot (Batch 9): code where hello's stack top
        // ends, stack top a window above, heap above hello's heap.
        assert_eq!(SH_CODE_BASE, 0x800000);
        assert_eq!(SH_STACK_TOP, 0xC00000);
        assert_eq!(SH_HEAP_BASE, 0x110_0000);
        assert_eq!(SH_ADDRESS_SPACE_END, 0x111_0000);
        assert_eq!(SH_LAYOUT.code_base, SH_CODE_BASE);
        assert_eq!(SH_LAYOUT.image_end, SH_STACK_TOP);
        assert_eq!(SH_LAYOUT.stack_top, SH_STACK_TOP);
        assert_eq!(SH_LAYOUT.stack_size, USER_STACK_SIZE);
        assert_eq!(SH_LAYOUT.heap_base, SH_HEAP_BASE);
        assert_eq!(SH_LAYOUT.heap_size, USER_HEAP_SIZE);
        // The shell's initial rsp is 8 mod 16 with room for the
        // StartInfo below its stack top, like hello's.
        let rsp = SH_STACK_TOP - START_INFO_OFFSET;
        assert_eq!(rsp, 0xbfff_e8);
        assert_eq!(rsp % 16, 8);
        assert!(rsp + 16 <= SH_STACK_TOP);
        // The two slots share one page table: no interval of one
        // may overlap any interval of the other (image, stack, or
        // heap) — overlapping slots would map two programs' pages
        // onto each other.
        assert!(!layouts_overlap(&HELLO_LAYOUT, &SH_LAYOUT));
        assert!(!layouts_overlap(&SH_LAYOUT, &HELLO_LAYOUT));
        // And the check is not vacuous: a layout overlapping
        // itself, or one shifted onto hello's heap, does overlap.
        assert!(layouts_overlap(&HELLO_LAYOUT, &HELLO_LAYOUT));
        let mut shifted = SH_LAYOUT;
        shifted.heap_base = USER_HEAP_BASE;
        assert!(layouts_overlap(&HELLO_LAYOUT, &shifted));
    }

    #[test]
    fn is_user_buffer_checks_one_mapped_interval() {
        // Each test registers its own unique range (the registry is
        // a process-wide static shared by parallel tests, so ranges
        // must not overlap between tests).
        register_user_region(0x400000, 0x404000);
        // Inside, exactly at the edges, and spanning are decided
        // by containment in ONE interval.
        assert!(is_user_buffer(0x400000, 0x100));
        assert!(is_user_buffer(0x403f00, 0x100));
        assert!(is_user_buffer(0x404000 - 1, 1));
        assert!(!is_user_buffer(0x403f00, 0x200)); // crosses the end
        assert!(!is_user_buffer(0x3fffff, 0x100)); // starts before
        assert!(!is_user_buffer(0x500000, 0x100)); // nowhere near
        assert!(!is_user_buffer(0x400000, usize::MAX)); // overflow
        assert!(is_user_buffer(0x500000, 0)); // zero length: nothing dereferenced
        assert!(is_user_buffer(0, 0));
    }

    #[test]
    fn is_user_buffer_rejects_kernel_addresses() {
        // The kernel heap and image are never user regions: a
        // userspace program passing a kernel pointer gets -EFAULT.
        assert!(!is_user_buffer(0x444444440000, 0x100));
        assert!(!is_user_buffer(0x10000000000, 0x100));
    }
}
