//! Paging and heap setup.
//!
//! * [`init`] — build an `OffsetPageTable` from the bootloader's
//!   physical-memory mapping, so the kernel can translate virtual
//!   addresses and map new pages.
//! * [`BootInfoFrameAllocator`] — hands out 4 KiB physical frames from
//!   the `Usable` regions of the bootloader memory map, and nothing else.
//! * [`init_heap`] — map 100 KiB for the kernel heap at a fixed virtual
//!   address and initialise the global [`ALLOCATOR`].
//!
//! The heap is what makes `Box`, `Vec`, and `Rc` available in the kernel
//! (`alloc` crate). Everything in this module is `unsafe` where it touches
//! page tables or raw frames, and the call sites in `main.rs` are the
//! only places that unsafety is discharged.

use bootloader_api::info::{MemoryRegion, MemoryRegionKind};
use bootloader_api::BootInfo;
use linked_list_allocator::LockedHeap;
use x86_64::structures::paging::mapper::MapToError;
use x86_64::structures::paging::{
    FrameAllocator, Mapper, OffsetPageTable, Page, PageTable, PageTableFlags, PhysFrame, Size4KiB,
};
use x86_64::{PhysAddr, VirtAddr};

/// Global heap allocator. Empty until [`init_heap`] runs.
///
/// Only registered as *the* global allocator on the kernel target.
/// Under `#[cfg(test)]` the test harness needs the normal std allocator
/// to run at all, so the attribute is gated off there — and the heap
/// test below exercises a local `Heap` instead of this static.
#[cfg_attr(not(test), global_allocator)]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Virtual address of the start of the kernel heap.
///
/// Deliberately distinctive (`0x4444…`): heap addresses are
/// instantly recognizable in boot logs. The worker task's stack
/// prints as `0x4444444402c0-0x4444444442c0`, which looks like a
/// fill/uninitialized value but is a real heap address (heap start
/// + 0x2c0 — the first allocations of the boot land just past the
/// base). Do not "fix" the address: it is the heap base, not
/// uninitialized memory.
pub const HEAP_START: usize = 0x_4444_4444_0000;
/// Size of the kernel heap, in bytes (100 KiB).
pub const HEAP_SIZE: usize = 100 * 1024;

/// Create an `OffsetPageTable` for the kernel's own page tables.
///
/// # Safety
///
/// The caller must guarantee that `physical_memory_offset` is the offset
/// the bootloader mapped all of physical memory at (requested in the
/// kernel's `BootloaderConfig`), and that the active level-4 table is
/// the one the bootloader set up.
pub unsafe fn init(physical_memory_offset: VirtAddr) -> OffsetPageTable<'static> {
    let level_4_table = unsafe { active_level_4_table(physical_memory_offset) };
    OffsetPageTable::new(level_4_table, physical_memory_offset)
}

/// The active level-4 page table, accessed through the physical-memory
/// mapping.
unsafe fn active_level_4_table(
    physical_memory_offset: VirtAddr,
) -> &'static mut PageTable {
    use x86_64::registers::control::Cr3;

    let (level_4_table_frame, _) = Cr3::read();
    let phys = level_4_table_frame.start_address();
    let virt = physical_memory_offset + phys.as_u64();
    let page_table_ptr: *mut PageTable = virt.as_mut_ptr();
    unsafe { &mut *page_table_ptr }
}

/// Map the heap pages and initialise the global allocator.
///
/// # Errors
///
/// Returns the mapping error if any heap page cannot be mapped (e.g.
/// the frame allocator ran out of usable frames).
pub fn init_heap(
    mapper: &mut OffsetPageTable,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> Result<(), MapToError<Size4KiB>> {
    let page_range = {
        let heap_start = VirtAddr::new(HEAP_START as u64);
        let heap_end = heap_start + HEAP_SIZE as u64;
        let heap_start_page = Page::containing_address(heap_start);
        let heap_end_page = Page::containing_address(heap_end - 1u64);
        Page::range_inclusive(heap_start_page, heap_end_page)
    };

    for page in page_range {
        let frame = frame_allocator
            .allocate_frame()
            .ok_or(MapToError::FrameAllocationFailed)?;
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        unsafe {
            mapper.map_to(page, frame, flags, frame_allocator)?.flush();
        }
    }

    unsafe {
        ALLOCATOR.lock().init(HEAP_START as *mut u8, HEAP_SIZE);
    }
    Ok(())
}

/// Map one 4 KiB MMIO page identity-mapped (virtual address ==
/// physical address) with PRESENT | WRITABLE | NO_CACHE, so the kernel
/// can touch a device's registers (e.g. the LAPIC at `0xFEE00000` and
/// the IOAPIC at `0xFEC00000`). The frame is the device's own address,
/// not one from the allocator — the allocator is only used for the
/// intermediate page-table frames `map_to` may need.
///
/// # Safety
///
/// The caller must guarantee `phys_addr` really is a device register
/// page (mapping RAM as uncacheable MMIO, or mapping the wrong device,
/// corrupts the machine's view of memory).
pub fn map_mmio_page(
    mapper: &mut OffsetPageTable,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
    phys_addr: u64,
) -> Result<(), MapToError<Size4KiB>> {
    let page = Page::containing_address(VirtAddr::new(phys_addr));
    let frame = PhysFrame::containing_address(PhysAddr::new(phys_addr));
    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_CACHE;
    unsafe {
        mapper.map_to(page, frame, flags, frame_allocator)?.flush();
    }
    Ok(())
}

/// A frame allocator backed by the bootloader's memory map.
///
/// Yields each 4 KiB frame of every `Usable` region exactly once, in
/// address order. Frames belonging to the bootloader, the kernel image,
/// or unknown firmware regions are never handed out.
pub struct BootInfoFrameAllocator {
    memory_regions: &'static bootloader_api::info::MemoryRegions,
    next: usize,
}

impl BootInfoFrameAllocator {
    /// Create an allocator over the memory regions from `BootInfo`.
    ///
    /// # Safety
    ///
    /// The caller must pass the regions the bootloader actually reported
    /// for this boot; using a stale or fabricated map could hand out
    /// frames that are in use.
    pub unsafe fn init(memory_regions: &'static bootloader_api::info::MemoryRegions) -> Self {
        BootInfoFrameAllocator {
            memory_regions,
            next: 0,
        }
    }

    /// Iterator over all usable 4 KiB frames, in address order.
    fn usable_frames(&self) -> impl Iterator<Item = PhysFrame> {
        // The bootloader's memory map: only `Usable` regions are ours.
        let regions = self.memory_regions.iter();
        let usable_regions = regions.filter(|r| r.kind == MemoryRegionKind::Usable);
        // Each region yields its 4 KiB frames, low address first.
        let addr_ranges = usable_regions.map(|r| (r.start..r.end).step_by(4096));
        let frame_starts = addr_ranges.flatten();
        frame_starts.map(|addr| PhysFrame::containing_address(PhysAddr::new(addr)))
    }
}

unsafe impl FrameAllocator<Size4KiB> for BootInfoFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        let frame = self.usable_frames().nth(self.next);
        self.next += 1;
        frame
    }
}

/// Re-exported so `main.rs` can name the type without reaching into
/// `bootloader_api` itself.
pub type KernelBootInfo = BootInfo;
/// Re-exported memory-region type, for documentation and tests.
pub type KernelMemoryRegion = MemoryRegion;

#[cfg(test)]
mod tests {
    use core::alloc::Layout;
    use linked_list_allocator::Heap;

    #[test]
    fn heap_alloc_works_after_init() {
        // A scratch heap on the test stack: the same linked-list heap
        // the kernel uses, just pointed at test memory instead of
        // mapped pages — and a *local* Heap, not the global allocator,
        // because the test harness itself needs the std allocator.
        static mut HEAP_BUF: [u8; 4096] = [0; 4096];
        let heap_ptr = &raw mut HEAP_BUF as *mut u8;
        let mut heap = Heap::empty();
        unsafe {
            heap.init(heap_ptr, 4096);
        }
        assert!(heap.free() > 0);

        let layout = Layout::from_size_align(8, 8).unwrap();
        let ptr = heap.allocate_first_fit(layout).unwrap();
        unsafe {
            ptr.as_ptr().write_bytes(0xab, 8);
            assert_eq!(*ptr.as_ptr(), 0xab);
        }
        assert!(heap.used() >= 8);

        unsafe {
            heap.deallocate(ptr, layout);
        }
        // Freed memory is reusable: allocate again and get a block back.
        // (`allocate_first_fit` returns `Result`, so unwrapping already
        // proves the pointer is non-null.)
        let again = heap.allocate_first_fit(layout).unwrap();
        unsafe {
            heap.deallocate(again, layout);
        }
    }
}
