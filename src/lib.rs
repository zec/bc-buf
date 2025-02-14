// Copyright (c) 2022-2024 The Pennsylvania State University and the project contributors.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Nonblocking single-writer, multiple-reader,
//! possibly-zero-allocation circular buffers
//! for sharing data across threads and processes
//! (with possible misses by readers).
//!
//! [`CBuf<T, SIZE>`] is the structure holding a circular buffer itself.
//! A single [`CBufWriter<T, SIZE>`] is used to add items to the (logical)
//! end of a buffer, and one or more [`CBufReader<T, SIZE>`]s are used to
//! (fallibly) read items from the buffer. [`CBuf`]s start out
//! conceptually empty; once `SIZE-1` items have been added, the
//! buffer is full, and the later addition of an item will (conceptually)
//! overwrite the oldest item in the buffer (the actual details are
//! slightly different but appear like this externally).
//!
//! A [`CBufIndex<SIZE>`] can be used to address items inside a buffer
//! (and one is used internally by a [`CBufReader`] to implement sequential
//! reads). A [`CBufIndex`] can become stale if enough additional items are
//! written to the buffer; staleness is detected with high probability
//! (~(2<sup>[`usize::BITS`]</sup>&nbsp;&minus;&nbsp;`SIZE`)&nbsp;/&nbsp;2<sup>[`usize::BITS`]</sup>
//! in the most general case, and higher in typical envisioned cases).

#![no_std]
#![warn(missing_docs)]

#[cfg(test)]
extern crate std;

pub(crate) mod iter;
pub(crate) mod utils;

use crate::utils::{AtomicIndex, Sequencer};
pub use crate::utils::{Bisect, CBufIndex};

use core::ops::Range;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, AtomicUsize, Ordering::SeqCst};

/// A convenience trait for the constraints on the possible types
/// that can be stored in [`CBuf`]s.
pub trait CBufItem: Sized + Copy + Send + 'static {}

/// Blanket implementation for all eligible types.
impl<T> CBufItem for T where T: Sized + Copy + Send + 'static {}

/// A circular buffer of `T`s which can store up to `SIZE-1` elements at a time.
///
/// The constant parameter `SIZE` must be a power of two that is greater than 1;
/// trying to use other values for `SIZE` will result in a compile-time panic.
pub struct CBuf<T: CBufItem, const SIZE: usize> {
    buf:  [T; SIZE],
    next: AtomicIndex<SIZE>,
}

/// A consumer of elements from a `&'a `[`CBuf<T, SIZE>`], or
/// a [`[T; SIZE]`](array) and [`AtomicUsize`] being used with
/// the same semantics as a [`CBuf<T, SIZE>`].
///
/// There may be many readers for a given [`CBuf`].
///
/// Items may be read from the circular buffer sequentially
/// using [`fetch_next_item`](Self::fetch_next_item)
/// (or the iterator returned by [`available_items_iter`](Self::available_items_iter),
/// which repeatedly calls [`fetch_next_item`](Self::fetch_next_item)),
/// or may be fetched by index using [`fetch`](Self::fetch).
///
/// Both sequential and indexed reads are fallible:
/// if too many elements have been written to the circular buffer by the [`CBufWriter`],
/// the [`CBufIndex<SIZE>`] used for [`fetch`](Self::fetch) (or used interally by
/// the [`CBufReader`] in [`fetch_next_item`](Self::fetch_next_item)) may become out-of-date, with
/// a newer element in place of the element referred to by the index.
#[derive(Clone)]
pub struct CBufReader<'a, T: CBufItem, const SIZE: usize> {
    /// Pointer to the [`CBuf`]'s backing array.
    buf:        *const T,
    /// Reference to the [`CBuf`]'s buffer state.
    next_ref:   &'a AtomicIndex<SIZE>,
    /// The index of the next index to read from, when reading sequentially.
    next_local: CBufIndex<SIZE>,
}

/// An inserter of elements into a `&'a mut `[`CBuf<T, SIZE>`], or
/// a [`[T; SIZE]`](array) and [`AtomicUsize`] being used with
/// the same semantics as a [`CBuf<T, SIZE>`].
///
/// There **must** be at most one writer for a given [`CBuf`] at any given time.
///
/// Items are inserted into the circular buffer using [`add_item`](Self::add_item).
pub struct CBufWriter<'a, T: CBufItem, const SIZE: usize> {
    /// Pointer to the [`CBuf`]'s backing array.
    buf:        *mut T,
    /// Reference to the [`CBuf`]'s buffer state.
    next_ref:   &'a AtomicIndex<SIZE>,
    /// The index of the next index to write to.
    next_local: CBufIndex<SIZE>,
}

impl<T: CBufItem, const SIZE: usize> CBuf<T, SIZE> {
    /// If `SIZE` is a power of two that's greater than 1 and less than
    /// 2<sup>[`usize::BITS`]</sup>, as it should be, this is just `true`;
    /// otherwise, this generates a compile-time panic.
    const IS_SIZE_OK: bool = if SIZE.is_power_of_two() && SIZE > 1 {
        true
    } else {
        panic!("SIZE param of some CBuf is *not* an allowed value")
    };

    /// Constant AND'ed with the value of [`Self::next`] to obtain
    /// the index in [`Self::buf`] at which the next item should be written.
    const IDX_MASK: usize = if Self::IS_SIZE_OK { SIZE - 1 } else { 0 };

    /// Constructs a new, logically-empty [`CBuf<T, SIZE>`].
    ///
    /// `filler_item` is only used to initialize the buffer's storage; it will
    /// not be returned by [`CBufReader<T, SIZE>`] or [`CBufWriter<T, SIZE>`] accessor methods.
    ///
    /// # Panics
    ///
    /// If `SIZE` is not a valid value (a power of 2 that is greater than 1),
    /// a compile-time panic will be generated.
    #[inline]
    pub const fn new(filler_item: T) -> Self {
        let initial_next: usize = if Self::IS_SIZE_OK { 0 } else { 1 };

        CBuf {
            buf:  [filler_item; SIZE],
            next: AtomicIndex::new(initial_next),
        }
    }

    /// Given a pointer `p` to a possibly-uninitialized [`CBuf<T, SIZE>`],
    /// initializes it in a logically-empty state, using
    /// `filler_item` to ensure the buffer memory has been initialized.
    ///
    /// If `p` is null, nothing is done.
    ///
    /// # Safety
    ///
    /// `p` must be a properly-aligned pointer; undefined behavior occurs
    /// otherwise.
    ///
    /// # Panics
    ///
    /// If `SIZE` is not a valid value (a power of 2 that is greater than 1),
    /// a compile-time panic will be generated.
    #[inline]
    pub unsafe fn initialize(p: *mut Self, filler_item: T) {
        use core::ptr::addr_of_mut;

        if p.is_null() || !Self::IS_SIZE_OK {
            return;
        }

        for i in 0..SIZE {
            write_volatile(addr_of_mut!((*p).buf[i]), filler_item);
        }
        write_volatile(addr_of_mut!((*p).next), AtomicIndex::new(0));
        fence(SeqCst);
    }

    /// Returns a [`CBufWriter`] for adding elements to `self`.
    #[inline]
    pub fn as_writer<'a>(&'a mut self) -> CBufWriter<'a, T, SIZE> {
        CBufWriter {
            buf:        self.buf.as_mut_ptr(),
            next_ref:   &self.next,
            next_local: self.next.load(SeqCst),
        }
    }

    /// Returns a [`CBufReader`] for reading elements from `self`.
    ///
    /// `next_index` for reading is initialized to end of available data.
    #[inline]
    pub fn as_reader<'a>(&'a self) -> CBufReader<'a, T, SIZE> {
        CBufReader {
            buf:        self.buf.as_ptr(),
            next_ref:   &self.next,
            next_local: self.next.load(SeqCst),
        }
    }

    /// Returns a pointer to the start of the backing array (of length `SIZE`)
    /// and a pointer to the circular-buffer state (stored in an [`AtomicUsize`]).
    ///
    /// Assuming `T` has a [layout] that is stable across compilations
    /// (e.g., `T` is `#[repr(C)]` and only has members with stable layouts),
    /// the return values can be transferred to separately-compiled code, which
    /// can pass the pointers to [`CBufReader::new_from_ptr`] and construct a [`CBufReader<T, SIZE>`].
    ///
    /// [layout]: https://doc.rust-lang.org/reference/type-layout.html
    #[inline]
    pub const fn as_ptrs(&self) -> (*const T, *const AtomicUsize) {
        (self.buf.as_ptr(), &self.next.n)
    }

    /// Returns a mutable pointer to the start of the backing array (of length `SIZE`)
    /// and a pointer to the circular-buffer state (stored in an [`AtomicUsize`]).
    ///
    /// Assuming `T` has a [layout] that is stable across compilations
    /// (e.g., `T` is `#[repr(C)]` and only has members with stable layouts),
    /// the return values can be transferred to separately-compiled code, which
    /// can pass the pointers to [`CBufWriter::new_from_ptr`] and construct a [`CBufWriter<T, SIZE>`].
    ///
    /// [layout]: https://doc.rust-lang.org/reference/type-layout.html
    #[inline]
    pub fn as_mut_ptrs(&mut self) -> (*mut T, *const AtomicUsize) {
        (self.buf.as_mut_ptr(), &self.next.n)
    }
}

impl<T: CBufItem + Default, const SIZE: usize> CBuf<T, SIZE> {
    /// A shortcut for [`CBuf::new`]`(`[`T::default`](Default::default)`())`.
    #[inline]
    pub fn new_with_default() -> Self {
        Self::new(T::default())
    }

    /// A shortcut for [`CBuf::initialize`]`(`[`T::default`](Default::default)`())`.
    #[inline]
    pub unsafe fn initialize_with_default(p: *mut Self) {
        Self::initialize(p, T::default());
    }
}

unsafe impl<'a, T: CBufItem, const SIZE: usize> Send for CBufWriter<'a, T, SIZE> {}

impl<'a, T: CBufItem, const SIZE: usize> CBufWriter<'a, T, SIZE> {
    const IS_SIZE_OK: bool = CBuf::<T, SIZE>::IS_SIZE_OK;
    const IDX_MASK: usize = CBuf::<T, SIZE>::IDX_MASK;

    /// Creates a new [`CBufWriter<T, SIZE>`]
    /// from `buf`, a pointer to the start of the `SIZE`-element array storing a [`CBuf<T, SIZE>`]'s items,
    /// and `next_idx`, a pointer to the [`CBuf`]'s buffer state,
    /// assuming they are non-null.
    ///
    /// Returns `None` if `buf` or `next_idx` is null.
    ///
    /// # Safety
    ///
    /// `buf` and `next_idx` must be properly aligned.
    ///
    /// The programmer must ensure that the pointees of `buf` and `next_idx` outlive
    /// the returned [`CBufWriter<T, SIZE>`].
    ///
    /// At any given time, there **must** be at most one [`CBufWriter`] for a given circular buffer.
    /// This is not enforced by this function&mdash;the programmer must ensure this is the case.
    ///
    /// # Panics
    ///
    /// If `SIZE` is not a valid value (a power of 2 that is greater than 1),
    /// a compile-time panic will be generated.
    #[inline]
    pub unsafe fn new_from_ptr(buf: *mut T, next_idx: *const AtomicUsize) -> Option<Self> {
        if !Self::IS_SIZE_OK || buf.is_null() || next_idx.is_null()
        /*|| !buf.is_aligned() || !next_idx.is_aligned()*/
        {
            return None;
        }

        let next_ref: &AtomicIndex<SIZE> = &*(next_idx as *const AtomicIndex<SIZE>);
        Some(Self {
            buf,
            next_ref,
            next_local: next_ref.load(SeqCst),
        })
    }

    /// Inserts `item` as the newest item in the associated circular buffer.
    ///
    /// If the buffer is full (contains `SIZE-1` items), the oldest item
    /// is implicitly removed from the buffer.
    #[inline(always)]
    pub fn add_item(&mut self, item: T) {
        self.add_item_seq(item, &());
    }

    /// The backing logic for [`add_item`](Self::add_item).
    ///
    /// Tests in this crate can use non-[`()`] [`Sequencer`] implementors
    /// to control concurrency between this [`CBufWriter`] and [`CBufReader`]s.
    #[inline]
    pub(crate) fn add_item_seq<S: Sequencer<WriteProtocolStep>>(&mut self, item: T, seq: &S) {
        macro_rules! wrap_atomic {
            ($label:ident, $($x:tt)*) => {
                seq.wait_for_go_ahead();
                { $($x)* }
                seq.send_result(WriteProtocolStep::$label);
            };
        }

        let next_next = self.next_local + 1;

        wrap_atomic! { PreWrite, }
        wrap_atomic! { BufferWrite,
            fence(SeqCst);
            unsafe { write_volatile(self.buf.add(self.next_local.as_usize() & Self::IDX_MASK), item) };
            fence(SeqCst);
        }
        wrap_atomic! { IndexPostUpdate, self.next_ref.store(next_next, SeqCst); }

        self.next_local = next_next;
    }

    /// Returns an iterator into the circular buffer's currently-stored items.
    #[inline]
    pub fn current_items_iter<'b>(&'b mut self) -> impl Iterator<Item = T> + 'b + Captures<'a>
    where
        'a: 'b,
    {
        iter::WriterIterator {
            writer_next: self.next_local,
            buf:         unsafe { &*(self.buf as *const [T; SIZE]) },
            idx:         self.next_local - (SIZE - 1),
            _writer:     self,
        }
    }

    /// Returns a [`CBufReader<T, SIZE>`] for this circular buffer.
    #[inline]
    pub fn as_reader<'b>(&'b mut self) -> CBufReader<'b, T, SIZE> {
        CBufReader {
            buf:        self.buf,
            next_ref:   self.next_ref,
            next_local: self.next_local - (SIZE - 1),
        }
    }
}

unsafe impl<'a, T: CBufItem, const SIZE: usize> Send for CBufReader<'a, T, SIZE> {}

impl<'a, T: CBufItem, const SIZE: usize> CBufReader<'a, T, SIZE> {
    const IS_SIZE_OK: bool = CBuf::<T, SIZE>::IS_SIZE_OK;
    const IDX_MASK: usize = CBuf::<T, SIZE>::IDX_MASK;

    /// The maximum number of times [`fetch_next_item`](Self::fetch_next_item)
    /// will try to fetch an item from the circular buffer
    /// before giving up and returning [`ReadResult::SpinFail`].
    pub const NUM_READ_TRIES: u32 = 16;

    /// Creates a new [`CBufReader<T, SIZE>`]
    /// from `buf`, a pointer to the start of the `SIZE`-element array storing a [`CBuf<T, SIZE>`]'s items,
    /// and `next_idx`, a pointer to the [`CBuf`]'s buffer state,
    /// assuming they are non-null.
    ///
    /// Returns `None` if `buf` or `next_idx` is null.
    ///
    /// # Safety
    ///
    /// `buf` and `next_idx` must be properly aligned.
    ///
    /// The programmer must ensure that the pointees of `buf` and `next_idx` outlive
    /// the returned [`CBufReader<T, SIZE>`].
    ///
    /// Unlike with [`CBufWriter`], there may be multiple [`CBufReader`]s for a given circular buffer.
    ///
    /// # Panics
    ///
    /// If `SIZE` is not a valid value (a power of 2 that is greater than 1),
    /// a compile-time panic will be generated.
    #[inline]
    pub unsafe fn new_from_ptr(buf: *const T, next_idx: *const AtomicUsize) -> Option<Self> {
        if !Self::IS_SIZE_OK || buf.is_null() || next_idx.is_null()
        /*|| !buf.is_aligned() || !next_idx.is_aligned()*/
        {
            return None;
        }

        let next_ref: &AtomicIndex<SIZE> = &*(next_idx as *const AtomicIndex<SIZE>);
        Some(Self {
            buf,
            next_ref,
            next_local: next_ref.load(SeqCst),
        })
    }

    /// Tries to fetch the item in the buffer immediately after the one fetched by
    /// the previous call to [`fetch_next_item`](Self::fetch_next_item). If successful,
    /// [`ReadResult::Success`] is returned, with the item in question as its payload.
    ///
    /// There are a few possible ways trying to fetch the next item can
    /// fail:
    ///
    /// * There may not be a next item, if the previously-fetched item was
    ///   the last one that's been added to the buffer. In this case,
    ///   [`ReadResult::None`] is returned.
    /// * Enough items have been added since the previously-fetched item that
    ///   the intended next item has already been overwritten. In this case,
    ///   [`ReadResult::Skipped`] is returned with a payload of either the
    ///   oldest valid item in the buffer (if `fast_forward` is `false`) or
    ///   the newest item in the buffer (if `fast_forward` is `true`).
    /// * If the [`CBufWriter`] is adding new items to the buffer while `fetch_next_item`
    ///   is trying to read, it's possible the read attempt will get clobbered.
    ///   If [`NUM_READ_TRIES`](Self::NUM_READ_TRIES) attempts to read occur and they all get clobbered
    ///   by the writer, [`ReadResult::SpinFail`] is returned.
    #[inline(always)]
    pub fn fetch_next_item(&mut self, fast_forward: bool) -> ReadResult<T> {
        self.fetch_next_item_seq(fast_forward, &())
    }

    /// The backing logic for [`fetch_next_item`](Self::fetch_next_item).
    ///
    /// Tests in this crate can use non-[`()`] [`Sequencer`] implementors
    /// to control concurrency between this [`CBufReader`]
    /// and the [`CBufWriter`] and any other [`CBufReader`]s.
    #[inline]
    pub(crate) fn fetch_next_item_seq<S>(&mut self, fast_forward: bool, seq: &S) -> ReadResult<T>
    where
        S: Sequencer<FetchCheckpoint<ReadResult<T>>>,
    {
        use ReadResult as RR;

        macro_rules! wrap_atomic {
            ($label:ident, $($x:tt)*) => {
                {
                    seq.wait_for_go_ahead();
                    let x = { $($x)* };
                    seq.send_result(FetchCheckpoint::Step(ReadProtocolStep::$label));
                    x
                }
            };
        }
        macro_rules! send_and_return {
            ($result:expr) => {
                seq.wait_for_go_ahead();
                let x = $result;
                seq.send_result(FetchCheckpoint::ReturnVal(x));
                return x;
            };
        }

        let mut skipped = false;

        for _ in 0..Self::NUM_READ_TRIES {
            let next_0 = wrap_atomic! { IndexCheckPre, self.next_ref.load(SeqCst) };
            let item = wrap_atomic! { BufferRead,
                fence(SeqCst);
                let item = unsafe { read_volatile(self.buf.add(self.next_local.as_usize() & Self::IDX_MASK)) };
                fence(SeqCst);
                item
            };
            let next_1 = wrap_atomic! { IndexCheckPost, self.next_ref.load(SeqCst) };
            #[cfg(test)]
            std::eprintln!(
                "next: 0x{:02x}    0: 0x{:02x}    1: 0x{:02x}",
                self.next_local.as_usize(),
                next_0.as_usize(),
                next_1.as_usize()
            );

            let next_next = self.next_local + 1;

            if next_1 == CBufIndex::ZERO {
                send_and_return!(RR::None);
            }
            if (self.next_local == next_0) && (self.next_local == next_1) {
                send_and_return!(RR::None);
            }

            // if the CBuf's `next`, while we read `item` from the array,
            // stayed with the range such that `item` was not written to _and_
            // had not been previously overwritten by a newer value:
            if next_0.is_in_range(next_next, SIZE - 1) && next_1.is_in_range(next_next, SIZE - 1) {
                self.next_local = next_next;
                send_and_return!(if !skipped { RR::Success(item) } else { RR::Skipped(item) });
            }

            if !(next_1.is_in_range(next_next, SIZE - 1)) {
                // we've gotten too far behind, and we've gotten wrapped by the writer,
                // or else the circular buffer's been re-initialized;
                // in either case, we need to play catch-up
                let offset = if fast_forward { 1 } else { SIZE - 1 };
                self.next_local = next_1 - offset;
                skipped = true;
            }

            if next_0 != next_1 {
                // writer is active, let's give a hint to cool this thread's jets:
                core::hint::spin_loop();
            }
        }

        send_and_return!(RR::SpinFail);
    }

    /// Returns an iterator that calls [`fetch_next_item`](Self::fetch_next_item) repeatedly
    /// until we reach the last-written item in the buffer.
    ///
    /// `fast_forward` is used as the `fast_forward` argument to calls to
    /// [`fetch_next_item`](Self::fetch_next_item).
    #[inline]
    pub fn available_items_iter<'b>(
        &'b mut self,
        fast_forward: bool,
    ) -> impl Iterator<Item = ReadResult<T>> + 'b + Captures<'a>
    where
        'a: 'b,
    {
        iter::ReaderIterator {
            reader: self,
            fast_forward,
            is_done: false,
        }
    }

    /// Resets the [`CBufReader`]'s internal current index
    /// (as used by [`fetch_next_item`](Self::fetch_next_item)
    /// and returned by [`current_next_index`](Self::current_next_index))
    /// to just past the last-written item in the buffer.
    #[inline]
    pub fn fast_forward(&mut self) {
        self.next_local = self.next_ref.load(SeqCst);
    }

    /// Resets the [`CBufReader`]'s internal current index
    /// (as used by [`fetch_next_item`](Self::fetch_next_item)
    /// and returned by [`current_next_index`](Self::current_next_index))
    /// to the earliest available item in the buffer.
    #[inline]
    pub fn rewind(&mut self) {
        self.next_local = self.next_ref.load(SeqCst) - (SIZE - 1);
    }

    /// Returns the current value of the index used for sequential reads
    /// (i.e., [`fetch_next_item`](Self::fetch_next_item)
    /// and [`available_items_iter`](Self::available_items_iter)).
    #[inline]
    pub fn current_next_index(&self) -> CBufIndex<SIZE> {
        self.next_local
    }

    /// Returns a new `CBufReader` with the sequential read index set to `next`.
    ///
    /// (c.f. [`fetch_next_item`](Self::fetch_next_item)
    /// and [`available_items_iter`](Self::available_items_iter)).
    #[inline]
    pub fn reader_starting_at(&self, next: CBufIndex<SIZE>) -> Self {
        let mut result = self.clone();
        result.next_local = next;
        result
    }

    /// Returns a new `CBufReader` with the sequential read index starting at
    /// the first item that makes the predicate true.
    ///
    /// Requires that the buffer contents are sorted by
    /// `predicate()` in `false`, `true` order.
    ///
    /// If `fromstart` is set, searches all available data, otherwise
    /// the search starts at the `current_next_index()`.
    ///
    /// Returns `InvalidIndexError` if no such index exists.
    ///
    /// If data is not sorted by `predicate()` (i.e. if at least one
    /// item that makes the predicate `false` is later than at
    /// least one item that makes the predicate `true`) the
    /// function may return either an index that makes the predicate
    /// true, or an `InvalidIndexError`.
    pub fn search(
        &self,
        predicate: &dyn Fn(T) -> bool,
        fromstart: bool,
    ) -> Result<Self, InvalidIndexError> {
        let valids = self.current_valid_index_range();
        let mut ifalse = if fromstart { valids.start } else { self.next_local };
        match self.fetch_with_index(ifalse) {
            FetchWithIndexResult::Success(v, idx) | FetchWithIndexResult::Skipped(v, idx) => {
                if predicate(v) {
                    return Ok(self.reader_starting_at(idx));
                }
                ifalse = idx
            }
            FetchWithIndexResult::None | FetchWithIndexResult::SpinFail => {
                return Err(InvalidIndexError(()))
            }
        };
        let mut itrue = valids.end - 1;
        // Verify that the predicate is true at the end
        match self.fetch(itrue) {
            Ok(v) => {
                if !predicate(v) {
                    return Err(InvalidIndexError(()));
                }
            }
            Err(_) => return Err(InvalidIndexError(())),
        }
        // At this point, ifalse < itrue and index to false and true values

        while ifalse + 1 < itrue {
            let imid;
            if let Some(didx) = itrue - ifalse {
                imid = ifalse + didx / 2
            } else {
                return Err(InvalidIndexError(()));
            }

            match self.fetch_with_index(imid) {
                FetchWithIndexResult::Success(v, idx) | FetchWithIndexResult::Skipped(v, idx) => {
                    if predicate(v) {
                        itrue = idx;
                    } else {
                        ifalse = idx;
                    }
                }
                FetchWithIndexResult::None | FetchWithIndexResult::SpinFail => {
                    // Not strictly accurate:  This is when the writer overtakes
                    // the data being searched.
                    return Err(InvalidIndexError(()));
                }
            }
        }
        return Ok(self.reader_starting_at(itrue));
    }

    /// Returns the half-open range of valid indexes for items currently in the circular buffer.
    #[inline]
    pub fn current_valid_index_range(&self) -> Range<CBufIndex<SIZE>> {
        let next = self.next_ref.load(SeqCst);
        Range {
            start: next - (SIZE - 1),
            end:   next,
        }
    }

    /// Tries to read the item at index `idx` in the buffer.
    ///
    /// If successful, returns the item;
    /// if not (see [`fetch_next_item`](Self::fetch_next_item) for possible failure modes),
    /// returns an [`InvalidIndexError`].
    #[inline(always)]
    pub fn fetch(&self, idx: CBufIndex<SIZE>) -> Result<T, InvalidIndexError> {
        self.fetch_seq(idx, &())
    }

    /// Tries to read the item at index `idx` in the buffer and return both a value and the index actually read.
    ///
    /// If successful, or if it had to skip forward to the earliest item, returns the item and its index.
    /// Can fail due to index beyond end (`None`) or spinlock.
    #[inline]
    pub fn fetch_with_index(&self, idx: CBufIndex<SIZE>) -> FetchWithIndexResult<T, SIZE> {
        let mut idx_try = idx;
        for _ in 0..Self::NUM_READ_TRIES {
            match self.fetch(idx_try) {
                Ok(t) => {
                    return if idx_try == idx {
                        FetchWithIndexResult::Success(t, idx_try)
                    } else {
                        FetchWithIndexResult::Skipped(t, idx_try)
                    }
                }
                Err(InvalidIndexError(())) => {
                    let valids = self.current_valid_index_range();
                    if idx_try >= valids.end {
                        return FetchWithIndexResult::None;
                    }
                    idx_try = valids.start
                }
            }
        }
        FetchWithIndexResult::SpinFail
    }

    /// The backing logic for [`fetch`](Self::fetch).
    ///
    /// Tests in this crate can use non-[`()`] [`Sequencer`] implementors
    /// to control concurrency.
    #[inline]
    pub(crate) fn fetch_seq<S>(&self, idx: CBufIndex<SIZE>, seq: &S) -> Result<T, InvalidIndexError>
    where
        S: Sequencer<FetchCheckpoint<Result<T, InvalidIndexError>>>,
    {
        macro_rules! wrap_atomic {
            ($label:ident, $($x:tt)*) => {
                {
                    seq.wait_for_go_ahead();
                    let x = { $($x)* };
                    seq.send_result(FetchCheckpoint::Step(ReadProtocolStep::$label));
                    x
                }
            };
        }

        let next_0 = wrap_atomic! { IndexCheckPre, self.next_ref.load(SeqCst) };
        let item = wrap_atomic! { BufferRead,
            fence(SeqCst);
            let item = unsafe { read_volatile(self.buf.add(idx.as_usize() & Self::IDX_MASK)) };
            fence(SeqCst);
            item
        };
        let next_1 = wrap_atomic! { IndexCheckPost, self.next_ref.load(SeqCst) };

        #[cfg(test)]
        std::eprintln!(
            "idx: {:02x}    next_0: {:02x}    next_1: {:02x}",
            idx.as_usize(),
            next_0.as_usize(),
            next_1.as_usize(),
        );

        let retval =
            if next_0.is_in_range(idx + 1, SIZE - 1) && next_1.is_in_range(idx + 1, SIZE - 1) {
                Ok(item)
            } else {
                Err(InvalidIndexError(()))
            };

        seq.wait_for_go_ahead();
        seq.send_result(FetchCheckpoint::ReturnVal(retval));
        retval
    }
}

/// The return value of [`CBufReader::fetch_next_item`]
/// and of the iterator returned by [`CBufReader::available_items_iter`].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ReadResult<T> {
    /// Normal case; fetched next item.
    Success(T),
    /// Fetched an item, but we missed some number.
    Skipped(T),
    /// No new items are available.
    None,
    /// After trying [`CBufReader::NUM_READ_TRIES`] times, not able to successfully read an item.
    SpinFail,
}

/// The return value of [`CBufReader::fetch_with_index`]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FetchWithIndexResult<T, const SIZE: usize> {
    /// Normal case; fetched next item.
    Success(T, CBufIndex<SIZE>),
    /// Fetched an item, but we missed some number.
    Skipped(T, CBufIndex<SIZE>),
    /// No new items are available.
    None,
    /// After trying [`CBufReader::NUM_READ_TRIES`] times, not able to successfully read an item.
    SpinFail,
}

/// Signifies that the index used in [`CBufReader::fetch`] is not valid.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct InvalidIndexError(());

/// One of the atomic steps in the protocol used by
/// [`CBufWriter::add_item_seq`] to add an item to a buffer.
///
/// Also, one of the checkpoints a [`Sequencer`] may stop at within
/// [`CBufWriter::add_item_seq`].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum WriteProtocolStep {
    PreWrite,
    BufferWrite,
    IndexPostUpdate,
}

/// One of the atomic steps in the protocol used by
/// [`CBufReader::fetch_next_item_seq`] to fetch an item from a buffer.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum ReadProtocolStep {
    IndexCheckPre,
    BufferRead,
    IndexCheckPost,
}

/// One of the checkpoints a [`Sequencer`] may stop at within
/// [`CBufReader::fetch_next_item_seq`].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum FetchCheckpoint<T> {
    Step(ReadProtocolStep),
    ReturnVal(T),
}

/// Dummy trait used to satiate the Rust compiler
/// in certain uses of lifetimes inside the concrete type
/// behind a function/method's `impl Trait` return type.
///
/// Idea taken straight from [a comment in rust-users].
///
/// [a comment in rust-users]: https://users.rust-lang.org/t/lifetimes-in-smol-executor/59157/8?u=yandros
pub trait Captures<'_a> {}
impl<T: ?Sized> Captures<'_> for T {}

#[cfg(test)]
mod tests;
