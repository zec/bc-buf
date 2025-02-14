// Copyright (c) 2022-2023 The Pennsylvania State University and the project contributors.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the reader and writer protocols.

#![cfg(test)]

use super::ReadResult as RR;
use super::*;
use crate::utils::TestSequencer;
use core::mem::drop;
use core::sync::atomic::Ordering::SeqCst;
use std::panic;
use std::sync::mpsc;
use std::thread::{spawn, JoinHandle};
use std::vec::Vec;

#[test]
fn can_construct_cbuf() {
    let a: CBuf<u32, 256> = CBuf::new(42);

    assert_eq!(a.next.load(SeqCst).as_usize(), 0usize);
    assert_eq!(a.buf.len(), 256);
    assert!(a.buf.iter().all(|x| *x == 42u32));

    let _b: CBuf<(u64, i32), 64> = CBuf::new((5u64, -3));

    let _c: CBuf<(), 16> = CBuf::new(());

    let _d: CBuf<(), 4096> = CBuf::new_with_default();

    let e: CBuf<isize, 128> = CBuf::new_with_default();

    assert_eq!(e.next.load(SeqCst).as_usize(), 0usize);
    assert_eq!(e.buf.len(), 128);
    assert!(e.buf.iter().all(|x| *x == <isize as Default>::default()));
}

#[test]
fn can_initialize_cbuf() {
    use core::mem::MaybeUninit;

    let mut un_cb: MaybeUninit<CBuf<u16, 128>> = MaybeUninit::uninit();
    let cb = unsafe {
        CBuf::initialize(un_cb.as_mut_ptr(), 1729);
        un_cb.assume_init()
    };

    assert_eq!(cb.next.load(SeqCst).as_usize(), 0usize);
    assert!(cb.buf.iter().all(|x| *x == 1729));
}

#[test]
fn add_an_item() {
    fn run_cbuf_add_item(
        initial_buffer: [i16; 4],
        initial_next: usize,
        item: i16,
        final_buffer: [i16; 4],
        final_next: usize,
    ) {
        let mut buf: CBuf<i16, 4> = CBuf {
            buf:  initial_buffer,
            next: AtomicIndex::new(initial_next),
        };
        let mut writer = buf.as_writer();

        writer.add_item(item);
        assert_eq!(buf.buf, final_buffer);
        assert_eq!(buf.next.load(SeqCst).as_usize(), final_next);
    }

    run_cbuf_add_item([-1, -2, -3, -4], 0, 99, [99, -2, -3, -4], 1);

    run_cbuf_add_item([99, -2, -3, -4], 1, 101, [99, 101, -3, -4], 2);

    run_cbuf_add_item([-1, -2, -3, -4], 3, 99, [-1, -2, -3, 99], 4);

    run_cbuf_add_item([-1, -2, -3, 99], 4, 101, [101, -2, -3, 99], 5);

    run_cbuf_add_item([-1, -2, -3, -4], 0x12, 99, [-1, -2, 99, -4], 0x13);

    run_cbuf_add_item([-1, -2, 99, -4], 0x13, 101, [-1, -2, 99, 101], 0x14);

    run_cbuf_add_item([-1, -2, 99, 101], 0x14, 202, [202, -2, 99, 101], 0x15);

    run_cbuf_add_item([-1, -2, -3, -4], usize::MAX - 1, 99, [-1, -2, 99, -4], usize::MAX);

    run_cbuf_add_item([-1, -2, 99, -4], usize::MAX, 101, [-1, -2, 99, 101], 4);
}

#[test]
fn write_then_read() {
    let mut cbuf: CBuf<u32, 16> = CBuf::new_with_default();
    let (buf_ptr, next_ptr) = cbuf.as_mut_ptrs();

    let mut writer = unsafe { CBufWriter::<_, 16>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();
    let mut reader1 = unsafe { CBufReader::<_, 16>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();

    assert_eq!(reader1.fetch_next_item(false), RR::None);

    writer.add_item(42);
    assert_eq!(reader1.fetch_next_item(false), RR::Success(42));

    let mut reader2 = unsafe { CBufReader::<_, 16>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();
    assert_eq!(reader2.fetch_next_item(false), RR::None);

    writer.add_item(43);
    assert_eq!(reader1.fetch_next_item(false), RR::Success(43));

    writer.add_item(44);
    assert_eq!(reader1.fetch_next_item(false), RR::Success(44));
    assert_eq!(reader1.fetch_next_item(false), RR::None);
    assert_eq!(reader2.fetch_next_item(false), RR::Success(43));
    assert_eq!(reader2.fetch_next_item(false), RR::Success(44));
    assert_eq!(reader2.fetch_next_item(false), RR::None);

    for _ in 0..3 {
        writer.add_item(99);
        assert_eq!(reader1.fetch_next_item(false), RR::Success(99));
    }

    for i in 100..115 {
        writer.add_item(i);
        assert_eq!(reader1.fetch_next_item(false), RR::Success(i));
    }

    assert_eq!(reader2.fetch_next_item(false), RR::Skipped(100));
    for i in 101..115 {
        assert_eq!(reader2.fetch_next_item(false), RR::Success(i));
    }

    for i in 200..215 {
        writer.add_item(i);
    }
    for i in 200..215 {
        assert_eq!(reader1.fetch_next_item(false), RR::Success(i));
    }

    writer.add_item(300);
    assert_eq!(reader2.fetch_next_item(true), RR::Skipped(300));

    drop(cbuf);
}

#[test]
fn write_wrap_read_search() {
    const SZ: usize = 16;
    let mut cbuf: CBuf<u16, SZ> = CBuf::new(9999);

    let ivalids = cbuf.as_reader().current_valid_index_range();
    assert_eq!(ivalids.start.as_usize(), 0);
    assert_eq!(ivalids.end.as_usize(), 0);

    let mut writer = cbuf.as_writer();

    for i in 0u16..3 * (SZ as u16) {
        writer.add_item(i);
    }

    let mut reader = cbuf.as_reader();

    let ivalids = reader.current_valid_index_range();

    assert_eq!(ivalids.start.as_usize(), 2 * SZ + 1);
    assert_eq!(ivalids.end.as_usize(), 3 * SZ);

    // reader.set_next_index(ivalids.start);
    reader.rewind();

    let mut i = ivalids.start.as_usize() as u16;
    for value in reader.available_items_iter(false) {
        match value {
            RR::Success(v) => {
                assert_eq!(v, i);
                i += 1;
            }
            _ => {
                panic!("Read back failed at {}: {:?}", i, value);
            }
        }
    }
    assert_eq!(i, 3 * SZ as u16);

    // Search for value no earlier than a read index
    let idx_partway = ivalids.start + 10;
    let partreader = reader.reader_starting_at(idx_partway);
    for i in ivalids.start.as_usize() - 1..ivalids.end.as_usize() + 1 {
        let predicate = |x| x >= i as u16;
        match partreader.search(&predicate, false) {
            Ok(mut foundreader) => {
                let idx = foundreader.current_next_index();
                let value = match foundreader.fetch_next_item(false) {
                    RR::Success(value) => value,
                    _ => {
                        panic!("Read after reduced search failed at {}", idx.as_usize());
                    }
                };
                assert_eq!(value, reader.fetch(idx).unwrap());
                assert!(predicate(value));
                assert!(idx >= idx_partway);
                if i >= idx_partway.as_usize() {
                    assert_eq!(idx.as_usize() as u16, value);
                } else {
                    assert!((i as u16) < value);
                }
            }
            Err(_) => {
                assert!(!predicate(partreader.fetch(ivalids.end - 1).unwrap()));
            }
        }
    }

    // Search for value from start
    for i in ivalids.start.as_usize() - 1..ivalids.end.as_usize() + 1 {
        let predicate = |x| x >= i as u16;
        match reader.search(&predicate, true) {
            Ok(mut foundreader) => {
                let idx = foundreader.current_next_index();
                let value = match foundreader.fetch_next_item(false) {
                    RR::Success(value) => value,
                    _ => {
                        panic!("Read after full search failed at {}", idx.as_usize());
                    }
                };
                assert!(predicate(value));
                if i >= ivalids.start.as_usize() {
                    assert_eq!(idx.as_usize() as u16, value);
                } else {
                    assert!((i as u16) < value);
                }
            }
            Err(_) => {
                assert!(!predicate(reader.fetch(ivalids.end - 1).unwrap()));
            }
        }
    }
}

macro_rules! assert_let {
    ($value:expr, $p:pat) => {
        let val = $value;
        match val {
            $p => (),
            _ => panic!(
                r"assertion failed: `left matches right`
left: `{:?}`
right: `{}`",
                val,
                stringify!($p)
            ),
        };
    };
}

macro_rules! step_writer {
    ($chan_pair:expr) => {
        let (ref goahead, ref result) = &$chan_pair;
        let _ = goahead.send(());
        assert_let!(result.recv(), Ok(_));
    };
}

macro_rules! step_reader {
    ($chan_pair:expr) => {
        let (ref goahead, ref result) = &$chan_pair;
        let _ = goahead.send(());
        assert_let!(result.recv(), Ok(FetchCheckpoint::Step(_)));
    };
}

macro_rules! expect_reader_ret {
    ($chan_pair:expr, $return_val:expr) => {
        let (ref goahead, ref result) = &$chan_pair;
        let _ = goahead.send(());
        assert_eq!(result.recv(), Ok(FetchCheckpoint::ReturnVal($return_val)));
    };
}

#[test]
fn very_simple_multithreaded() {
    let mut cbuf: CBuf<(u8, u16), 4> = CBuf::new((0, 0));
    let (buf_ptr, next_ptr) = cbuf.as_mut_ptrs();

    let mut writer = unsafe { CBufWriter::<_, 4>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();
    let mut reader = unsafe { CBufReader::<_, 4>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();

    let (ts_wr, chans_wr) = TestSequencer::new();
    let (ts_rd, chans_rd) = TestSequencer::new();

    let jh_wr = spawn(move || {
        writer.add_item_seq((1, 1), &ts_wr);
    });

    let jh_rd = spawn(move || {
        let _ = reader.fetch_next_item_seq(false, &ts_rd);
        let _ = reader.fetch_next_item_seq(false, &ts_rd);
    });

    for _ in 0..3 {
        step_reader!(chans_rd);
    }
    expect_reader_ret!(chans_rd, RR::None);

    for _ in 0..3 {
        step_writer!(chans_wr);
    }
    for _ in 0..3 {
        step_reader!(chans_rd);
    }
    expect_reader_ret!(chans_rd, RR::Success((1, 1)));

    for jh in [jh_wr, jh_rd] {
        let _ = jh.join();
    }

    drop(cbuf);
}

#[test]
fn simple_interleaved_read_and_write() {
    let mut cbuf: CBuf<i16, 16> = CBuf::new(0);
    let (buf_ptr, next_ptr) = cbuf.as_mut_ptrs();

    let mut writer = unsafe { CBufWriter::<_, 16>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();
    let mut reader = unsafe { CBufReader::<_, 16>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();

    cbuf.buf = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16];
    cbuf.next.store(CBufIndex::new(0x21), SeqCst);
    writer.next_local = CBufIndex::new(0x21);
    reader.next_local = CBufIndex::new(0x12);

    let (ts_wr, chans_wr) = TestSequencer::new();
    let (ts_rd, chans_rd) = TestSequencer::new();

    let jh_wr = spawn(move || {
        writer.add_item_seq(42, &ts_wr);
        writer.add_item_seq(43, &ts_wr);
        writer.add_item_seq(44, &ts_wr);
        writer.add_item_seq(45, &ts_wr);
    });
    let jh_rd = spawn(move || {
        let _ = reader.fetch_next_item_seq(false, &ts_rd);
        let _ = reader.fetch_next_item_seq(true, &ts_rd);
        let _ = reader.fetch_next_item_seq(false, &ts_rd);
    });

    step_reader!(chans_rd);
    for _ in 0..3 {
        step_writer!(chans_wr);
    }
    step_reader!(chans_rd);
    step_reader!(chans_rd);
    for _ in 0..3 {
        step_reader!(chans_rd);
    }
    expect_reader_ret!(chans_rd, RR::Skipped(-4));

    step_reader!(chans_rd);
    for _ in 0..(2 * 3) {
        step_writer!(chans_wr);
    }
    step_reader!(chans_rd);
    step_reader!(chans_rd);
    for _ in 0..3 {
        step_reader!(chans_rd);
    }
    expect_reader_ret!(chans_rd, RR::Skipped(44));

    step_reader!(chans_rd);
    for _ in 0..3 {
        step_writer!(chans_wr);
    }
    for _ in 0..2 {
        step_reader!(chans_rd);
    }
    for _ in 0..3 {
        step_reader!(chans_rd);
    }
    expect_reader_ret!(chans_rd, RR::Success(45));

    for jh in [jh_wr, jh_rd] {
        let _ = jh.join();
    }

    drop(cbuf);
}

#[test]
fn get_to_spinfail() {
    #[allow(non_snake_case)]
    let NUM_READ_TRIES = CBufReader::<(), 4096>::NUM_READ_TRIES;

    let mut cbuf: CBuf<(), 4096> = CBuf::new_with_default();
    let (buf_ptr, next_ptr) = cbuf.as_mut_ptrs();

    let mut writer = unsafe { CBufWriter::<_, 4096>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();
    let mut reader = unsafe { CBufReader::<_, 4096>::new_from_ptr(buf_ptr, next_ptr) }.unwrap();

    cbuf.next.store(CBufIndex::new(0x2005), SeqCst);
    writer.next_local = CBufIndex::new(0x2005);
    reader.next_local = CBufIndex::new(0x1006);

    let (ts_wr, chans_wr) = TestSequencer::new();
    let (ts_rd, chans_rd) = TestSequencer::new();

    let join_handles = [
        spawn(move || {
            for _ in 0..NUM_READ_TRIES {
                writer.add_item_seq((), &ts_wr);
            }
        }),
        spawn(move || {
            reader.fetch_next_item_seq(false, &ts_rd);
        }),
    ];

    for _ in 0..NUM_READ_TRIES {
        step_reader!(chans_rd);

        for _ in 0..3 {
            step_writer!(chans_wr);
        }

        step_reader!(chans_rd);
        step_reader!(chans_rd);
    }
    expect_reader_ret!(chans_rd, RR::SpinFail);

    for jh in join_handles {
        let _ = jh.join();
    }

    drop(cbuf);
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum TraceStep<T> {
    Reader(FetchCheckpoint<T>),
    Writer(WriteProtocolStep),
}

fn iterate_over_event_sequences<GEN, RD, WR, T, RDRET, const SIZE: usize, A>(
    generator: &GEN,
    read_thread_generator: &RD,
    initial_read_next: usize,
    write_thread_generator: &WR,
    action: &mut A,
) where
    T: CBufItem,
    GEN: Fn() -> CBuf<T, SIZE>,
    RD: Fn(CBufReader<'static, T, SIZE>, TestSequencer<FetchCheckpoint<RDRET>>) -> JoinHandle<()>,
    RDRET: Send + Clone,
    WR: Fn(CBufWriter<'static, T, SIZE>, TestSequencer<WriteProtocolStep>) -> JoinHandle<()>,
    A: FnMut(&[TraceStep<RDRET>]),
{
    use std::boxed::Box;

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum EventStep {
        Reader,
        Writer,
    }

    type ChannelPair<X> = (mpsc::Sender<()>, mpsc::Receiver<X>);

    struct RunningState<U, W, const SZ: usize>
    where
        U: CBufItem,
    {
        cbuf:     Box<CBuf<U, SZ>>,
        trace:    Vec<TraceStep<W>>,
        chans_wr: ChannelPair<WriteProtocolStep>,
        chans_rd: ChannelPair<FetchCheckpoint<W>>,
        jhandles: [JoinHandle<()>; 2],
    }

    // impl Fn(&[EventStep]) -> RunningState<T, SIZE>
    let mut replay_event_chain = |chain: &[EventStep]| {
        use EventStep as ES;
        use TraceStep as TS;

        //std::eprintln!("replaying {:?}", chain);

        let mut cbuf = Box::new(generator());
        let mut trace: Vec<TraceStep<RDRET>> = Vec::with_capacity(chain.len());

        let buf_writer =
            unsafe { CBufWriter::new_from_ptr(cbuf.buf.as_mut_ptr(), &cbuf.next.n) }.unwrap();
        let mut buf_reader =
            unsafe { CBufReader::new_from_ptr(cbuf.buf.as_ptr(), &cbuf.next.n) }.unwrap();
        buf_reader.next_local = CBufIndex::new(initial_read_next);

        let (ts_wr, chans_wr) = TestSequencer::new();
        let (ts_rd, chans_rd) = TestSequencer::new();

        let jhandles =
            [write_thread_generator(buf_writer, ts_wr), read_thread_generator(buf_reader, ts_rd)];

        for step in chain {
            match step {
                ES::Reader => {
                    let (goahead, result) = &chans_rd;
                    let _ = goahead.send(());
                    match result.recv() {
                        Ok(r) => {
                            trace.push(TS::Reader(r));
                        }
                        Err(e) => {
                            panic!(
                                "Unexpected error `{:?}` during replay of event chain {:?}",
                                e, chain
                            );
                        }
                    }
                }
                ES::Writer => {
                    let (goahead, result) = &chans_wr;
                    let _ = goahead.send(());
                    match result.recv() {
                        Ok(r) => {
                            trace.push(TS::Writer(r));
                        }
                        Err(e) => {
                            panic!(
                                "Unexpected error `{:?}` during replay of event chain {:?}",
                                e, chain
                            );
                        }
                    }
                }
            }
        }

        RunningState {
            cbuf,
            trace,
            chans_wr,
            chans_rd,
            jhandles,
        }
    };

    let root_state = replay_event_chain(&[]);

    let mut recurse_over_event_chains =
        |event_steps: Vec<EventStep>, currently_running_state: RunningState<T, RDRET, SIZE>| {
            // Helper function used to allow the logic of the closure to be recursive.
            // Technique due to <https://stackoverflow.com/a/72862424>.
            fn helper<REC, A, T, W, const SIZE: usize>(
                event_steps: Vec<EventStep>,
                currently_running_state: RunningState<T, W, SIZE>,
                replay_event_chain: &mut REC,
                action: &mut A,
            ) where
                T: CBufItem,
                REC: FnMut(&[EventStep]) -> RunningState<T, W, SIZE>,
                A: FnMut(&[TraceStep<W>]),
                TraceStep<W>: Clone,
            {
                use EventStep as ES;
                use TraceStep as TS;

                //std::eprintln!("recursing on {:?}", &event_steps[..]);

                // Try another writer step.
                let mut wr_events = event_steps.clone();
                let mut wr_trace = currently_running_state.trace.clone();

                let (goahead, result) = &currently_running_state.chans_wr;
                let _ = goahead.send(());
                match result.recv() {
                    Err(_) => {
                        // The writer thread has finished; run the reader to completion.
                        let (goahead, result) = &currently_running_state.chans_rd;
                        while let Ok(val) = {
                            let _ = goahead.send(());
                            result.recv()
                        } {
                            wr_trace.push(TS::Reader(val));
                        }
                        for jh in currently_running_state.jhandles {
                            let _ = jh.join();
                        }
                        drop(currently_running_state.cbuf);
                        //std::eprintln!("applying action on trace {:?}", &wr_events[..]);
                        action(&wr_trace);

                        // "Try another reader step" below would just
                        // repeat this sequence of events, so there's
                        // no need to run it.
                        return;
                    }
                    Ok(val) => {
                        wr_trace.push(TS::Writer(val));
                        wr_events.push(ES::Writer);
                        helper(
                            wr_events,
                            RunningState {
                                trace: wr_trace,
                                ..currently_running_state
                            },
                            replay_event_chain,
                            action,
                        );
                    }
                }

                // Try another reader step.
                // We consumed the passed-in threads in "try another writer step",
                // so we need to recreate them first.
                let mut reader_state = replay_event_chain(&event_steps);
                let mut reader_events = event_steps.clone();

                let (goahead, result) = &reader_state.chans_rd;
                let _ = goahead.send(());
                match result.recv() {
                    Err(_) => {
                        // The reader thread has finished. Wind up execution.
                        let (goahead, result) = &reader_state.chans_wr;
                        while let Ok(_) = {
                            let _ = goahead.send(());
                            result.recv()
                        } {}
                        for jh in reader_state.jhandles {
                            let _ = jh.join();
                        }
                        drop(reader_state.cbuf);

                        // If the situation was symmetrical between reader and writer,
                        // we'd now call action()... but we've already done that with
                        // this particular sequence of events in the recursion in
                        // "try another writer step" above, so we'd just be repeating
                        // that sequence.
                        return;
                    }
                    Ok(val) => {
                        reader_state.trace.push(TS::Reader(val));
                        reader_events.push(ES::Reader);
                        helper(reader_events, reader_state, replay_event_chain, action);
                    }
                }
            }
            helper(event_steps, currently_running_state, &mut replay_event_chain, action);
        };

    recurse_over_event_chains(Vec::new(), root_state);
}

// named after the unary iota function in APL
const fn iota<const SIZE: usize>() -> [u32; SIZE] {
    let mut x = [0u32; SIZE];
    let mut i = 0;
    while i < SIZE {
        x[i] = i as u32;
        i += 1;
    }
    x
}

#[test]
fn mini_model_example() {
    let mut i = 0usize;
    iterate_over_event_sequences(
        &|| CBuf {
            buf:  iota::<16>(),
            next: AtomicIndex::new(0x14),
        },
        &|mut reader, sequencer| {
            spawn(move || {
                let _ = reader.fetch_next_item_seq(false, &sequencer);
            })
        },
        0x13,
        &|mut writer, sequencer| {
            spawn(move || {
                writer.add_item_seq(42, &sequencer);
            })
        },
        &mut |trace| {
            std::eprintln!("{}: {:?}", i, trace);
            i += 1;
            assert!(trace.len() > 0);
        },
    );
    //panic!("panicking just so we can see the list of all traces");
}

/// A test case with one write and one read where neither should interfere with each other.
#[test]
fn mini_model_no_interference_1w1r() {
    use FetchCheckpoint as FC;
    use TraceStep as TS;

    iterate_over_event_sequences(
        &|| CBuf {
            buf:  iota::<16>(),
            next: AtomicIndex::new(0x14),
        },
        &|mut reader, sequencer| {
            spawn(move || {
                let _ = reader.fetch_next_item_seq(false, &sequencer);
            })
        },
        0x11,
        &|mut writer, sequencer| {
            spawn(move || {
                writer.add_item_seq(42, &sequencer);
            })
        },
        &mut |trace| {
            std::eprintln!("testing {:?}", trace);

            assert!(trace.len() > 0);

            assert_eq!(
                trace.iter().filter(|ref t| matches!(t, TS::Reader(FC::ReturnVal(_)))).count(),
                1
            );

            for t in trace {
                if let TS::Reader(FC::ReturnVal(val)) = *t {
                    assert_eq!(val, ReadResult::Success(1));
                }
            }
        },
    );
}

trait SliceExt {
    type Item;

    fn prev_with<F: Fn(&Self::Item) -> bool>(
        &self,
        index: usize,
        predicate: F,
    ) -> Option<Self::Item>;
    fn next_with<F: Fn(&Self::Item) -> bool>(
        &self,
        index: usize,
        predicate: F,
    ) -> Option<Self::Item>;
}

impl<T: Clone> SliceExt for [T] {
    type Item = T;

    fn prev_with<F: Fn(&Self::Item) -> bool>(
        &self,
        index: usize,
        predicate: F,
    ) -> Option<Self::Item> {
        let mut i = index;
        while i > 0 {
            i -= 1;
            if predicate(&self[i]) {
                return Some(self[i].clone());
            }
        }
        None
    }

    fn next_with<F: Fn(&Self::Item) -> bool>(
        &self,
        index: usize,
        predicate: F,
    ) -> Option<Self::Item> {
        for i in &self[(index + 1)..] {
            if predicate(i) {
                return Some(i.clone());
            }
        }
        None
    }
}

macro_rules! prev_with_pat {
    ($array:expr, $index:expr, $pattern:pat) => {
        $array.prev_with($index, |ref item| matches!(item, $pattern))
    };
}

macro_rules! next_with_pat {
    ($array:expr, $index:expr, $pattern:pat) => {
        $array.next_with($index, |ref item| matches!(item, $pattern))
    };
}

/// A test case with one write and one read where interference is possible in some cases.
#[test]
fn mini_model_interference_1w1r_no_ff() {
    use FetchCheckpoint as FC;
    use ReadProtocolStep as RPS;
    use TraceStep as TS;
    use WriteProtocolStep as WPS;

    iterate_over_event_sequences(
        &|| CBuf {
            buf:  iota::<16>(),
            next: AtomicIndex::new(0x23),
        },
        &|mut reader, sequencer| {
            spawn(move || {
                let _ = reader.fetch_next_item_seq(false, &sequencer);
            })
        },
        0x14,
        &|mut writer, sequencer| {
            spawn(move || {
                writer.add_item_seq(42, &sequencer);
            })
        },
        &mut |trace| {
            std::eprintln!("testing {:?}", trace);

            // something should have happened
            assert!(trace.len() > 0);

            // there should be one return from fetch_next_item_seq() :)
            assert_eq!(
                trace.iter().filter(|t| matches!(t, TS::Reader(FC::ReturnVal(_)))).count(),
                1
            );

            // return value should be one of Success(4), Skipped(5), and SpinFail
            let return_value =
                match prev_with_pat!(trace, trace.len(), TS::Reader(FC::ReturnVal(_))) {
                    Some(TS::Reader(FC::ReturnVal(rval))) => rval,
                    _ => unreachable!(),
                };
            assert_let!(return_value, RR::Success(4) | RR::Skipped(5) | RR::SpinFail);

            for (i, t) in trace.iter().enumerate() {
                // if, in the interval between the IndexCheckPre and IndexCheckPost,
                // an IndexPostUpdate occurs, the reader should do another read sequence
                // (or call it quits with a return of SpinFail)
                if *t == TS::Reader(FC::Step(RPS::IndexCheckPost)) {
                    let mut j = i;
                    let interleaved_postupdate = 'x: loop {
                        while j > 0 {
                            j -= 1;
                            match trace[j] {
                                TS::Reader(FC::Step(RPS::IndexCheckPre)) => {
                                    break 'x false;
                                }
                                TS::Writer(WPS::IndexPostUpdate) => {
                                    break 'x true;
                                }
                                _ => {}
                            }
                        }
                        panic!("An IndexCheckPost occurred without a preceding IndexCheckPre");
                    };
                    if interleaved_postupdate {
                        assert!(next_with_pat!(
                            trace,
                            i,
                            TS::Reader(FC::Step(RPS::IndexCheckPre))
                                | TS::Reader(FC::ReturnVal(RR::SpinFail))
                        )
                        .is_some());
                    }
                }

                // if the last read sequence occurs entirely before the write sequence,
                // we should return Success(4)
                if *t == TS::Reader(FC::Step(RPS::IndexCheckPost))
                    && next_with_pat!(trace, i, TS::Reader(FC::Step(RPS::IndexCheckPost))).is_none()
                {
                    if prev_with_pat!(trace, i, TS::Writer(_)).is_none() {
                        assert_eq!(return_value, RR::Success(4));
                    }
                }

                // if the last read sequence occurs entirely after the write sequence,
                // we should return Skipped(5) or, if *right* after the write sequence,
                // possibly SpinFail
                if *t == TS::Reader(FC::Step(RPS::IndexCheckPre))
                    && next_with_pat!(trace, i, TS::Reader(FC::Step(RPS::IndexCheckPre))).is_none()
                {
                    if next_with_pat!(trace, i, TS::Writer(_)).is_none() {
                        if i >= 1 && matches!(trace[i - 1], TS::Writer(_)) {
                            assert_let!(return_value, RR::Skipped(5) | RR::SpinFail);
                        } else {
                            assert_eq!(return_value, RR::Skipped(5));
                        }
                    }
                }
            }
        },
    );
}

#[test]
fn mini_model_interference_1w1r_ff() {
    use FetchCheckpoint as FC;
    use ReadProtocolStep as RPS;
    use TraceStep as TS;
    use WriteProtocolStep as WPS;

    iterate_over_event_sequences(
        &|| CBuf {
            buf:  iota::<16>(),
            next: AtomicIndex::new(0x23),
        },
        &|mut reader, sequencer| {
            spawn(move || {
                let _ = reader.fetch_next_item_seq(true, &sequencer);
            })
        },
        0x14,
        &|mut writer, sequencer| {
            spawn(move || {
                writer.add_item_seq(42, &sequencer);
            })
        },
        &mut |trace| {
            std::eprintln!("testing {:?}", trace);

            // something should have happened in the course of things:
            assert!(trace.len() > 0);

            // there should be only one return from fetch_next_item_seq()
            assert_eq!(
                trace.iter().filter(|t| matches!(t, TS::Reader(FC::ReturnVal(_)))).count(),
                1
            );

            // return value of fetch should be one of Success(4), Skipped(42), and SpinFail
            let return_value =
                match prev_with_pat!(trace, trace.len(), TS::Reader(FC::ReturnVal(_))) {
                    Some(TS::Reader(FC::ReturnVal(rval))) => rval,
                    _ => unreachable!(),
                };
            assert_let!(return_value, RR::Success(4) | RR::Skipped(42) | RR::SpinFail);

            for (i, t) in trace.iter().enumerate() {
                // if, in the interval between the IndexCheckPre and IndexCheckPost,
                // an IndexPostUpdate occurs, the reader should do another read sequence
                // (or call it quits with a return of SpinFail)
                if *t == TS::Reader(FC::Step(RPS::IndexCheckPost)) {
                    let mut j = i;
                    let interleaved_postupdate = 'x: loop {
                        while j > 0 {
                            j -= 1;
                            match trace[j] {
                                TS::Reader(FC::Step(RPS::IndexCheckPre)) => {
                                    break 'x false;
                                }
                                TS::Writer(WPS::IndexPostUpdate) => {
                                    break 'x true;
                                }
                                _ => {}
                            }
                        }
                        panic!("An IndexCheckPost occurred without a preceding IndexCheckPre");
                    };
                    if interleaved_postupdate {
                        assert!(next_with_pat!(
                            trace,
                            i,
                            TS::Reader(FC::Step(RPS::IndexCheckPre))
                                | TS::Reader(FC::ReturnVal(RR::SpinFail))
                        )
                        .is_some());
                    }
                }

                // if the last read sequence occurs entirely before the write sequence,
                // we should return Success(4)
                if *t == TS::Reader(FC::Step(RPS::IndexCheckPost))
                    && next_with_pat!(trace, i, TS::Reader(FC::Step(RPS::IndexCheckPost))).is_none()
                {
                    if prev_with_pat!(trace, i, TS::Writer(_)).is_none() {
                        assert_eq!(return_value, RR::Success(4));
                    }
                }

                // if the last read sequence occurs entirely after the write sequence,
                // we should return Skipped(42), or, if *right* after the write sequence,
                // possibly SpinFail
                if *t == TS::Reader(FC::Step(RPS::IndexCheckPre))
                    && next_with_pat!(trace, i, TS::Reader(FC::Step(RPS::IndexCheckPre))).is_none()
                {
                    if next_with_pat!(trace, i, TS::Writer(_)).is_none() {
                        if i >= 1 && matches!(trace[i - 1], TS::Writer(_)) {
                            assert_let!(return_value, RR::Skipped(42) | RR::SpinFail);
                        } else {
                            assert_let!(return_value, RR::Skipped(42));
                        }
                    }
                }
            }
        },
    );
}

#[test]
fn fetch_model_no_interference() {
    use FetchCheckpoint as FC;
    use TraceStep as TS;

    for (index, value) in [(0x15, 5), (0x17, 7), (0x20, 0), (0x22, 2)] {
        iterate_over_event_sequences(
            &|| CBuf {
                buf:  iota::<16>(),
                next: AtomicIndex::new(0x23),
            },
            &|reader, sequencer| {
                spawn(move || {
                    let _ = reader.fetch_seq(CBufIndex::new(index), &sequencer);
                })
            },
            0,
            &|mut writer, sequencer| {
                spawn(move || {
                    writer.add_item_seq(42, &sequencer);
                })
            },
            &mut |trace| {
                std::eprintln!("testing {:?}", trace);

                // something should have happened:
                assert!(trace.len() > 0);

                // the writer should proceed through the entire write protocol:
                assert_eq!(trace.iter().filter(|t| matches!(*t, TS::Writer(_))).count(), 3);

                // the reader should proceed through the entire read protocol exactly once:
                assert_eq!(
                    trace.iter().filter(|t| matches!(*t, TS::Reader(FC::Step(_)))).count(),
                    3
                );

                // there should be only one return from fetch_seq(),
                // and it should be Ok(value):
                let returns: Vec<_> = trace
                    .iter()
                    .filter_map(|t| {
                        if let TS::Reader(FC::ReturnVal(rv)) = *t {
                            Some(rv)
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(returns.len(), 1);
                assert_eq!(returns[0], Ok(value));
            },
        );
    }
}

#[test]
fn fetch_model_definite_interference() {
    use FetchCheckpoint as FC;
    use TraceStep as TS;

    iterate_over_event_sequences(
        &|| CBuf {
            buf:  iota::<16>(),
            next: AtomicIndex::new(0x23),
        },
        &|reader, sequencer| {
            spawn(move || {
                let _ = reader.fetch_seq(CBufIndex::new(0x13), &sequencer);
            })
        },
        0,
        &|mut writer, sequencer| {
            spawn(move || {
                writer.add_item_seq(42, &sequencer);
            })
        },
        &mut |trace| {
            std::eprintln!("testing {:?}", trace);

            // something should have happened:
            assert!(trace.len() > 0);

            // the writer should proceed through the entire write protocol:
            assert_eq!(trace.iter().filter(|t| matches!(*t, TS::Writer(_))).count(), 3);

            // the reader should proceed through the entire read protocol exactly once:
            assert_eq!(trace.iter().filter(|t| matches!(*t, TS::Reader(FC::Step(_)))).count(), 3);

            // there should be only one return from fetch_seq(),
            // and it should be Err:
            let returns: Vec<_> =
                trace
                    .iter()
                    .filter_map(|t| {
                        if let TS::Reader(FC::ReturnVal(rv)) = *t {
                            Some(rv)
                        } else {
                            None
                        }
                    })
                    .collect();
            assert_eq!(returns.len(), 1);
            assert!(matches!(returns[0], Err(_)));
        },
    );
}

#[test]
fn fetch_model_possible_interference() {
    use FetchCheckpoint as FC;
    use ReadProtocolStep as RPS;
    use TraceStep as TS;
    use WriteProtocolStep as WPS;

    let (mut trace_err, mut trace_ok) = (false, false);

    iterate_over_event_sequences(
        &|| CBuf {
            buf:  iota::<16>(),
            next: AtomicIndex::new(0x23),
        },
        &|reader, sequencer| {
            spawn(move || {
                let _ = reader.fetch_seq(CBufIndex::new(0x14), &sequencer);
            })
        },
        0,
        &|mut writer, sequencer| {
            spawn(move || {
                writer.add_item_seq(42, &sequencer);
            })
        },
        &mut |trace| {
            std::eprintln!("testing {:?}", trace);

            // something should have happened:
            assert!(trace.len() > 0);

            // the writer should proceed through the entire write protocol:
            assert_eq!(trace.iter().filter(|t| matches!(*t, TS::Writer(_))).count(), 3);

            // the reader should proceed through the entire read protocol exactly once:
            assert_eq!(trace.iter().filter(|t| matches!(*t, TS::Reader(FC::Step(_)))).count(), 3);

            // there should be only one return from fetch_seq(),
            // and it should be either Ok(4) or Err:
            let returns: Vec<_> =
                trace
                    .iter()
                    .filter_map(|t| {
                        if let TS::Reader(FC::ReturnVal(rv)) = *t {
                            Some(rv)
                        } else {
                            None
                        }
                    })
                    .collect();
            assert_eq!(returns.len(), 1);
            assert!(matches!(returns[0], Ok(4) | Err(_)));

            trace_ok |= matches!(returns[0], Ok(_));
            trace_err |= matches!(returns[0], Err(_));

            for (i, t) in trace.iter().enumerate() {
                // if, before IndexCheckPost,
                // an IndexPostUpdate occurs, the reader should return Err; otherwise,
                // it should return Ok.
                if *t == TS::Reader(FC::Step(RPS::IndexCheckPost)) {
                    let update_before_check =
                        prev_with_pat!(trace, i, TS::Writer(WPS::IndexPostUpdate)).is_some();

                    if update_before_check {
                        assert!(matches!(returns[0], Err(_)));
                    } else {
                        assert_eq!(returns[0], Ok(4));
                    }
                }
            }
        },
    );

    // some trace should to fetch a value, and some trace should succeeded:
    assert!(trace_err);
    assert!(trace_ok);
}
