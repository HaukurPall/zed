//! Terminal emulation backed by libghostty-vt.
//!
//! The PTY is still spawned through alacritty's `tty` module; this module owns
//! the IO loop that reads from it and feeds libghostty-vt.

use std::{
    borrow::Cow,
    cell::{Cell as StdCell, RefCell},
    convert::Infallible,
    io::{self, ErrorKind, Read, Write},
    num::NonZeroUsize,
    os::fd::AsRawFd,
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
};

use alacritty_terminal::{
    event::{OnResize, WindowSize},
    tty::{self, EventedPty, EventedReadWrite},
};
use anyhow::{Context as _, Result};
use futures::channel::mpsc::UnboundedSender;
use libghostty_vt::{
    Terminal as GhosttyTerminal,
    render::{CellIterator, CursorVisualStyle, RenderState, RowIterator},
    screen::{CellContentTag, CellWide},
    selection::{
        FormatOptions, SelectLineOptions, SelectWordOptions, Selection as GhosttySelection,
    },
    style::{StyleColor, Underline},
    terminal::{
        ClipboardWriteError, ConformanceLevel, CursorStyle, DeviceAttributeFeature,
        DeviceAttributes, DeviceType, Mode, Point as GhosttyPoint, PointCoordinate, PointSpace,
        PrimaryDeviceAttributes, ScrollViewport, SecondaryDeviceAttributes, SizeReportSize,
    },
};
use parking_lot::{Mutex, MutexGuard};
use polling::{Event as PollEvent, Events, PollMode, Poller};
use vte::ansi::{Color, NamedColor, Rgb};

use crate::{
    Cell, Content, Cursor, CursorShape, Hyperlink, HyperlinkData, IndexedCell, Modes, Point,
    PtyEvent, Range, RenderableCells, Scroll, Search, Selection, SelectionRange, SelectionSide,
    SelectionType, TerminalBackendEvent, TerminalBounds, ViMotion, adjusted_last_hovered_word,
    pty_info::ProcessIdGetter,
    terminal_settings::{AlternateScroll, CursorShape as SettingsCursorShape},
};

#[cfg(windows)]
compile_error!("the `ghostty` terminal backend does not support Windows yet");

/// Keys under which alacritty's `Pty` registers its file and child-exit
/// signal with the poller.
const PTY_READ_WRITE_TOKEN: usize = 0;
const PTY_CHILD_EVENT_TOKEN: usize = 1;

const READ_BUFFER_SIZE: usize = 0x10_0000;
/// Bytes parsed per lock acquisition, so rendering isn't starved during floods.
const MAX_LOCKED_READ: usize = u16::MAX as usize;

/// Characters that end a word for double-click selection, matching alacritty's.
const WORD_BOUNDARY_CHARS: &[char] = &[
    ',', '│', '`', '|', ':', '"', '\'', ' ', '(', ')', '[', ']', '{', '}', '<', '>', '\t', '─',
];

pub(super) type HyperlinkHandle = Infallible;

#[derive(Clone, Debug)]
pub(super) struct SearchData {
    regex: regex::Regex,
}

#[derive(Clone)]
pub(super) struct TermConfig {
    pub(super) scrolling_history: usize,
    cursor_shape: SettingsCursorShape,
    clipboard_writes: bool,
}

/// Zed-owned copy of a rendered cell.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) struct CellData {
    character: char,
    zerowidth: Option<Arc<[char]>>,
    fg: Color,
    bg: Color,
    flags: CellFlags,
    hyperlink: Option<Hyperlink>,
}

impl Default for CellData {
    fn default() -> Self {
        Self {
            character: ' ',
            zerowidth: None,
            fg: Color::Named(NamedColor::Foreground),
            bg: Color::Named(NamedColor::Background),
            flags: CellFlags::default(),
            hyperlink: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CellFlags(u16);

impl CellFlags {
    const INVERSE: u16 = 1 << 0;
    const WIDE_CHAR_SPACER: u16 = 1 << 1;
    const DIM: u16 = 1 << 2;
    const UNDERLINE: u16 = 1 << 3;
    const UNDERCURL: u16 = 1 << 4;
    const STRIKEOUT: u16 = 1 << 5;
    const BOLD: u16 = 1 << 6;
    const ITALIC: u16 = 1 << 7;

    fn has(self, flag: u16) -> bool {
        self.0 & flag != 0
    }

    fn set(&mut self, flag: u16, value: bool) {
        if value {
            self.0 |= flag;
        }
    }
}

pub(super) type GridIter<'a> = Box<dyn Iterator<Item = IndexedCell> + 'a>;

/// Reusable libghostty render objects; rebuilt in place on every snapshot.
struct RenderCache {
    state: RenderState<'static>,
    rows: RowIterator<'static>,
    cells: CellIterator<'static>,
}

pub(super) struct Term {
    terminal: GhosttyTerminal<'static, 'static>,
    render: RefCell<RenderCache>,
    config: TermConfig,
    /// The selection as Zed last set it; libghostty keeps its own copy in grid coordinates.
    selection: Option<Selection>,
    vi_mode: bool,
    cursor_blinking: StdCell<bool>,
    events: EventSink,
}

// SAFETY: libghostty-vt has no thread affinity. Its headers document using
// one terminal from an IO thread and a renderer thread as long as access is
// serialized by a lock, which `TermLock` provides; `Term` is never used
// without holding it.
unsafe impl Send for Term {}

pub(super) struct TermLock(Mutex<Term>);

impl TermLock {
    pub(super) fn lock(&self) -> MutexGuard<'_, Term> {
        self.0.lock()
    }

    pub(super) fn lock_unfair(&self) -> MutexGuard<'_, Term> {
        self.0.lock()
    }
}

#[derive(Clone)]
struct EventSink {
    sender: UnboundedSender<PtyEvent>,
    generation: u64,
}

impl EventSink {
    fn send(&self, event: TerminalBackendEvent) {
        self.sender
            .unbounded_send(PtyEvent::Event(event, self.generation))
            .ok();
    }
}

pub(super) fn display_only_term_config(
    scrolling_history: usize,
    cursor_shape: SettingsCursorShape,
) -> TermConfig {
    TermConfig {
        scrolling_history,
        cursor_shape,
        clipboard_writes: false,
    }
}

pub(super) fn pty_term_config(
    scrolling_history: usize,
    cursor_shape: SettingsCursorShape,
) -> TermConfig {
    TermConfig {
        scrolling_history,
        cursor_shape,
        clipboard_writes: true,
    }
}

pub(super) fn set_default_cursor_style(config: &mut TermConfig, cursor_shape: SettingsCursorShape) {
    config.cursor_shape = cursor_shape;
}

pub(super) fn apply_config(term: &TermLock, config: &TermConfig) {
    let mut term = term.lock();
    term.config = config.clone();
    apply_cursor_style(&mut term.terminal, config.cursor_shape);
}

fn apply_cursor_style(
    terminal: &mut GhosttyTerminal<'static, 'static>,
    shape: SettingsCursorShape,
) {
    let style = match shape {
        SettingsCursorShape::Block => CursorStyle::Block,
        SettingsCursorShape::Underline => CursorStyle::Underline,
        SettingsCursorShape::Bar => CursorStyle::Bar,
        SettingsCursorShape::Hollow => CursorStyle::BlockHollow,
    };
    terminal
        .set_default_cursor_style(Some(style))
        .expect("set default cursor style")
        .set_default_cursor_blink(Some(false))
        .expect("set default cursor blink");
}

const DEVICE_ATTRIBUTES: PrimaryDeviceAttributes = PrimaryDeviceAttributes::new(
    ConformanceLevel::VT220,
    &[
        DeviceAttributeFeature::COLUMNS_132,
        DeviceAttributeFeature::SELECTIVE_ERASE,
        DeviceAttributeFeature::ANSI_COLOR,
    ],
);

pub(super) fn new_term(
    config: &TermConfig,
    bounds: TerminalBounds,
    events_tx: UnboundedSender<PtyEvent>,
    alternate_scroll: AlternateScroll,
    generation: u64,
) -> Arc<TermLock> {
    let events = EventSink {
        sender: events_tx,
        generation,
    };
    let (columns, rows) = grid_size(bounds);
    let mut terminal = GhosttyTerminal::new(columns, rows).expect("create libghostty terminal");
    terminal
        .set_scrollback_max_lines(Some(config.scrolling_history))
        .expect("set scrollback limit");
    apply_cursor_style(&mut terminal, config.cursor_shape);
    if let AlternateScroll::Off = alternate_scroll {
        terminal
            .set_default_mode(Mode::ALT_SCROLL, false)
            .expect("disable alternate scroll");
    }

    let sink = events.clone();
    terminal
        .on_pty_write(move |_, data| {
            sink.send(TerminalBackendEvent::PtyWrite(
                String::from_utf8_lossy(data).into_owned(),
            ))
        })
        .expect("register pty write callback");
    let sink = events.clone();
    terminal
        .on_bell(move |_| sink.send(TerminalBackendEvent::Bell))
        .expect("register bell callback");
    let sink = events.clone();
    terminal
        .on_title_changed(move |terminal| {
            let title = terminal.title().expect("read title");
            sink.send(if title.is_empty() {
                TerminalBackendEvent::ResetTitle
            } else {
                TerminalBackendEvent::Title(String::from_utf8_lossy(title).into_owned())
            })
        })
        .expect("register title callback");
    terminal
        .on_size(|terminal| {
            Some(SizeReportSize {
                rows: terminal.rows().ok()?,
                columns: terminal.cols().ok()?,
                cell_width: terminal.width_px().ok()? / u32::from(terminal.cols().ok()?.max(1)),
                cell_height: terminal.height_px().ok()? / u32::from(terminal.rows().ok()?.max(1)),
            })
        })
        .expect("register size callback");
    terminal
        .on_device_attributes(|_| {
            Some(DeviceAttributes {
                primary: DEVICE_ATTRIBUTES,
                secondary: SecondaryDeviceAttributes {
                    device_type: DeviceType::VT220,
                    firmware_version: 1,
                    rom_cartridge: 0,
                },
                tertiary: Default::default(),
            })
        })
        .expect("register device attributes callback");
    terminal
        .on_xtversion(|_| Some("zed"))
        .expect("register xtversion callback");
    if config.clipboard_writes {
        let sink = events.clone();
        terminal
            .on_clipboard_write(move |_, write| {
                let text = write
                    .contents()
                    .find(|content| content.mime == b"text/plain")
                    .map(|content| String::from_utf8_lossy(content.data).into_owned());
                match text {
                    Some(text) => {
                        sink.send(TerminalBackendEvent::ClipboardStore(text));
                        write.reply(Ok(()), false);
                    }
                    None => write.reply(Err(ClipboardWriteError::Unsupported), false),
                }
            })
            .expect("register clipboard write callback");
    }

    let mut term = Term {
        terminal,
        render: RefCell::new(RenderCache {
            state: RenderState::new().expect("create render state"),
            rows: RowIterator::new().expect("create row iterator"),
            cells: CellIterator::new().expect("create cell iterator"),
        }),
        config: config.clone(),
        selection: None,
        vi_mode: false,
        cursor_blinking: StdCell::new(false),
        events,
    };
    resize(&mut term, bounds);
    Arc::new(TermLock(Mutex::new(term)))
}

fn grid_size(bounds: TerminalBounds) -> (u16, u16) {
    (
        bounds.num_columns().clamp(1, u16::MAX as usize) as u16,
        bounds.num_lines().clamp(1, u16::MAX as usize) as u16,
    )
}

fn window_size(bounds: TerminalBounds) -> WindowSize {
    let (num_cols, num_lines) = grid_size(bounds);
    WindowSize {
        num_lines,
        num_cols,
        cell_width: f32::from(bounds.cell_width()) as u16,
        cell_height: f32::from(bounds.line_height()) as u16,
    }
}

pub(super) fn resize(term: &mut Term, bounds: TerminalBounds) {
    let size = window_size(bounds);
    term.terminal
        .resize(
            size.num_cols,
            size.num_lines,
            u32::from(size.cell_width),
            u32::from(size.cell_height),
        )
        .expect("resize terminal");
}

/// Feeds bytes that arrive outside the PTY IO loop into a terminal.
pub(super) struct OutputParser;

impl OutputParser {
    pub(super) fn new() -> Self {
        Self
    }

    pub(super) fn advance(&mut self, term: &mut Term, bytes: &[u8]) {
        term.terminal.vt_write(bytes);
    }
}

// --- PTY ---

impl From<&tty::Pty> for ProcessIdGetter {
    fn from(pty: &tty::Pty) -> Self {
        Self::new(pty.file().as_raw_fd(), pty.child().id())
    }
}

pub(super) fn current_child_signal_mask() -> io::Result<tty::SignalMask> {
    tty::SignalMask::current()
}

pub(super) fn pty_options(
    shell: Option<(String, Vec<String>)>,
    working_directory: Option<PathBuf>,
    env: impl IntoIterator<Item = (String, String)>,
    child_signal_mask: Option<tty::SignalMask>,
) -> tty::Options {
    tty::Options {
        shell: shell.map(|(program, args)| tty::Shell::new(program, args)),
        working_directory,
        drain_on_exit: true,
        env: env.into_iter().collect(),
        child_signal_mask,
    }
}

pub(super) fn open_pty(
    options: &tty::Options,
    bounds: TerminalBounds,
    window_id: u64,
) -> io::Result<tty::Pty> {
    tty::new(options, window_size(bounds), window_id)
}

enum Msg {
    Input(Cow<'static, [u8]>),
    Resize(WindowSize),
    Shutdown,
}

pub(super) struct PtySender {
    sender: Sender<Msg>,
    poller: Arc<Poller>,
}

impl PtySender {
    fn send(&self, msg: Msg) -> bool {
        self.sender.send(msg).is_ok() && self.poller.notify().is_ok()
    }

    pub(super) fn notify(&self, input: impl Into<Cow<'static, [u8]>>) {
        let input = input.into();
        if !input.is_empty() {
            self.send(Msg::Input(input));
        }
    }

    pub(super) fn resize(&self, bounds: TerminalBounds) {
        if !self.send(Msg::Resize(window_size(bounds))) {
            log::error!("failed to resize pty: IO loop is gone");
        }
    }

    pub(super) fn shutdown(&self) {
        if !self.send(Msg::Shutdown) {
            log::debug!("failed to shut down pty IO loop: already gone");
        }
    }
}

pub(super) fn spawn_event_loop(
    term: Arc<TermLock>,
    events_tx: UnboundedSender<PtyEvent>,
    pty: tty::Pty,
    drain_on_exit: bool,
) -> Result<PtySender> {
    let poller = Arc::new(Poller::new().context("failed to create pty poller")?);
    let (sender, receiver) = mpsc::channel();
    let io_loop = IoLoop {
        term,
        events: EventSink {
            sender: events_tx,
            generation: 0,
        },
        pty,
        poller: poller.clone(),
        receiver,
        drain_on_exit,
        pending_writes: Default::default(),
    };
    std::thread::Builder::new()
        .name("terminal pty io".into())
        .spawn(move || io_loop.run())
        .context("failed to spawn pty IO thread")?;
    Ok(PtySender { sender, poller })
}

struct IoLoop {
    term: Arc<TermLock>,
    events: EventSink,
    pty: tty::Pty,
    poller: Arc<Poller>,
    receiver: Receiver<Msg>,
    drain_on_exit: bool,
    pending_writes: std::collections::VecDeque<(Cow<'static, [u8]>, usize)>,
}

impl IoLoop {
    fn run(mut self) {
        let mut interest = PollEvent::readable(0);
        // SAFETY: the pty outlives its registration; it is deregistered before being dropped.
        if let Err(error) = unsafe { self.pty.register(&self.poller, interest, PollMode::Level) } {
            log::error!("failed to register pty with poller: {error}");
            return;
        }

        let mut buffer = vec![0u8; READ_BUFFER_SIZE];
        let mut events = Events::with_capacity(NonZeroUsize::new(64).unwrap());
        'io: loop {
            events.clear();
            if let Err(error) = self.poller.wait(&mut events, None) {
                if error.kind() == ErrorKind::Interrupted {
                    continue;
                }
                log::error!("pty poll failed: {error}");
                break;
            }

            if !self.drain_messages() {
                break;
            }

            for event in events.iter() {
                match event.key {
                    PTY_CHILD_EVENT_TOKEN => {
                        if let Some(tty::ChildEvent::Exited(status)) = self.pty.next_child_event() {
                            if let Some(status) = status {
                                self.events.send(TerminalBackendEvent::ChildExit(status));
                            }
                            if self.drain_on_exit {
                                self.read(&mut buffer).ok();
                            }
                            self.events.send(TerminalBackendEvent::Exit);
                            self.events.send(TerminalBackendEvent::Wakeup);
                            break 'io;
                        }
                    }
                    PTY_READ_WRITE_TOKEN => {
                        if event.is_interrupt() {
                            continue;
                        }
                        if event.readable
                            && let Err(error) = self.read(&mut buffer)
                        {
                            // Linux reports EIO once the child side hangs up; the
                            // child-exit event follows.
                            #[cfg(target_os = "linux")]
                            if error.raw_os_error() == Some(libc::EIO) {
                                continue;
                            }
                            log::error!("failed to read from pty: {error}");
                            break 'io;
                        }
                        if event.writable
                            && let Err(error) = self.write()
                        {
                            log::error!("failed to write to pty: {error}");
                            break 'io;
                        }
                    }
                    _ => {}
                }
            }

            let needs_write = !self.pending_writes.is_empty();
            if needs_write != interest.writable {
                interest.writable = needs_write;
                if let Err(error) = self.pty.reregister(&self.poller, interest, PollMode::Level) {
                    log::error!("failed to update pty poll interest: {error}");
                    break;
                }
            }
        }
        self.pty.deregister(&self.poller).ok();
    }

    /// Returns false on shutdown.
    fn drain_messages(&mut self) -> bool {
        loop {
            match self.receiver.try_recv() {
                Ok(Msg::Input(input)) => self.pending_writes.push_back((input, 0)),
                Ok(Msg::Resize(size)) => self.pty.on_resize(size),
                Ok(Msg::Shutdown) | Err(TryRecvError::Disconnected) => return false,
                Err(TryRecvError::Empty) => return true,
            }
        }
    }

    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<()> {
        let mut parsed = 0;
        loop {
            match self.pty.reader().read(buffer) {
                Ok(0) => break,
                Ok(count) => {
                    self.term.lock().terminal.vt_write(&buffer[..count]);
                    parsed += count;
                    if parsed >= MAX_LOCKED_READ {
                        break;
                    }
                }
                Err(error) => match error.kind() {
                    ErrorKind::Interrupted => continue,
                    ErrorKind::WouldBlock => break,
                    _ => {
                        if parsed > 0 {
                            self.events.send(TerminalBackendEvent::Wakeup);
                        }
                        return Err(error);
                    }
                },
            }
        }
        if parsed > 0 {
            self.events.send(TerminalBackendEvent::Wakeup);
        }
        Ok(())
    }

    fn write(&mut self) -> std::io::Result<()> {
        while let Some((input, written)) = self.pending_writes.front_mut() {
            match self.pty.writer().write(&input[*written..]) {
                Ok(0) => break,
                Ok(count) => {
                    *written += count;
                    if *written == input.len() {
                        self.pending_writes.pop_front();
                    }
                }
                Err(error) => match error.kind() {
                    ErrorKind::Interrupted | ErrorKind::WouldBlock => break,
                    _ => return Err(error),
                },
            }
        }
        Ok(())
    }
}

// --- Coordinates ---

fn history_rows(term: &Term) -> usize {
    term.terminal
        .scrollback_rows()
        .expect("read scrollback rows")
}

fn screen_rows(term: &Term) -> usize {
    usize::from(term.terminal.rows().expect("read rows"))
}

fn screen_columns(term: &Term) -> usize {
    usize::from(term.terminal.cols().expect("read columns"))
}

/// Zed points count lines from the top of the active screen; scrollback is negative.
fn screen_point(term: &Term, point: Point) -> GhosttyPoint {
    let history = history_rows(term) as i64;
    let last_row = (history + screen_rows(term) as i64 - 1).max(0);
    let y = (i64::from(point.line) + history).clamp(0, last_row);
    let x = point.column.min(screen_columns(term).saturating_sub(1));
    GhosttyPoint::Screen(PointCoordinate {
        x: x as u16,
        y: y as u32,
    })
}

fn zed_point(term: &Term, coordinate: PointCoordinate) -> Point {
    Point::new(
        coordinate.y as i32 - history_rows(term) as i32,
        usize::from(coordinate.x),
    )
}

pub(super) fn display_offset(term: &Term) -> usize {
    let scrollbar = term.terminal.scrollbar().expect("read scrollbar");
    (scrollbar.total - scrollbar.offset - scrollbar.len) as usize
}

pub(super) fn history_size(term: &Term) -> usize {
    history_rows(term)
}

pub(super) fn cursor_line(term: &Term) -> i32 {
    i32::from(term.terminal.cursor_y().expect("read cursor row"))
}

pub(super) fn cursor_blinking(term: &Term) -> bool {
    term.cursor_blinking.get()
}

/// libghostty answers color queries itself, so Zed's fallback is never needed.
pub(super) fn palette_color(_term: &Term, _index: usize) -> Option<Rgb> {
    None
}

pub(super) fn total_lines(term: &Term) -> usize {
    term.terminal.total_rows().expect("read total rows")
}

pub(super) fn screen_lines(term: &Term) -> usize {
    screen_rows(term)
}

pub(super) fn used_lines(term: &Term) -> usize {
    if is_alt_screen(term) {
        return total_lines(term);
    }
    history_rows(term) + cursor_line(term).max(0) as usize + 1
}

fn is_alt_screen(term: &Term) -> bool {
    term.terminal
        .mode(Mode::ALT_SCREEN_SAVE)
        .expect("read mode")
        || term.terminal.mode(Mode::ALT_SCREEN).expect("read mode")
        || term
            .terminal
            .mode(Mode::ALT_SCREEN_LEGACY)
            .expect("read mode")
}

pub(super) fn scroll_display(term: &mut Term, scroll: Scroll) {
    let rows = screen_rows(term) as isize;
    // Zed's deltas count lines scrolled back into history; libghostty's count towards the bottom.
    let scroll = match scroll {
        Scroll::Delta(delta) => ScrollViewport::Delta(-(delta as isize)),
        Scroll::PageUp => ScrollViewport::Delta(-rows),
        Scroll::PageDown => ScrollViewport::Delta(rows),
        Scroll::Top => ScrollViewport::Top,
        Scroll::Bottom => ScrollViewport::Bottom,
    };
    term.terminal.scroll_viewport(scroll);
}

pub(super) fn scroll_to_point(term: &mut Term, point: Point) {
    let GhosttyPoint::Screen(coordinate) = screen_point(term, point) else {
        unreachable!()
    };
    let top = display_top_row(term);
    let rows = screen_rows(term);
    let y = coordinate.y as usize;
    if y < top || y >= top + rows {
        let row = y.saturating_sub(rows / 2);
        term.terminal.scroll_viewport(ScrollViewport::Row(row));
    }
}

fn display_top_row(term: &Term) -> usize {
    let scrollbar = term.terminal.scrollbar().expect("read scrollbar");
    scrollbar.offset as usize
}

// --- Selection ---

pub(super) fn set_selection(term: &mut Term, selection: Option<&Selection>) {
    term.selection = selection.cloned();
    install_selection(term);
}

pub(super) fn update_selection(term: &mut Term, point: Point, side: SelectionSide) -> bool {
    let Some(selection) = term.selection.as_mut() else {
        return false;
    };
    selection.update(point, side);
    install_selection(term);
    true
}

fn install_selection(term: &Term) {
    let selection = term
        .selection
        .as_ref()
        .and_then(|selection| ghostty_selection(term, selection));
    term.terminal
        .set_selection(selection.as_ref())
        .expect("set selection");
}

fn ghostty_selection<'t>(term: &'t Term, selection: &Selection) -> Option<GhosttySelection<'t>> {
    let (mut start, mut end) = (selection.start, selection.end);
    if (end.point, end.side == SelectionSide::Right)
        < (start.point, start.side == SelectionSide::Right)
    {
        std::mem::swap(&mut start, &mut end);
    }
    let grid_ref = |point| term.terminal.grid_ref(screen_point(term, point)).ok();
    match selection.ty {
        SelectionType::Simple => {
            // A side says which half of the cell the pointer was on; a cell is
            // selected once the pointer crosses its middle.
            let mut start_point = start.point;
            if start.side == SelectionSide::Right {
                start_point.column += 1;
            }
            let mut end_point = end.point;
            if end.side == SelectionSide::Left {
                if end_point.column == 0 {
                    return None;
                }
                end_point.column -= 1;
            }
            if end_point < start_point {
                return None;
            }
            Some(GhosttySelection::new(
                grid_ref(start_point)?,
                grid_ref(end_point)?,
                false,
            ))
        }
        SelectionType::Semantic => {
            let word = |point| {
                term.terminal
                    .select_word(
                        SelectWordOptions::new(grid_ref(point)?)
                            .with_boundary_codepoints(WORD_BOUNDARY_CHARS),
                    )
                    .ok()
                    .flatten()
            };
            let first = word(start.point)?;
            let last = word(end.point)?;
            Some(GhosttySelection::new(first.start(), last.end(), false))
        }
        SelectionType::Lines => {
            let line = |point| {
                term.terminal
                    .select_line(SelectLineOptions::new(grid_ref(point)?))
                    .ok()
                    .flatten()
            };
            let first = line(start.point)?;
            let last = line(end.point)?;
            Some(GhosttySelection::new(first.start(), last.end(), false))
        }
    }
}

pub(super) fn selection_text(term: &Term) -> Option<String> {
    let bytes = term
        .terminal
        .format_selection_alloc(None, FormatOptions::new().with_unwrap(true).with_trim(true))
        .expect("format selection")?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn selection_range(term: &Term) -> Option<SelectionRange> {
    let selection = term.terminal.selection().expect("read selection")?;
    let point = |grid_ref| {
        term.terminal
            .point_from_grid_ref(&grid_ref, PointSpace::Screen)
            .expect("resolve selection point")
            .map(|coordinate| zed_point(term, coordinate))
    };
    let (start, end) = (point(selection.start())?, point(selection.end())?);
    Some(SelectionRange {
        start: start.min(end),
        end: start.max(end),
        is_block: selection.is_rectangle(),
    })
}

// --- Vi mode (not supported by this backend yet) ---

pub(super) fn toggle_vi_mode(term: &mut Term) {
    term.vi_mode = !term.vi_mode;
}

pub(super) fn vi_motion(_term: &mut Term, _motion: ViMotion) {}

pub(super) fn vi_goto_point(term: &mut Term, point: Point) {
    scroll_to_point(term, point);
}

pub(super) fn update_vi_cursor_for_scroll(_term: &mut Term, _scroll: Scroll) {}

pub(super) fn update_selection_to_vi_cursor(_term: &mut Term) -> Option<Point> {
    None
}

// --- Content ---

fn modes(term: &Term) -> Modes {
    let mode = |mode| term.terminal.mode(mode).expect("read mode");
    let mut modes = Modes::empty();
    for (ghostty_mode, zed_mode) in [
        (Mode::DECCKM, Modes::APP_CURSOR),
        (Mode::KEYPAD_KEYS, Modes::APP_KEYPAD),
        (Mode::CURSOR_VISIBLE, Modes::SHOW_CURSOR),
        (Mode::WRAPAROUND, Modes::LINE_WRAP),
        (Mode::ORIGIN, Modes::ORIGIN),
        (Mode::INSERT, Modes::INSERT),
        (Mode::LINEFEED, Modes::LINE_FEED_NEW_LINE),
        (Mode::FOCUS_EVENT, Modes::FOCUS_IN_OUT),
        (Mode::ALT_SCROLL, Modes::ALTERNATE_SCROLL),
        (Mode::BRACKETED_PASTE, Modes::BRACKETED_PASTE),
        (Mode::SGR_MOUSE, Modes::SGR_MOUSE),
        (Mode::UTF8_MOUSE, Modes::UTF8_MOUSE),
        (Mode::NORMAL_MOUSE, Modes::MOUSE_REPORT_CLICK),
        (Mode::BUTTON_MOUSE, Modes::MOUSE_DRAG),
        (Mode::ANY_MOUSE, Modes::MOUSE_MOTION),
    ] {
        if mode(ghostty_mode) {
            modes.insert(zed_mode);
        }
    }
    if is_alt_screen(term) {
        modes.insert(Modes::ALT_SCREEN);
    }
    if term.vi_mode {
        modes.insert(Modes::VI);
    }
    modes
}

fn zed_color(color: StyleColor, default: NamedColor) -> Color {
    match color {
        StyleColor::None => Color::Named(default),
        StyleColor::Palette(index) => indexed_color(index.0),
        StyleColor::Rgb(rgb) => Color::Spec(Rgb {
            r: rgb.r,
            g: rgb.g,
            b: rgb.b,
        }),
    }
}

/// The first 16 palette entries are the theme's named ANSI colors.
fn indexed_color(index: u8) -> Color {
    const NAMED: [NamedColor; 16] = [
        NamedColor::Black,
        NamedColor::Red,
        NamedColor::Green,
        NamedColor::Yellow,
        NamedColor::Blue,
        NamedColor::Magenta,
        NamedColor::Cyan,
        NamedColor::White,
        NamedColor::BrightBlack,
        NamedColor::BrightRed,
        NamedColor::BrightGreen,
        NamedColor::BrightYellow,
        NamedColor::BrightBlue,
        NamedColor::BrightMagenta,
        NamedColor::BrightCyan,
        NamedColor::BrightWhite,
    ];
    match NAMED.get(usize::from(index)) {
        Some(named) => Color::Named(*named),
        None => Color::Indexed(index),
    }
}

/// Copies the visible cells out of libghostty's render state.
fn visible_cells(term: &Term) -> (Vec<IndexedCell>, usize) {
    let display_offset = display_offset(term) as i32;
    let mut cache = term.render.borrow_mut();
    let RenderCache { state, rows, cells } = &mut *cache;
    let snapshot = state.update(&term.terminal).expect("update render state");
    let blinking = snapshot.cursor_blinking().expect("read cursor blinking");
    if blinking != term.cursor_blinking.replace(blinking) {
        term.events.send(TerminalBackendEvent::CursorBlinkingChange);
    }

    let mut output = Vec::with_capacity(
        usize::from(snapshot.rows().expect("read rows"))
            * usize::from(snapshot.cols().expect("read columns")),
    );
    let mut graphemes = Vec::new();
    let mut uri = Vec::new();
    let mut row_iteration = rows.update(&snapshot).expect("iterate rows");
    let mut viewport_row: u32 = 0;
    while let Some(row) = row_iteration.next() {
        let line = viewport_row as i32 - display_offset;
        let mut cell_iteration = cells.update(row).expect("iterate cells");
        let mut column = 0;
        while let Some(cell) = cell_iteration.next() {
            let raw = cell.raw_cell().expect("read cell");
            let style = cell.style().expect("read cell style");
            let mut data = CellData {
                fg: zed_color(style.fg_color, NamedColor::Foreground),
                bg: match raw.content_tag().expect("read cell content") {
                    CellContentTag::BgColorPalette => {
                        indexed_color(raw.bg_color_palette().expect("read cell background").0)
                    }
                    CellContentTag::BgColorRgb => {
                        let rgb = raw.bg_color_rgb().expect("read cell background");
                        Color::Spec(Rgb {
                            r: rgb.r,
                            g: rgb.g,
                            b: rgb.b,
                        })
                    }
                    _ => zed_color(style.bg_color, NamedColor::Background),
                },
                ..CellData::default()
            };

            graphemes.resize(cell.graphemes_len().expect("read graphemes") as usize, ' ');
            cell.graphemes_buf(&mut graphemes).expect("read graphemes");
            if let Some((&first, rest)) = graphemes.split_first() {
                data.character = first;
                if !rest.is_empty() {
                    data.zerowidth = Some(Arc::from(rest));
                }
            }

            let flags = &mut data.flags;
            flags.set(CellFlags::INVERSE, style.inverse);
            flags.set(
                CellFlags::WIDE_CHAR_SPACER,
                matches!(
                    raw.wide().expect("read cell width"),
                    CellWide::SpacerTail | CellWide::SpacerHead
                ),
            );
            flags.set(CellFlags::DIM, style.faint);
            flags.set(CellFlags::UNDERLINE, style.underline != Underline::None);
            flags.set(CellFlags::UNDERCURL, style.underline == Underline::Curly);
            flags.set(CellFlags::STRIKEOUT, style.strikethrough);
            flags.set(CellFlags::BOLD, style.bold);
            flags.set(CellFlags::ITALIC, style.italic);

            if raw.has_hyperlink().expect("read cell hyperlink")
                && let Ok(grid_ref) =
                    term.terminal
                        .grid_ref(GhosttyPoint::Viewport(PointCoordinate {
                            x: column,
                            y: viewport_row,
                        }))
            {
                uri.resize(uri.capacity().max(256), 0);
                let len = loop {
                    match grid_ref.hyperlink_uri(&mut uri) {
                        Ok(len) => break len,
                        Err(libghostty_vt::Error::OutOfSpace { required }) => {
                            uri.resize(required, 0)
                        }
                        Err(error) => panic!("read hyperlink uri: {error}"),
                    }
                };
                data.hyperlink = Some(Hyperlink::new(
                    None::<String>,
                    String::from_utf8_lossy(&uri[..len]).into_owned(),
                ));
            }

            output.push(IndexedCell {
                point: Point::new(line, usize::from(column)),
                cell: Cell { cell: data },
            });
            column += 1;
        }
        viewport_row += 1;
    }
    (output, display_offset as usize)
}

fn cursor(term: &Term) -> Cursor {
    let mut cache = term.render.borrow_mut();
    let snapshot = cache
        .state
        .update(&term.terminal)
        .expect("update render state");
    let cursor = snapshot.cursor().expect("read cursor");
    let shape = if !cursor.visible {
        CursorShape::Hidden
    } else {
        match cursor.visual_style {
            CursorVisualStyle::Bar => CursorShape::Bar,
            CursorVisualStyle::Underline => CursorShape::Underline,
            CursorVisualStyle::BlockHollow => CursorShape::HollowBlock,
            _ => CursorShape::Block,
        }
    };
    Cursor {
        shape,
        point: Point::new(
            cursor_line(term),
            usize::from(term.terminal.cursor_x().expect("read cursor column")),
        ),
    }
}

pub(super) fn make_content(term: &Term, last_content: &Content) -> Content {
    let (cells, display_offset) = visible_cells(term);
    let cursor = cursor(term);
    let columns = screen_columns(term);
    let screen_lines = screen_rows(term);
    let total_lines = total_lines(term);
    let history = history_rows(term);

    let cursor_char = cells
        .iter()
        .find(|cell| cell.point == cursor.point)
        .map_or(' ', |cell| cell.character());

    let bottom_line = screen_lines as i32 - 1 - display_offset as i32;
    let bottom_row_occupied = cursor.point.line >= bottom_line
        || cells
            .iter()
            .rev()
            .take_while(|cell| cell.point.line >= bottom_line)
            .any(|cell| cell.character() != ' ');

    let (last_hovered_word, grid_lines_change) = adjusted_last_hovered_word(
        columns,
        screen_lines,
        total_lines,
        display_offset,
        last_content,
    );

    let selection = selection_range(term);
    Content {
        cells,
        mode: modes(term),
        total_lines,
        display_offset,
        columns,
        screen_lines,
        selection_text: selection.and_then(|_| selection_text(term)),
        selection,
        cursor,
        cursor_char,
        terminal_bounds: last_content.terminal_bounds,
        last_hovered_word,
        grid_lines_change,
        scrolled_to_top: display_offset == history,
        scrolled_to_bottom: display_offset == 0,
        bottom_row_occupied,
    }
}

pub(super) fn renderable_cells(term: &Term) -> RenderableCells<'static> {
    RenderableCells {
        cells: Box::new(visible_cells(term).0.into_iter()),
    }
}

/// All text in the terminal, scrollback included.
pub(super) fn content_text(term: &Term) -> String {
    let Some(all) = term.terminal.select_all().expect("select all") else {
        return String::new();
    };
    let bytes = term
        .terminal
        .format_selection_alloc(
            None,
            FormatOptions::new()
                .with_selection(&all)
                .with_unwrap(true)
                .with_trim(true),
        )
        .expect("format terminal content");
    bytes.map_or_else(String::new, |bytes| {
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

pub(super) fn full_content_range(term: &Term) -> Range {
    Range::new(
        Point::new(-(history_rows(term) as i32), 0),
        Point::new(
            screen_rows(term) as i32 - 1,
            screen_columns(term).saturating_sub(1),
        ),
    )
}

pub(super) fn last_non_empty_lines(term: &Term, line_count: usize) -> Vec<String> {
    let text = content_text(term);
    let mut lines = text
        .lines()
        .rev()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .take(line_count)
        .map(str::to_string)
        .collect::<Vec<_>>();
    lines.reverse();
    lines
}

/// Clears the scrollback and moves the cursor's line to the top of the screen.
pub(super) fn clear_saved_screen(term: &mut Term) {
    let cursor_row = term.terminal.cursor_y().expect("read cursor row");
    let column = term.terminal.cursor_x().expect("read cursor column");
    let mut sequence = String::from("\x1b[3J");
    if cursor_row > 0 {
        sequence.push_str(&format!("\x1b[{cursor_row}S\x1b[3J"));
    }
    sequence.push_str(&format!("\x1b[1;{}H", column + 1));
    term.terminal.vt_write(sequence.as_bytes());
}

/// Not supported by this backend yet: the scrollback keeps its size.
pub(super) fn shrink_to_used(_term: &mut Term) {}

/// Appends a stringified task summary to the terminal, after its output.
///
/// SAFETY: kept `unsafe` to match the alacritty backend; writing through the
/// parser keeps libghostty's state consistent.
pub(super) unsafe fn append_text_to_term(term: &mut Term, text_lines: &[&str]) {
    let mut text = String::from("\r\n");
    for line in text_lines {
        text.push_str(line);
        text.push_str("\r\n");
    }
    term.terminal.vt_write(text.as_bytes());
}

// --- Search and hyperlinks ---

impl Search {
    pub fn new(search: &str) -> Option<Self> {
        Some(Self {
            search: SearchData {
                regex: regex::Regex::new(search).ok()?,
            },
        })
    }
}

/// Not supported by this backend yet.
pub(super) fn search_matches(_term: &Term, _searcher: Search) -> Vec<Range> {
    Vec::new()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HyperlinkMatch {
    pub(crate) text: String,
    pub(crate) is_url: bool,
    pub(crate) range: Range,
}

#[derive(Default)]
pub(crate) struct RegexSearches;

impl RegexSearches {
    pub(crate) fn new(
        _path_hyperlink_regexes: impl IntoIterator<Item: AsRef<str>>,
        _path_hyperlink_timeout: std::time::Duration,
    ) -> Self {
        Self
    }
}

/// Finds the OSC 8 hyperlink under `point`; URL and path detection are not supported yet.
pub(super) fn find_from_terminal_point(
    term: &Term,
    point: Point,
    _regex_searches: &mut RegexSearches,
    _path_style: util::paths::PathStyle,
) -> Option<HyperlinkMatch> {
    let grid_ref = term.terminal.grid_ref(screen_point(term, point)).ok()?;
    let mut uri = vec![0; 2048];
    let len = grid_ref.hyperlink_uri(&mut uri).ok()?;
    if len == 0 {
        return None;
    }
    Some(HyperlinkMatch {
        text: String::from_utf8_lossy(&uri[..len]).into_owned(),
        is_url: true,
        range: Range::new(point, point),
    })
}

// --- Zed type impls ---

impl Hyperlink {
    pub fn new<T: ToString>(id: Option<T>, uri: String) -> Self {
        Self {
            data: HyperlinkData::Owned {
                id: id.map(|id| Arc::from(id.to_string())),
                uri: Arc::from(uri),
            },
        }
    }

    pub fn id(&self) -> Option<&str> {
        match &self.data {
            HyperlinkData::Backend(never) => match *never {},
            HyperlinkData::Owned { id, .. } => id.as_deref(),
        }
    }

    pub fn uri(&self) -> &str {
        match &self.data {
            HyperlinkData::Backend(never) => match *never {},
            HyperlinkData::Owned { uri, .. } => uri,
        }
    }
}

impl Cell {
    #[inline]
    pub fn character(&self) -> char {
        self.cell.character
    }

    #[cfg(test)]
    pub(crate) fn set_character(&mut self, character: char) {
        self.cell.character = character;
    }

    #[inline]
    pub fn foreground(&self) -> Color {
        self.cell.fg
    }

    #[inline]
    pub fn background(&self) -> Color {
        self.cell.bg
    }

    #[inline]
    pub fn zerowidth(&self) -> Option<&[char]> {
        self.cell.zerowidth.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn push_zerowidth(&mut self, character: char) {
        let mut chars = self.cell.zerowidth.as_deref().unwrap_or_default().to_vec();
        chars.push(character);
        self.cell.zerowidth = Some(Arc::from(chars));
    }

    #[inline]
    pub fn hyperlink(&self) -> Option<Hyperlink> {
        self.cell.hyperlink.clone()
    }

    #[inline]
    pub fn is_inverse(&self) -> bool {
        self.cell.flags.has(CellFlags::INVERSE)
    }

    #[inline]
    pub fn is_wide_char_spacer(&self) -> bool {
        self.cell.flags.has(CellFlags::WIDE_CHAR_SPACER)
    }

    #[inline]
    pub fn is_dim(&self) -> bool {
        self.cell.flags.has(CellFlags::DIM)
    }

    #[inline]
    pub fn has_underline(&self) -> bool {
        self.cell.flags.has(CellFlags::UNDERLINE)
    }

    #[inline]
    pub fn has_undercurl(&self) -> bool {
        self.cell.flags.has(CellFlags::UNDERCURL)
    }

    #[inline]
    pub fn has_strikeout(&self) -> bool {
        self.cell.flags.has(CellFlags::STRIKEOUT)
    }

    #[inline]
    pub fn is_bold(&self) -> bool {
        self.cell.flags.has(CellFlags::BOLD)
    }

    #[inline]
    pub fn is_italic(&self) -> bool {
        self.cell.flags.has(CellFlags::ITALIC)
    }

    #[inline]
    pub fn has_visible_style_modifier(&self) -> bool {
        self.has_underline() || self.is_inverse() || self.has_strikeout()
    }
}

impl Iterator for RenderableCells<'_> {
    type Item = IndexedCell;

    fn next(&mut self) -> Option<Self::Item> {
        self.cells.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.cells.size_hint()
    }
}
