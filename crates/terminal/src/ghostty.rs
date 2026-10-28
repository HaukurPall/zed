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
    ops::Range as StdRange,
    os::fd::AsRawFd,
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    time::{Duration, Instant},
};

use alacritty_terminal::{
    event::{OnResize, WindowSize},
    tty::{self, EventedPty, EventedReadWrite},
};
use anyhow::{Context as _, Result};
use futures::channel::mpsc::UnboundedSender;
use libghostty_vt::{
    Terminal as GhosttyTerminal,
    render::{
        CellIterator, CursorVisualStyle, Dirty, RenderState, RowId, RowIteration, RowIterator,
    },
    screen::{CellContentTag, CellWide, GridRef, Screen, TrackedGridRef},
    selection::{FormatOptions, Selection as GhosttySelection},
    style::{Palette, RgbColor, StyleColor, Underline},
    terminal::{
        ClipboardWriteError, ConformanceLevel, CursorStyle, DeviceAttributeFeature,
        DeviceAttributes, DeviceType, Mode, Point as GhosttyPoint, PointCoordinate, PointSpace,
        PrimaryDeviceAttributes, ScrollViewport, SecondaryDeviceAttributes, SizeReportSize,
    },
};
use parking_lot::{Mutex, MutexGuard};
use polling::{Event as PollEvent, Events, PollMode, Poller};
use theme::Theme;
use util::paths::{PathStyle, UrlExt as _};
use vte::ansi::{Color, NamedColor, Rgb};

use crate::{
    Cell, Content, Cursor, CursorShape, Hyperlink, HyperlinkData, IndexedCell, Modes, Point,
    PtyEvent, Range, RenderableCells, Scroll, Search, Selection, SelectionAnchor, SelectionRange,
    SelectionSide, SelectionType, TerminalBackendEvent, TerminalBounds, ViMotion,
    adjusted_last_hovered_word, get_color_at_index,
    mappings::colors::to_vte_rgb,
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

/// How long a program may hold the screen before its updates are shown anyway.
const RENDER_HOLD_TIMEOUT: Duration = Duration::from_secs(1);

/// DECSCUSR 0: switch to the default cursor style.
const RESET_CURSOR_STYLE: &[u8] = b"\x1b[0 q";

/// libghostty's render objects and the cells copied out of them.
struct Renderer {
    state: RenderState<'static>,
    rows: RowIterator<'static>,
    cells: CellIterator<'static>,
    /// Each viewport row's cells, kept until libghostty marks the row dirty.
    row_cells: Vec<(RowId, Vec<CellData>)>,
    /// The frame on screen when the running program began a render hold.
    held: Option<HeldFrame>,
}

#[derive(Clone)]
struct Frame {
    cells: Vec<IndexedCell>,
    display_offset: usize,
    cursor: Cursor,
    cursor_blinking: bool,
}

struct HeldFrame {
    frame: Frame,
    since: Instant,
}

pub(super) struct Term {
    terminal: GhosttyTerminal<'static, 'static>,
    /// Shared with the render hold callback, which captures a frame into it.
    renderer: Rc<RefCell<Renderer>>,
    config: TermConfig,
    /// The theme whose colors libghostty reports to programs.
    theme: Option<Arc<Theme>>,
    selection: Option<TermSelection>,
    vi_mode: bool,
    cursor_blinking: StdCell<bool>,
    events: EventSink,
}

/// The selection as Zed last set it; libghostty keeps its own copy in grid coordinates.
struct TermSelection {
    selection: Selection,
    /// Follows the start's cell as output scrolls it.
    start: TrackedGridRef,
    /// The screen `start` belongs to.
    screen: Screen,
}

// SAFETY: libghostty-vt has no thread affinity. Its headers document using
// one terminal from an IO thread and a renderer thread as long as access is
// serialized by a lock, which `TermLock` provides; `Term` is never used
// without holding it. The `Rc` is shared only with callbacks owned by the
// terminal, which run under the same lock.
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
    // Applies at the next DECSCUSR reset, so a style the program chose is kept.
    apply_cursor_style(&mut term.terminal, config.cursor_shape);
}

/// Gives libghostty the theme's colors, which it reports to programs that ask.
pub(super) fn set_theme(term: &mut Term, theme: &Arc<Theme>) {
    if term
        .theme
        .as_ref()
        .is_some_and(|applied| Arc::ptr_eq(applied, theme))
    {
        return;
    }
    let color = |index| {
        let rgb = to_vte_rgb(get_color_at_index(index, theme));
        RgbColor {
            r: rgb.r,
            g: rgb.g,
            b: rgb.b,
        }
    };
    term.terminal
        .set_default_fg_color(Some(color(256)))
        .expect("set foreground color")
        .set_default_bg_color(Some(color(257)))
        .expect("set background color")
        .set_default_cursor_color(Some(color(258)))
        .expect("set cursor color")
        .set_default_color_palette(Some(Palette(std::array::from_fn(color))))
        .expect("set color palette");
    term.theme = Some(theme.clone());
}

/// Writes bytes after aborting any escape sequence a previous writer left unfinished.
fn write_from_ground(terminal: &mut GhosttyTerminal<'static, 'static>, bytes: &[u8]) {
    if !terminal.is_vt_ground().expect("read parser state") {
        // CAN cancels the sequence in progress.
        terminal.vt_write(b"\x18");
    }
    terminal.vt_write(bytes);
}

/// Matches libghostty's reply to a kitty keyboard protocol query (`CSI ? flags u`).
fn is_kitty_keyboard_reply(data: &[u8]) -> bool {
    data.strip_prefix(b"\x1b[?")
        .and_then(|rest| rest.strip_suffix(b"u"))
        .is_some_and(|flags| !flags.is_empty() && flags.iter().all(u8::is_ascii_digit))
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
    terminal.vt_write(RESET_CURSOR_STYLE);
    if let AlternateScroll::Off = alternate_scroll {
        terminal
            .set_default_mode(Mode::ALT_SCROLL, false)
            .expect("disable alternate scroll");
    }

    let sink = events.clone();
    terminal
        .on_pty_write(move |_, data| {
            // Zed encodes keys the legacy way, so programs must not think the
            // kitty keyboard protocol is available.
            if !is_kitty_keyboard_reply(data) {
                sink.send(TerminalBackendEvent::PtyWrite(
                    String::from_utf8_lossy(data).into_owned(),
                ))
            }
        })
        .expect("register pty write callback");
    let renderer = Rc::new(RefCell::new(Renderer::new()));
    let hold_renderer = renderer.clone();
    terminal
        .on_render_hold(move |terminal, held| {
            let mut renderer = hold_renderer.borrow_mut();
            let frame = held.then(|| renderer.frame(terminal));
            renderer.held = frame.map(|frame| HeldFrame {
                frame,
                since: Instant::now(),
            });
        })
        .expect("register render hold callback");
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
        renderer,
        config: config.clone(),
        theme: None,
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
///
/// libghostty keeps its parser state in the terminal, so a new parser first
/// aborts any escape sequence a previous one left unfinished.
pub(super) struct OutputParser {
    started: bool,
}

impl OutputParser {
    pub(super) fn new() -> Self {
        Self { started: false }
    }

    pub(super) fn advance(&mut self, term: &mut Term, bytes: &[u8]) {
        if std::mem::replace(&mut self.started, true) {
            term.terminal.vt_write(bytes);
        } else {
            write_from_ground(&mut term.terminal, bytes);
        }
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
            let hold_time_left = self.term.lock().renderer.borrow().hold_time_left();
            events.clear();
            if let Err(error) = self.poller.wait(&mut events, hold_time_left) {
                if error.kind() == ErrorKind::Interrupted {
                    continue;
                }
                log::error!("pty poll failed: {error}");
                break;
            }
            if expire_render_hold(&mut self.term.lock()) {
                self.events.send(TerminalBackendEvent::Wakeup);
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

/// Moves a point outside the grid onto its nearest cell.
fn clamp_point(term: &Term, point: Point) -> Point {
    Point::new(
        point
            .line
            .clamp(-(history_rows(term) as i32), screen_rows(term) as i32 - 1),
        point.column.min(screen_columns(term) - 1),
    )
}

/// Zed points count lines from the top of the active screen; scrollback is negative.
fn screen_point(term: &Term, point: Point) -> GhosttyPoint {
    let point = clamp_point(term, point);
    GhosttyPoint::Screen(PointCoordinate {
        x: point.column as u16,
        y: (point.line + history_rows(term) as i32) as u32,
    })
}

fn zed_point(term: &Term, coordinate: PointCoordinate) -> Point {
    Point::new(
        coordinate.y as i32 - history_rows(term) as i32,
        usize::from(coordinate.x),
    )
}

pub(super) fn display_offset(term: &Term) -> usize {
    viewport_offset(&term.terminal)
}

/// Rows between the bottom of the viewport and the bottom of the screen.
fn viewport_offset(terminal: &GhosttyTerminal<'static, 'static>) -> usize {
    let scrollbar = terminal.scrollbar().expect("read scrollbar");
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
    let cursor_line = cursor_line(term).max(0) as usize;
    let last_occupied_line = (cursor_line + 1..screen_rows(term))
        .rev()
        .find(|&line| row_has_text(term, line as u32))
        .unwrap_or(cursor_line);
    history_rows(term) + last_occupied_line + 1
}

/// Whether a row of the active screen has any text.
fn row_has_text(term: &Term, y: u32) -> bool {
    (0..screen_columns(term) as u16).any(|x| {
        term.terminal
            .grid_ref(GhosttyPoint::Active(PointCoordinate { x, y }))
            .and_then(|grid_ref| grid_ref.cell()?.has_text())
            .expect("read cell")
    })
}

fn is_alt_screen(term: &Term) -> bool {
    term.terminal.active_screen().expect("read active screen") == Screen::Alternate
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
    term.selection = selection.map(|selection| TermSelection {
        start: term
            .terminal
            .track_grid_ref(screen_point(term, selection.start.point))
            .expect("track selection start"),
        selection: selection.clone(),
        screen: term.terminal.active_screen().expect("read active screen"),
    });
    install_selection(term);
}

pub(super) fn update_selection(term: &mut Term, point: Point, side: SelectionSide) -> bool {
    let Some(tracked) = term.selection.as_mut() else {
        return false;
    };
    tracked.selection.update(point, side);
    install_selection(term);
    true
}

fn install_selection(term: &mut Term) {
    let history = history_rows(term) as i32;
    let screen = term.terminal.active_screen().expect("read active screen");
    if term
        .selection
        .as_ref()
        .is_some_and(|tracked| tracked.screen != screen)
    {
        term.selection = None;
    }
    if let Some(tracked) = &mut term.selection {
        match tracked
            .start
            .point(PointSpace::Screen)
            .expect("resolve selection start")
        {
            Some(start) => {
                tracked.selection.start.point =
                    Point::new(start.y as i32 - history, usize::from(start.x))
            }
            // The start's row left the scrollback.
            None => term.selection = None,
        }
    }
    let selection = term
        .selection
        .as_ref()
        .and_then(|tracked| ghostty_selection(term, &tracked.selection));
    term.terminal
        .set_selection(selection.as_ref())
        .expect("set selection");
}

/// The cells a selection covers, by alacritty's rules.
fn ghostty_selection<'t>(term: &'t Term, selection: &Selection) -> Option<GhosttySelection<'t>> {
    let clamp = |anchor: SelectionAnchor| SelectionAnchor {
        point: clamp_point(term, anchor.point),
        side: anchor.side,
    };
    let (mut start, mut end) = (clamp(selection.start), clamp(selection.end));
    if start.point > end.point {
        std::mem::swap(&mut start, &mut end);
    }
    let (first, last) = match selection.ty {
        SelectionType::Simple => simple_range(screen_columns(term), start, end)?,
        SelectionType::Semantic => (word_start(term, start.point), word_end(term, end.point)),
        SelectionType::Lines => (line_start(term, start.point), line_end(term, end.point)),
    };
    let grid_ref = |point| {
        term.terminal
            .grid_ref(screen_point(term, point))
            .expect("resolve selection point")
    };
    Some(GhosttySelection::new(
        grid_ref(first),
        grid_ref(last),
        false,
    ))
}

/// A side says which half of a cell the pointer was on; a cell is selected
/// once the pointer crosses its middle.
fn simple_range(
    columns: usize,
    mut start: SelectionAnchor,
    mut end: SelectionAnchor,
) -> Option<(Point, Point)> {
    let is_empty = (start.point == end.point && start.side == end.side)
        || (start.side == SelectionSide::Right
            && end.side == SelectionSide::Left
            && start.point.line == end.point.line
            && start.point.column + 1 == end.point.column);
    if is_empty {
        return None;
    }
    if end.side == SelectionSide::Left && start.point != end.point {
        end.point = if end.point.column == 0 {
            Point::new(end.point.line - 1, columns - 1)
        } else {
            Point::new(end.point.line, end.point.column - 1)
        };
    }
    if start.side == SelectionSide::Right && start.point != end.point {
        start.point = if start.point.column + 1 == columns {
            Point::new(start.point.line + 1, 0)
        } else {
            Point::new(start.point.line, start.point.column + 1)
        };
    }
    (start.point <= end.point).then_some((start.point, end.point))
}

fn previous_point(term: &Term, point: Point) -> Option<Point> {
    if point.column > 0 {
        Some(Point::new(point.line, point.column - 1))
    } else {
        (point.line > -(history_rows(term) as i32))
            .then(|| Point::new(point.line - 1, screen_columns(term) - 1))
    }
}

fn next_point(term: &Term, point: Point) -> Option<Point> {
    if point.column + 1 < screen_columns(term) {
        Some(Point::new(point.line, point.column + 1))
    } else {
        (point.line + 1 < screen_rows(term) as i32).then(|| Point::new(point.line + 1, 0))
    }
}

/// Whether the line continues on the next one.
fn line_wraps(term: &Term, line: i32) -> bool {
    row_is_wrapped(term, (line + history_rows(term) as i32) as u32)
}

/// The character in a cell, or `None` for the spacer half of a wide character.
fn cell_char(term: &Term, point: Point) -> Option<char> {
    let cell = term
        .terminal
        .grid_ref(screen_point(term, point))
        .and_then(|grid_ref| grid_ref.cell())
        .expect("read cell");
    if matches!(
        cell.wide().expect("read cell width"),
        CellWide::SpacerTail | CellWide::SpacerHead
    ) {
        return None;
    }
    let codepoint = cell.codepoint().expect("read cell codepoint");
    Some(
        char::from_u32(codepoint)
            .filter(|&c| c != '\0')
            .unwrap_or(' '),
    )
}

fn is_word_boundary(term: &Term, point: Point) -> bool {
    cell_char(term, point).is_some_and(|c| WORD_BOUNDARY_CHARS.contains(&c))
}

/// The first cell of the word around `point`, searching across wrapped lines.
fn word_start(term: &Term, point: Point) -> Point {
    let last_column = screen_columns(term) - 1;
    let mut current = point;
    while let Some(previous) = previous_point(term, current) {
        if previous.column == last_column && !line_wraps(term, previous.line) {
            break;
        }
        current = previous;
        if is_word_boundary(term, previous) {
            let mut after = next_point(term, previous);
            while let Some(point) = after
                && cell_char(term, point).is_none()
            {
                after = next_point(term, point);
            }
            return after.unwrap_or(previous);
        }
    }
    current
}

/// The last cell of the word around `point`, searching across wrapped lines.
fn word_end(term: &Term, point: Point) -> Point {
    let last_column = screen_columns(term) - 1;
    if point.column == last_column && !line_wraps(term, point.line) {
        return point;
    }
    let mut current = point;
    while let Some(next) = next_point(term, current) {
        current = next;
        if is_word_boundary(term, next) {
            return previous_point(term, next).unwrap_or(next);
        }
        if next.column == last_column && !line_wraps(term, next.line) {
            break;
        }
    }
    current
}

fn line_start(term: &Term, mut point: Point) -> Point {
    let top = -(history_rows(term) as i32);
    while point.line > top && line_wraps(term, point.line - 1) {
        point.line -= 1;
    }
    Point::new(point.line, 0)
}

fn line_end(term: &Term, mut point: Point) -> Point {
    while point.line + 1 < screen_rows(term) as i32 && line_wraps(term, point.line) {
        point.line += 1;
    }
    Point::new(point.line, screen_columns(term) - 1)
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

/// Ends a render hold the program kept past the timeout. Returns whether it did.
fn expire_render_hold(term: &mut Term) -> bool {
    let mut renderer = term.renderer.borrow_mut();
    if renderer.held.is_none() || renderer.hold_time_left().is_some() {
        return false;
    }
    renderer.held = None;
    drop(renderer);
    // Lets the program's next hold start a new one.
    term.terminal
        .set_mode(Mode::SYNC_OUTPUT, false)
        .expect("end synchronized output");
    true
}

impl Renderer {
    fn new() -> Self {
        Self {
            state: RenderState::new().expect("create render state"),
            rows: RowIterator::new().expect("create row iterator"),
            cells: CellIterator::new().expect("create cell iterator"),
            row_cells: Vec::new(),
            held: None,
        }
    }

    /// The frame to show: the held one during a render hold, otherwise the live screen.
    fn display_frame(&mut self, terminal: &GhosttyTerminal<'static, 'static>) -> Frame {
        if self.hold_time_left().is_none() {
            self.held = None;
        }
        match &self.held {
            Some(held) => held.frame.clone(),
            None => self.frame(terminal),
        }
    }

    /// Time left before the current render hold is ignored; `None` without one or once it has expired.
    fn hold_time_left(&self) -> Option<Duration> {
        RENDER_HOLD_TIMEOUT.checked_sub(self.held.as_ref()?.since.elapsed())
    }

    /// Copies the visible screen out of the terminal.
    fn frame(&mut self, terminal: &GhosttyTerminal<'static, 'static>) -> Frame {
        let display_offset = viewport_offset(terminal);
        let Self {
            state,
            rows,
            cells,
            row_cells,
            ..
        } = self;
        let snapshot = state.update(terminal).expect("update render state");
        let all_dirty = snapshot.dirty().expect("read dirty state") == Dirty::Full;
        let cursor_state = snapshot.cursor().expect("read cursor");
        {
            let mut row_iteration = rows.update(&snapshot).expect("iterate rows");
            let mut viewport_row = 0;
            while let Some(row) = row_iteration.next() {
                let id = row.id().expect("read row id");
                let cached = !all_dirty
                    && !row.dirty().expect("read row dirty state")
                    && row_cells
                        .get(viewport_row)
                        .is_some_and(|(cached_id, _)| *cached_id == id);
                if !cached {
                    let row_data = (id, read_row(terminal, cells, row, viewport_row as u32));
                    match row_cells.get_mut(viewport_row) {
                        Some(slot) => *slot = row_data,
                        None => row_cells.push(row_data),
                    }
                }
                viewport_row += 1;
            }
            row_cells.truncate(viewport_row);
        }
        snapshot.clean().expect("mark render state clean");

        let cells = row_cells
            .iter()
            .enumerate()
            .flat_map(|(row, (_, cells))| {
                let line = row as i32 - display_offset as i32;
                cells
                    .iter()
                    .enumerate()
                    .map(move |(column, cell)| IndexedCell {
                        point: Point::new(line, column),
                        cell: Cell { cell: cell.clone() },
                    })
            })
            .collect();

        let shape = if !cursor_state.visible {
            CursorShape::Hidden
        } else {
            match cursor_state.visual_style {
                CursorVisualStyle::Bar => CursorShape::Bar,
                CursorVisualStyle::Underline => CursorShape::Underline,
                CursorVisualStyle::BlockHollow => CursorShape::HollowBlock,
                _ => CursorShape::Block,
            }
        };
        let mut point = Point::new(
            i32::from(terminal.cursor_y().expect("read cursor row")),
            usize::from(terminal.cursor_x().expect("read cursor column")),
        );
        // A cursor on the right half of a wide character is drawn over all of it.
        if cursor_state
            .viewport
            .is_some_and(|viewport| viewport.at_wide_tail)
        {
            point.column -= 1;
        }
        Frame {
            cells,
            display_offset,
            cursor: Cursor { shape, point },
            cursor_blinking: cursor_state.blinking,
        }
    }
}

/// Copies one row out of libghostty's render state.
fn read_row(
    terminal: &GhosttyTerminal<'static, 'static>,
    cells: &mut CellIterator<'static>,
    row: &RowIteration<'static, '_>,
    viewport_row: u32,
) -> Vec<CellData> {
    let mut output = Vec::new();
    let mut graphemes = Vec::new();
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
            raw.wide().expect("read cell width") == CellWide::SpacerTail,
        );
        flags.set(CellFlags::DIM, style.faint);
        flags.set(CellFlags::UNDERLINE, style.underline != Underline::None);
        flags.set(CellFlags::UNDERCURL, style.underline == Underline::Curly);
        flags.set(CellFlags::STRIKEOUT, style.strikethrough);
        flags.set(CellFlags::BOLD, style.bold);
        flags.set(CellFlags::ITALIC, style.italic);

        if raw.has_hyperlink().expect("read cell hyperlink") {
            let grid_ref = terminal
                .grid_ref(GhosttyPoint::Viewport(PointCoordinate {
                    x: column,
                    y: viewport_row,
                }))
                .expect("resolve visible cell");
            data.hyperlink =
                hyperlink_uri(&grid_ref).map(|uri| Hyperlink::new(None::<String>, uri));
        }

        output.push(data);
        column += 1;
    }
    output
}

pub(super) fn make_content(term: &Term, last_content: &Content) -> Content {
    let Frame {
        cells,
        display_offset,
        cursor,
        cursor_blinking,
    } = term.renderer.borrow_mut().display_frame(&term.terminal);
    if cursor_blinking != term.cursor_blinking.replace(cursor_blinking) {
        term.events.send(TerminalBackendEvent::CursorBlinkingChange);
    }
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
        cells: Box::new(
            term.renderer
                .borrow_mut()
                .display_frame(&term.terminal)
                .cells
                .into_iter(),
        ),
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
    write_from_ground(&mut term.terminal, sequence.as_bytes());
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

const URL_REGEX: &str = r#"(ipfs:|ipns:|magnet:|mailto:|gemini://|gopher://|https://|http://|news:|file://|git://|ssh:|ftp://|zed://)[^\u{0000}-\u{001F}\u{007F}-\u{009F}<>"\s{-}\^⟨⟩`']+"#;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HyperlinkMatch {
    pub(crate) text: String,
    pub(crate) is_url: bool,
    pub(crate) range: Range,
}

pub(crate) struct RegexSearches {
    url_regex: regex::Regex,
    path_hyperlink_regexes: Vec<regex::Regex>,
    path_hyperlink_timeout: Duration,
}

impl Default for RegexSearches {
    fn default() -> Self {
        Self::new(Vec::<String>::new(), Duration::ZERO)
    }
}

impl RegexSearches {
    pub(crate) fn new(
        path_hyperlink_regexes: impl IntoIterator<Item: AsRef<str>>,
        path_hyperlink_timeout: Duration,
    ) -> Self {
        Self {
            url_regex: regex::Regex::new(URL_REGEX).expect("URL regex is valid"),
            path_hyperlink_regexes: path_hyperlink_regexes
                .into_iter()
                .filter_map(|regex| {
                    regex::Regex::new(regex.as_ref())
                        .inspect_err(|error| {
                            log::warn!(
                                "Ignoring path hyperlink regex specified in \
                                 `terminal.path_hyperlink_regexes`:\n\n\t{}\n\nError: {error}",
                                regex.as_ref(),
                            )
                        })
                        .ok()
                })
                .collect(),
            path_hyperlink_timeout,
        }
    }
}

/// The text of a wrapped line, with the grid position of each character.
struct LineText {
    text: String,
    /// Byte offset of each character in `text`, with its first and last cell.
    chars: Vec<(usize, Point, Point)>,
}

impl LineText {
    fn char_index_at(&self, point: Point) -> Option<usize> {
        self.chars
            .iter()
            .position(|(_, start, end)| *start <= point && point <= *end)
    }

    fn char_index_of_byte(&self, offset: usize) -> usize {
        self.chars.partition_point(|(byte, _, _)| *byte < offset)
    }

    /// Grid range of the characters in `bytes`, which must be non-empty.
    fn range(&self, bytes: StdRange<usize>) -> Range {
        let first = self.char_index_of_byte(bytes.start);
        let last = self.char_index_of_byte(bytes.end) - 1;
        Range::new(self.chars[first].1, self.chars[last].2)
    }
}

fn row_is_wrapped(term: &Term, y: u32) -> bool {
    term.terminal
        .grid_ref(GhosttyPoint::Screen(PointCoordinate { x: 0, y }))
        .and_then(|grid_ref| grid_ref.row()?.is_wrapped())
        .expect("read row")
}

/// The wrapped line containing `point`, read cell by cell.
fn line_text(term: &Term, point: Point) -> LineText {
    let GhosttyPoint::Screen(coordinate) = screen_point(term, point) else {
        unreachable!()
    };
    let last_row = (history_rows(term) + screen_rows(term) - 1) as u32;
    let mut first = coordinate.y;
    while first > 0 && row_is_wrapped(term, first - 1) {
        first -= 1;
    }
    let mut last = coordinate.y;
    while last < last_row && row_is_wrapped(term, last) {
        last += 1;
    }

    let columns = screen_columns(term) as u16;
    let mut line = LineText {
        text: String::new(),
        chars: Vec::new(),
    };
    let mut graphemes = ['\0'; 16];
    for y in first..=last {
        for x in 0..columns {
            let grid_ref = term
                .terminal
                .grid_ref(GhosttyPoint::Screen(PointCoordinate { x, y }))
                .expect("resolve cell");
            let cell = grid_ref.cell().expect("read cell");
            let wide = cell.wide().expect("read cell width");
            if matches!(wide, CellWide::SpacerTail | CellWide::SpacerHead) {
                continue;
            }
            let start = zed_point(term, PointCoordinate { x, y });
            let mut end = start;
            if wide == CellWide::Wide {
                end.column += 1;
            }
            line.chars.push((line.text.len(), start, end));
            let len = grid_ref.graphemes(&mut graphemes).unwrap_or(0);
            match graphemes[..len].first() {
                None | Some('\t') => line.text.push(' '),
                Some(_) => line.text.extend(&graphemes[..len]),
            }
        }
    }
    line
}

fn hyperlink_uri(grid_ref: &GridRef<'_>) -> Option<String> {
    if !grid_ref
        .cell()
        .and_then(|cell| cell.has_hyperlink())
        .expect("read cell hyperlink")
    {
        return None;
    }
    let mut uri = vec![0; 2048];
    let len = loop {
        match grid_ref.hyperlink_uri(&mut uri) {
            Ok(len) => break len,
            Err(libghostty_vt::Error::OutOfSpace { required }) => uri.resize(required, 0),
            Err(error) => panic!("read hyperlink uri: {error}"),
        }
    };
    (len > 0).then(|| String::from_utf8_lossy(&uri[..len]).into_owned())
}

/// The OSC 8 link at `point`, spanning the neighboring cells that carry the same link.
fn osc8_hyperlink(term: &Term, point: Point) -> Option<(String, Range)> {
    let uri_at = |point| {
        let grid_ref = term.terminal.grid_ref(screen_point(term, point)).ok()?;
        hyperlink_uri(&grid_ref)
    };
    let uri = uri_at(point)?;
    let line = line_text(term, point);
    let index = line.char_index_at(point)?;
    let same_link = |&(_, start, _): &(usize, Point, Point)| uri_at(start).as_ref() == Some(&uri);
    let first = line.chars[..index]
        .iter()
        .rposition(|char| !same_link(char))
        .map_or(0, |position| position + 1);
    let last = line.chars[index..]
        .iter()
        .position(|char| !same_link(char))
        .map_or(line.chars.len(), |position| index + position)
        - 1;
    Some((uri, Range::new(line.chars[first].1, line.chars[last].2)))
}

/// Drops trailing punctuation that usually ends the sentence rather than the URL.
fn trim_url_punctuation(url: &str) -> &str {
    let (open, mut close) = url.chars().fold((0, 0), |(open, close), c| match c {
        '(' => (open + 1, close),
        ')' => (open, close + 1),
        _ => (open, close),
    });
    let mut url = url;
    while let Some(last) = url.chars().last() {
        let trim = match last {
            '.' | ',' | ':' | ';' | '(' => true,
            ')' if close > open => {
                close -= 1;
                true
            }
            _ => false,
        };
        if !trim {
            break;
        }
        url = &url[..url.len() - last.len_utf8()];
    }
    url
}

/// Byte offset just past the first unbalanced `(`, if any.
fn first_unbalanced_open_paren(s: &str) -> Option<usize> {
    let mut balance: i32 = 0;
    let mut first_unmatched = None;
    for (i, c) in s.char_indices() {
        match c {
            '(' => {
                if balance == 0 {
                    first_unmatched = Some(i + c.len_utf8());
                }
                balance += 1;
            }
            ')' => {
                balance -= 1;
                if balance <= 0 {
                    balance = 0;
                    first_unmatched = None;
                }
            }
            _ => {}
        }
    }
    first_unmatched.filter(|_| balance > 0)
}

fn url_match(line: &LineText, hovered: usize, regex: &regex::Regex) -> Option<(String, Range)> {
    let found = regex
        .find_iter(&line.text)
        .find(|found| found.range().contains(&hovered))?;
    let url = trim_url_punctuation(found.as_str());
    if url.is_empty() || found.start() + url.len() <= hovered {
        return None;
    }
    Some((
        url.to_string(),
        line.range(found.start()..found.start() + url.len()),
    ))
}

fn path_match(
    line: &LineText,
    hovered: usize,
    regexes: &[regex::Regex],
    timeout: Duration,
) -> Option<(String, Range)> {
    if regexes.is_empty() || timeout.is_zero() {
        return None;
    }
    let started = Instant::now();
    let text = line.text.trim_ascii_end();
    if text.len() <= hovered {
        return None;
    }
    for regex in regexes {
        let mut path_found = false;
        for captures in regex.captures_iter(text) {
            path_found = true;
            let whole = captures
                .get(0)
                .expect("capture 0 is the whole match")
                .range();
            let parse = |name: &str| {
                captures
                    .name(name)
                    .and_then(|capture| capture.as_str().parse::<u32>().ok())
            };
            let (mut path_range, position) = match captures.name("path") {
                Some(path) => (
                    path.range(),
                    parse("line").map(|line| (line, parse("column"))),
                ),
                None => (whole.clone(), None),
            };
            let mut link_range = captures.name("link").map_or(whole, |link| link.range());
            if let Some(trim) = first_unbalanced_open_paren(&text[path_range.clone()]) {
                path_range.start += trim;
                link_range.start = link_range.start.max(path_range.start);
            }
            if link_range.is_empty() || !link_range.contains(&hovered) {
                continue;
            }
            let mut path = text[path_range].to_string();
            if let Some((line_number, column)) = position {
                path += &format!(":{line_number}");
                if let Some(column) = column {
                    path += &format!(":{column}");
                }
            }
            return Some((path, line.range(link_range)));
        }
        if path_found {
            return None;
        }
        if started.elapsed() > timeout {
            log::warn!(
                "Timed out processing path hyperlink regexes after {}ms",
                started.elapsed().as_millis()
            );
            return None;
        }
    }
    None
}

/// Treats `file://` links as paths, so line numbers at their end are handled like a path's.
fn normalize_hyperlink_match(
    text: String,
    is_url: bool,
    range: Range,
    path_style: PathStyle,
) -> HyperlinkMatch {
    if !is_url || !text.starts_with("file://") {
        return HyperlinkMatch {
            text,
            is_url,
            range,
        };
    }
    let path = url::Url::parse(&text)
        .ok()
        .and_then(|url| url.to_file_path_ext(path_style).ok())
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| text.strip_prefix("file://").unwrap_or(&text).to_string());
    HyperlinkMatch {
        text: path,
        is_url: false,
        range,
    }
}

/// Finds the OSC 8 link, URL or path under `point`.
pub(super) fn find_from_terminal_point(
    term: &Term,
    point: Point,
    regex_searches: &mut RegexSearches,
    path_style: PathStyle,
) -> Option<HyperlinkMatch> {
    if let Some((uri, range)) = osc8_hyperlink(term, point) {
        return Some(normalize_hyperlink_match(uri, true, range, path_style));
    }
    let line = line_text(term, point);
    let hovered = line.chars[line.char_index_at(point)?].0;
    if let Some((url, range)) = url_match(&line, hovered, &regex_searches.url_regex) {
        return Some(normalize_hyperlink_match(url, true, range, path_style));
    }
    let (path, range) = path_match(
        &line,
        hovered,
        &regex_searches.path_hyperlink_regexes,
        regex_searches.path_hyperlink_timeout,
    )?;
    Some(normalize_hyperlink_match(path, false, range, path_style))
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

#[cfg(test)]
mod tests {
    use futures::channel::mpsc::{UnboundedReceiver, unbounded};
    use gpui::{bounds, point, px, size};

    use super::*;

    fn test_term(
        cursor_shape: SettingsCursorShape,
    ) -> (Arc<TermLock>, UnboundedReceiver<PtyEvent>) {
        let bounds = TerminalBounds::new(
            px(10.),
            px(10.),
            bounds(point(px(0.), px(0.)), size(px(100.), px(30.))),
        );
        let (events_tx, events_rx) = unbounded();
        let term = new_term(
            &pty_term_config(100, cursor_shape),
            bounds,
            events_tx,
            AlternateScroll::On,
            0,
        );
        (term, events_rx)
    }

    fn line_text(frame: &Frame, line: i32) -> String {
        frame
            .cells
            .iter()
            .filter(|cell| cell.point.line == line)
            .map(|cell| cell.character())
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    fn live_frame(term: &Term) -> Frame {
        term.renderer.borrow_mut().display_frame(&term.terminal)
    }

    fn select(term: &mut Term, ty: SelectionType, start: Point, end: Point) {
        let mut selection = Selection::new(ty, start, SelectionSide::Left);
        selection.update(end, SelectionSide::Right);
        set_selection(term, Some(&selection));
    }

    #[test]
    fn starts_with_the_configured_cursor_shape() {
        let (term, _events) = test_term(SettingsCursorShape::Bar);
        assert_eq!(live_frame(&term.lock()).cursor.shape, CursorShape::Bar);
    }

    #[test]
    fn keeps_the_programs_cursor_style_when_the_default_changes() {
        let (term, _events) = test_term(SettingsCursorShape::Block);
        term.lock().terminal.vt_write(b"\x1b[6 q");
        apply_config(&term, &pty_term_config(100, SettingsCursorShape::Hollow));
        assert_eq!(live_frame(&term.lock()).cursor.shape, CursorShape::Bar);
    }

    #[test]
    fn does_not_answer_kitty_keyboard_queries() {
        let (term, mut events) = test_term(SettingsCursorShape::Block);
        term.lock().terminal.vt_write(b"\x1b[?u\x1b[c");
        let mut replies = Vec::new();
        while let Ok(PtyEvent::Event(TerminalBackendEvent::PtyWrite(reply), _)) = events.try_recv()
        {
            replies.push(reply);
        }
        assert_eq!(replies, ["\x1b[?62;1;6;22c"]);
    }

    #[test]
    fn shows_the_frame_from_the_start_of_a_render_hold() {
        let (term, _events) = test_term(SettingsCursorShape::Block);
        let mut term = term.lock();
        term.terminal.vt_write(b"before\x1b[?2026h\rafter ");
        assert_eq!(line_text(&live_frame(&term), 0), "before");
        term.terminal.vt_write(b"\x1b[?2026l");
        assert_eq!(line_text(&live_frame(&term), 0), "after");
    }

    #[test]
    fn simple_selection_edges_wrap_to_neighboring_lines() {
        let anchor = |line, column, side| SelectionAnchor {
            point: Point::new(line, column),
            side,
        };
        // Ending on the left half of the first column ends on the line above.
        assert_eq!(
            simple_range(
                10,
                anchor(0, 2, SelectionSide::Left),
                anchor(1, 0, SelectionSide::Left)
            ),
            Some((Point::new(0, 2), Point::new(0, 9)))
        );
        // Starting on the right half of the last column starts on the line below.
        assert_eq!(
            simple_range(
                10,
                anchor(0, 9, SelectionSide::Right),
                anchor(1, 3, SelectionSide::Right)
            ),
            Some((Point::new(1, 0), Point::new(1, 3)))
        );
        assert_eq!(
            simple_range(
                10,
                anchor(0, 2, SelectionSide::Right),
                anchor(0, 3, SelectionSide::Left)
            ),
            None
        );
    }

    #[test]
    fn semantic_and_line_selections_cover_blank_cells() {
        let (term, _events) = test_term(SettingsCursorShape::Block);
        let mut term = term.lock();
        term.terminal.vt_write(b"foo bar");
        select(
            &mut term,
            SelectionType::Semantic,
            Point::new(0, 1),
            Point::new(0, 1),
        );
        assert_eq!(selection_text(&term).as_deref(), Some("foo"));

        // A blank cell is a word boundary of its own.
        select(
            &mut term,
            SelectionType::Semantic,
            Point::new(0, 8),
            Point::new(0, 8),
        );
        assert_eq!(
            selection_range(&term).map(|range| (range.start, range.end)),
            Some((Point::new(0, 8), Point::new(0, 8)))
        );

        select(
            &mut term,
            SelectionType::Lines,
            Point::new(2, 0),
            Point::new(2, 0),
        );
        assert_eq!(
            selection_range(&term).map(|range| (range.start, range.end)),
            Some((Point::new(2, 0), Point::new(2, 9)))
        );
    }

    #[test]
    fn selection_start_follows_scrolled_output() {
        let (term, _events) = test_term(SettingsCursorShape::Block);
        let mut term = term.lock();
        term.terminal.vt_write(b"one\r\ntwo\r\nthree");
        select(
            &mut term,
            SelectionType::Simple,
            Point::new(1, 0),
            Point::new(1, 2),
        );
        term.terminal.vt_write(b"\r\nfour");
        assert!(update_selection(
            &mut term,
            Point::new(0, 2),
            SelectionSide::Right
        ));
        assert_eq!(selection_text(&term).as_deref(), Some("two"));
    }

    #[test]
    fn switching_screens_clears_the_selection() {
        let (term, _events) = test_term(SettingsCursorShape::Block);
        let mut term = term.lock();
        term.terminal.vt_write(b"one");
        select(
            &mut term,
            SelectionType::Simple,
            Point::new(0, 0),
            Point::new(0, 2),
        );
        term.terminal.vt_write(b"\x1b[?1049h");
        update_selection(&mut term, Point::new(0, 1), SelectionSide::Right);
        assert_eq!(selection_text(&term), None);
    }
}
