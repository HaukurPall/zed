use gpui::{Context, Task};
use parking_lot::{MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard};
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

#[cfg(target_os = "windows")]
use windows::Win32::{Foundation::HANDLE, System::Threading::GetProcessId};

use sysinfo::{Pid, Process, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

use crate::{Event, Terminal};

#[derive(Clone, Copy)]
pub struct ProcessIdGetter {
    handle: i32,
    fallback_pid: u32,
}

impl ProcessIdGetter {
    pub(crate) fn new(handle: i32, fallback_pid: u32) -> ProcessIdGetter {
        ProcessIdGetter {
            handle,
            fallback_pid,
        }
    }

    pub fn fallback_pid(&self) -> Pid {
        Pid::from_u32(self.fallback_pid)
    }
}

#[cfg(unix)]
impl ProcessIdGetter {
    fn pid(&self) -> Option<Pid> {
        // Negative pid means error.
        // Zero pid means no foreground process group is set on the PTY yet.
        // Avoid killing the current process by returning a zero pid.
        let pid = unsafe { libc::tcgetpgrp(self.handle) };
        if pid > 0 {
            return Some(Pid::from_u32(pid as u32));
        }

        if self.fallback_pid > 0 {
            return Some(Pid::from_u32(self.fallback_pid));
        }

        None
    }
}

#[cfg(windows)]
impl ProcessIdGetter {
    fn pid(&self) -> Option<Pid> {
        let pid = unsafe { GetProcessId(HANDLE(self.handle as _)) };
        // the GetProcessId may fail and returns zero, which will lead to a stack overflow issue
        if pid == 0 {
            // in the builder process, there is a small chance, almost negligible,
            // that this value could be zero, which means child_watcher returns None,
            // GetProcessId returns 0.
            if self.fallback_pid == 0 {
                return None;
            }
            return Some(Pid::from_u32(self.fallback_pid));
        }
        Some(Pid::from_u32(pid))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ProcessInfo {
    pub(crate) name: String,
    pub(crate) cwd: PathBuf,
    pub(crate) argv: Vec<String>,
}

/// Fetches Zed-relevant Pseudo-Terminal (PTY) process information
pub(crate) struct PtyProcessInfo {
    system: RwLock<System>,
    refresh_kind: ProcessRefreshKind,
    pid_getter: ProcessIdGetter,
    /// Set when the spawned child is `login`, which runs the shell as its own
    /// child.
    login_shell: Option<LoginShell>,
    last_foreground_pid: Mutex<Option<Pid>>,
    pub(crate) current: RwLock<Option<ProcessInfo>>,
    task: Mutex<Option<Task<()>>>,
}

#[derive(Default)]
struct LoginShell {
    /// Known once the shell has been seen in the foreground.
    shell: OnceLock<Pid>,
    /// The last foreground process group whose leader isn't `login`'s child.
    /// A process never becomes `login`'s child later, so it isn't looked up
    /// again.
    not_shell: Mutex<Option<Pid>>,
}

impl PtyProcessInfo {
    pub(crate) fn new(pid_getter: ProcessIdGetter, shell_under_login: bool) -> PtyProcessInfo {
        // Task enumeration is on by default and would retain a `Process` entry
        // per thread, each pinning an open `/proc/<pid>/task/<tid>/stat` handle
        // on Linux (#58651).
        let process_refresh_kind = ProcessRefreshKind::nothing()
            .with_cmd(UpdateKind::Always)
            .with_cwd(UpdateKind::Always)
            .with_exe(UpdateKind::Always)
            .without_tasks();
        // `System::new_with_specifics` with a process refresh kind would
        // snapshot every process on the machine into this terminal's `System`,
        // retaining one open procfs handle per process for the lifetime of the
        // terminal (#58651). Refresh only the spawned child so that
        // `kill_child_process` works before the first foreground refresh.
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid_getter.fallback_pid()]),
            true,
            process_refresh_kind,
        );

        PtyProcessInfo {
            system: RwLock::new(system),
            refresh_kind: process_refresh_kind,
            pid_getter,
            login_shell: shell_under_login.then(LoginShell::default),
            last_foreground_pid: Mutex::new(None),
            current: RwLock::new(None),
            task: Mutex::new(None),
        }
    }

    pub(crate) fn pid_getter(&self) -> &ProcessIdGetter {
        &self.pid_getter
    }

    /// Whether `pid`, a foreground process group, is the shell the terminal
    /// started.
    pub(crate) fn is_shell(&self, pid: Pid) -> bool {
        let child = self.pid_getter.fallback_pid();
        if pid == child {
            return true;
        }
        let Some(login_shell) = &self.login_shell else {
            return false;
        };
        if let Some(shell) = login_shell.shell.get() {
            return pid == *shell;
        }
        if *login_shell.not_shell.lock() == Some(pid) {
            return false;
        }
        // `login` runs a single child, the shell, which moves to its own
        // process group when it has job control.
        if parent_pid(pid) == Some(child) {
            return *login_shell.shell.get_or_init(|| pid) == pid;
        }
        *login_shell.not_shell.lock() = Some(pid);
        false
    }

    fn refresh(&self) -> Option<MappedRwLockReadGuard<'_, Process>> {
        let pid = self.pid_getter.pid()?;
        let fallback_pid = self.pid_getter.fallback_pid();
        let mut system = self.system.write();
        // sysinfo never evicts processes that are absent from the refreshed pid
        // set, so entries for former foreground processes (each pinning an open
        // `/proc/<pid>/stat` handle on Linux) would otherwise accumulate for as
        // long as this terminal lives (#58651). Rebuild the `System` whenever
        // the foreground process changes to keep the map bounded.
        if self.last_foreground_pid.lock().replace(pid) != Some(pid) {
            *system = System::new();
        }
        let pids = [pid, fallback_pid];
        let pids = if pid == fallback_pid {
            &pids[..1]
        } else {
            &pids[..]
        };
        system.refresh_processes_specifics(ProcessesToUpdate::Some(pids), true, self.refresh_kind);
        drop(system);
        RwLockReadGuard::try_map(self.system.read(), |system| system.process(pid)).ok()
    }

    fn get_child(&self) -> Option<MappedRwLockReadGuard<'_, Process>> {
        let pid = self.pid_getter.fallback_pid();
        RwLockReadGuard::try_map(self.system.read(), |system| system.process(pid)).ok()
    }

    #[cfg(unix)]
    pub(crate) fn kill_current_process(&self) -> bool {
        let Some(pid) = self.pid_getter.pid() else {
            return false;
        };
        unsafe { libc::killpg(pid.as_u32() as i32, libc::SIGKILL) == 0 }
    }

    #[cfg(not(unix))]
    pub(crate) fn kill_current_process(&self) -> bool {
        self.refresh().is_some_and(|process| process.kill())
    }

    pub(crate) fn kill_child_process(&self) -> bool {
        self.get_child().is_some_and(|process| process.kill())
    }

    #[cfg(unix)]
    pub(crate) fn terminate_child_process(&self) -> bool {
        let pid = self.pid_getter.fallback_pid();
        unsafe { libc::killpg(pid.as_u32() as i32, libc::SIGTERM) == 0 }
    }

    #[cfg(not(unix))]
    pub(crate) fn terminate_child_process(&self) -> bool {
        false
    }

    fn load(&self) -> Option<ProcessInfo> {
        let process = self.refresh()?;
        let cwd = process.cwd().map_or(PathBuf::new(), |p| p.to_owned());

        let info = ProcessInfo {
            name: process.name().to_str()?.to_owned(),
            cwd,
            argv: process
                .cmd()
                .iter()
                .filter_map(|s| s.to_str().map(ToOwned::to_owned))
                .collect(),
        };
        *self.current.write() = Some(info.clone());
        Some(info)
    }

    #[cfg(all(test, unix))]
    pub(crate) fn load_for_test(&self) -> Option<ProcessInfo> {
        self.load()
    }

    /// Updates the cached process info, emitting a [`Event::TitleChanged`] event if the Zed-relevant info has changed
    pub(crate) fn emit_title_changed_if_changed(self: &Arc<Self>, cx: &mut Context<'_, Terminal>) {
        if self.task.lock().is_some() {
            return;
        }
        let this = self.clone();
        let change_task = cx.background_executor().spawn(async move {
            let previous = this.current.read().clone();
            let current = this.load();
            let has_changed = match (previous.as_ref(), current.as_ref()) {
                (None, None) => false,
                (Some(prev), Some(now)) => prev.cwd != now.cwd || prev.name != now.name,
                _ => true,
            };
            if has_changed {
                *this.current.write() = current;
            }
            has_changed
        });
        let this = Arc::downgrade(self);
        *self.task.lock() = Some(cx.spawn(async move |term, cx| {
            if change_task.await {
                term.update(cx, |terminal, cx| {
                    terminal.record_working_directory();
                    cx.emit(Event::TitleChanged);
                })
                .ok();
            }
            if let Some(this) = this.upgrade() {
                this.task.lock().take();
            }
        }));
    }

    pub(crate) fn pid(&self) -> Option<Pid> {
        self.pid_getter.pid()
    }
}

fn parent_pid(pid: Pid) -> Option<Pid> {
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().without_tasks(),
    );
    system.process(pid)?.parent()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// Regression test for <https://github.com/zed-industries/zed/issues/58651>:
    /// on Linux, sysinfo keeps an open `/proc/<pid>/stat` handle for every
    /// `Process` entry retained in a `System`, and never evicts entries that are
    /// absent from the refreshed pid set. The per-terminal `System` must
    /// therefore not snapshot every process on the machine, nor accumulate an
    /// entry per foreground process that has ever run in this terminal.
    #[test]
    #[allow(
        clippy::disallowed_methods,
        reason = "the test needs real short-lived child processes and may block"
    )]
    fn process_map_stays_bounded() {
        let mut info = PtyProcessInfo::new(ProcessIdGetter::new(-1, std::process::id()), false);
        assert!(
            info.get_child().is_some(),
            "the spawned child must be inspectable for kill_child_process \
             before the first foreground refresh"
        );
        assert!(info.load_for_test().is_some());
        let initial_len = info.system.read().processes().len();
        assert!(
            initial_len <= 2,
            "creating a terminal retained {initial_len} process entries"
        );

        for _ in 0..3 {
            let mut child = std::process::Command::new("sleep")
                .arg("30")
                .spawn()
                .expect("failed to spawn child process");
            info.pid_getter = ProcessIdGetter::new(-1, child.id());
            assert!(info.load_for_test().is_some());
            child.kill().expect("failed to kill child process");
            child.wait().expect("failed to wait for child process");
        }

        let churned_len = info.system.read().processes().len();
        assert!(
            churned_len <= 2,
            "foreground process churn retained {churned_len} process entries"
        );
    }

    /// `sh` stands in for `login`: it waits for the shell it runs, and an
    /// interactive shell moves to its own process group.
    #[test]
    fn shell_under_login_is_the_child_of_login() {
        use alacritty_terminal::tty::EventedReadWrite as _;
        use std::io::Write as _;

        let options = crate::pty_options(
            Some((
                "sh".to_owned(),
                vec!["-c".to_owned(), "bash --norc --noprofile -i; :".to_owned()],
            )),
            None,
            [("HISTFILE".to_owned(), String::new())],
            None,
        );
        let mut pty =
            crate::open_pty(&options, crate::TerminalBounds::default(), 0).expect("open a PTY");
        let pid_getter = ProcessIdGetter::from(&pty);
        let login = pid_getter.fallback_pid();
        let info = PtyProcessInfo::new(pid_getter, true);

        let shell = wait_for_foreground(&pid_getter, |pid| pid != login);
        assert_eq!(parent_pid(shell), Some(login));
        assert!(info.is_shell(shell));
        assert!(!PtyProcessInfo::new(pid_getter, false).is_shell(shell));

        pty.writer()
            .write_all(b"sleep 30\n")
            .expect("write to the PTY");
        let command = wait_for_foreground(&pid_getter, |pid| pid != shell);
        assert!(!info.is_shell(command));

        // The shell's job isn't taken for the shell before the shell is known.
        let info = PtyProcessInfo::new(pid_getter, true);
        assert!(!info.is_shell(command));
        assert!(!info.is_shell(command));
        assert!(info.is_shell(shell));

        pty.writer().write_all(b"\x03").expect("write to the PTY");
        wait_for_foreground(&pid_getter, |pid| pid == shell);
        pty.writer()
            .write_all(b"bash --norc --noprofile -i\n")
            .expect("write to the PTY");
        let nested_shell = wait_for_foreground(&pid_getter, |pid| pid != shell);
        assert_eq!(parent_pid(nested_shell), Some(shell));
        assert!(!info.is_shell(nested_shell));
    }

    fn wait_for_foreground(pid_getter: &ProcessIdGetter, condition: impl Fn(Pid) -> bool) -> Pid {
        for _ in 0..500 {
            if let Some(pid) = pid_getter.pid()
                && condition(pid)
            {
                return pid;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the PTY's foreground process didn't change");
    }
}
