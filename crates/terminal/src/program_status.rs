//! What programs in a terminal report about their work: progress (OSC 9;4) and
//! program status (OSC 7501, <https://www.superlogical.com/rex/docs/build/program-status>).

/// Progress reported with OSC 9;4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalProgress {
    pub state: ProgressState,
    /// 0 through 100, or `None` when the program gave no percentage.
    pub percent: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressState {
    Normal,
    Indeterminate,
    Error,
    Paused,
}

/// Ordered from least to most in need of the user.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProgramState {
    Idle,
    Done,
    Working,
    Error,
    Blocked,
}

/// One OSC 7501 report. Text fields are empty when the program left them out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramStatusReport {
    /// Empty for the program itself; `/` separates a record from its parent.
    pub id: String,
    pub state: ProgramState,
    pub progress: Option<u8>,
    pub app: String,
    pub title: String,
    pub message: String,
}

impl ProgramStatusReport {
    /// A short name for the record: its title, else its app.
    pub fn label(&self) -> &str {
        if self.title.is_empty() {
            &self.app
        } else {
            &self.title
        }
    }
}

/// What a terminal shows about the work of the program running in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TerminalStatus {
    Program(ProgramStatusReport),
    Progress(TerminalProgress),
}

/// The spec asks for at least 64 records and at most 256.
const MAX_RECORDS: usize = 256;

/// The records a terminal keeps, one per id, as the spec describes.
#[derive(Debug, Default)]
pub struct ProgramStatusRecords {
    /// Each report with the update count when it arrived.
    records: Vec<(ProgramStatusReport, u64)>,
    updates: u64,
}

impl ProgramStatusRecords {
    /// Stores a report in place of its record. Returns the record when it has
    /// just entered a state that needs the user: blocked, done or error.
    pub fn apply(&mut self, report: ProgramStatusReport) -> Option<ProgramStatusReport> {
        let previous = self
            .records
            .iter()
            .position(|(record, _)| record.id == report.id)
            .map(|index| self.records.swap_remove(index).0.state);
        if self.records.len() == MAX_RECORDS
            && let Some(oldest) = self
                .records
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, updated))| *updated)
                .map(|(index, _)| index)
        {
            self.records.swap_remove(oldest);
        }
        let needs_user = matches!(
            report.state,
            ProgramState::Blocked | ProgramState::Done | ProgramState::Error
        ) && previous != Some(report.state);
        self.updates += 1;
        self.records.push((report.clone(), self.updates));
        needs_user.then(|| self.with_inherited_app(report))
    }

    /// Removes the record with this id and every record beneath it; an empty id removes all.
    pub fn clear(&mut self, id: &str) {
        self.records
            .retain(|(record, _)| !is_at_or_beneath(&record.id, id));
    }

    /// Whether any record is about work that ends with the program: idle, working or blocked.
    pub fn has_running(&self) -> bool {
        self.records
            .iter()
            .any(|(record, _)| Self::is_running(record.state))
    }

    /// Removes the records of work that ends with the program. Returns whether
    /// anything was removed.
    pub fn clear_running(&mut self) -> bool {
        let count = self.records.len();
        self.records
            .retain(|(record, _)| !Self::is_running(record.state));
        self.records.len() != count
    }

    /// Removes done and error records, for when the user has seen them.
    /// Returns whether anything was removed.
    pub fn clear_finished(&mut self) -> bool {
        let count = self.records.len();
        self.records.retain(|(record, _)| {
            !matches!(record.state, ProgramState::Done | ProgramState::Error)
        });
        self.records.len() != count
    }

    fn is_running(state: ProgramState) -> bool {
        matches!(
            state,
            ProgramState::Idle | ProgramState::Working | ProgramState::Blocked
        )
    }

    /// The record that most needs the user, the newest among equals.
    pub fn most_urgent(&self) -> Option<ProgramStatusReport> {
        let (record, _) = self
            .records
            .iter()
            .max_by_key(|(record, updated)| (record.state, *updated))?;
        Some(self.with_inherited_app(record.clone()))
    }

    /// The record with the app of its nearest ancestor when it has none.
    fn with_inherited_app(&self, mut record: ProgramStatusReport) -> ProgramStatusReport {
        if record.app.is_empty() {
            record.app = ancestors(&record.id)
                .find_map(|id| {
                    self.records
                        .iter()
                        .find(|(ancestor, _)| ancestor.id == id && !ancestor.app.is_empty())
                })
                .map(|(ancestor, _)| ancestor.app.clone())
                .unwrap_or_default();
        }
        record
    }
}

fn is_at_or_beneath(id: &str, ancestor: &str) -> bool {
    ancestor.is_empty()
        || id
            .strip_prefix(ancestor)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// The ids above `id`, nearest first, ending with the root record's empty id.
fn ancestors(id: &str) -> impl Iterator<Item = &str> {
    let parents = id.rmatch_indices('/').map(move |(index, _)| &id[..index]);
    let root = (!id.is_empty()).then_some("");
    parents.chain(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(id: &str, state: ProgramState, app: &str) -> ProgramStatusReport {
        ProgramStatusReport {
            id: id.to_owned(),
            state,
            progress: None,
            app: app.to_owned(),
            title: String::new(),
            message: String::new(),
        }
    }

    fn ids(records: &ProgramStatusRecords) -> Vec<&str> {
        let mut ids: Vec<_> = records
            .records
            .iter()
            .map(|(record, _)| record.id.as_str())
            .collect();
        ids.sort();
        ids
    }

    #[test]
    fn clearing_a_record_removes_it_and_its_children_only() {
        let mut records = ProgramStatusRecords::default();
        for id in ["", "build", "build/test", "buildx"] {
            records.apply(report(id, ProgramState::Working, ""));
        }
        records.clear("build");
        assert_eq!(ids(&records), ["", "buildx"]);
        records.clear("");
        assert!(ids(&records).is_empty());
    }

    #[test]
    fn a_report_replaces_its_record() {
        let mut records = ProgramStatusRecords::default();
        records.apply(report("", ProgramState::Working, "cargo"));
        records.apply(report("", ProgramState::Done, ""));
        assert_eq!(
            records.most_urgent(),
            Some(report("", ProgramState::Done, ""))
        );
    }

    #[test]
    fn records_need_the_user_once_per_state_change() {
        let mut records = ProgramStatusRecords::default();
        assert_eq!(records.apply(report("", ProgramState::Working, "")), None);
        assert_eq!(
            records.apply(report("", ProgramState::Blocked, "claude")),
            Some(report("", ProgramState::Blocked, "claude"))
        );
        assert_eq!(
            records.apply(report("", ProgramState::Blocked, "claude")),
            None
        );
        // A child inherits the app of its nearest ancestor.
        assert_eq!(
            records.apply(report("task/step", ProgramState::Done, "")),
            Some(report("task/step", ProgramState::Done, "claude"))
        );
    }

    #[test]
    fn the_most_urgent_record_inherits_its_app() {
        let mut records = ProgramStatusRecords::default();
        records.apply(report("", ProgramState::Working, "deploy"));
        records.apply(report("us-east/db", ProgramState::Blocked, ""));
        records.apply(report("eu-west", ProgramState::Done, ""));
        assert_eq!(
            records.most_urgent(),
            Some(report("us-east/db", ProgramState::Blocked, "deploy"))
        );
    }

    #[test]
    fn clearing_running_work_keeps_results_until_seen() {
        let mut records = ProgramStatusRecords::default();
        records.apply(report("a", ProgramState::Idle, ""));
        records.apply(report("b", ProgramState::Working, ""));
        records.apply(report("c", ProgramState::Blocked, ""));
        records.apply(report("d", ProgramState::Done, ""));
        records.apply(report("e", ProgramState::Error, ""));
        assert!(records.has_running());
        assert!(records.clear_running());
        assert_eq!(ids(&records), ["d", "e"]);
        assert!(!records.has_running());
        assert!(!records.clear_running());
        assert!(records.clear_finished());
        assert!(ids(&records).is_empty());
    }

    #[test]
    fn a_full_set_drops_the_record_updated_longest_ago() {
        let mut records = ProgramStatusRecords::default();
        for index in 0..MAX_RECORDS {
            records.apply(report(&index.to_string(), ProgramState::Idle, ""));
        }
        records.apply(report("0", ProgramState::Idle, ""));
        records.apply(report("new", ProgramState::Idle, ""));
        assert_eq!(records.records.len(), MAX_RECORDS);
        assert!(ids(&records).contains(&"0"));
        assert!(!ids(&records).contains(&"1"));
    }
}
