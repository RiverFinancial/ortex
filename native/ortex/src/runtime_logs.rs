//! Bounded callback queue: runtime threads never call BEAM APIs or Elixir Logger.
use ort::logging::{LogLevel, LoggerFunction};
use rustler::Atom;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

const MAX_ENTRIES: usize = 256;
const MAX_FIELD_BYTES: usize = 16_384;
pub type Entries = Vec<(Atom, String, String, String, String)>;

mod atoms {
    rustler::atoms! { debug, info, warning, error, critical }
}

pub fn level(value: i32) -> Result<LogLevel, String> {
    match value {
        0 => Ok(LogLevel::Verbose),
        1 => Ok(LogLevel::Info),
        2 => Ok(LogLevel::Warning),
        3 => Ok(LogLevel::Error),
        4 => Ok(LogLevel::Fatal),
        _ => Err("runtime log level must be 0..4".into()),
    }
}

struct Entry {
    level: LogLevel,
    category: String,
    id: String,
    location: String,
    message: String,
}

#[derive(Default)]
struct State {
    entries: VecDeque<Entry>,
    dropped: usize,
}

pub struct LogBuffer {
    minimum: LogLevel,
    state: Mutex<State>,
}

fn bounded(text: &str) -> String {
    if text.len() <= MAX_FIELD_BYTES {
        return text.to_owned();
    }
    let mut end = MAX_FIELD_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} [truncated]", &text[..end])
}

impl LogBuffer {
    pub fn new(minimum: LogLevel) -> Arc<Self> {
        Arc::new(Self {
            minimum,
            state: Mutex::new(State::default()),
        })
    }

    pub fn logger(self: &Arc<Self>) -> LoggerFunction {
        let buffer = self.clone();
        Arc::new(move |level, category, id, location, message| {
            if level < buffer.minimum {
                return;
            }
            let mut state = buffer.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.entries.len() == MAX_ENTRIES {
                state.dropped = state.dropped.saturating_add(1);
                return;
            }
            state.entries.push_back(Entry {
                level,
                category: bounded(category),
                id: bounded(id),
                location: bounded(location),
                message: bounded(message),
            });
        })
    }

    pub fn drain(&self) -> (Entries, usize) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let entries = state
            .entries
            .drain(..)
            .map(|entry| {
                let level = match entry.level {
                    LogLevel::Verbose => atoms::debug(),
                    LogLevel::Info => atoms::info(),
                    LogLevel::Warning => atoms::warning(),
                    LogLevel::Error => atoms::error(),
                    LogLevel::Fatal => atoms::critical(),
                };
                (
                    level,
                    entry.category,
                    entry.id,
                    entry.location,
                    entry.message,
                )
            })
            .collect();
        let dropped = std::mem::take(&mut state.dropped);
        (entries, dropped)
    }
}
