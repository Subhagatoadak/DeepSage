use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::widgets::TableState;
use std::time::{Duration, Instant};

use crate::backends::RunningModel;
use crate::config::Config;
use crate::hardware::{ModelRecommendation, SystemInfo};
use crate::monitor::ResourceStats;
use crate::registry::{ModelEntry, Registry};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tab {
    Dashboard = 0,
    Models    = 1,
    System    = 2,
    Logs      = 3,
}

impl Tab {
    #[allow(dead_code)]
    pub const ALL: &'static [Tab] = &[Tab::Dashboard, Tab::Models, Tab::System, Tab::Logs];
    pub const NAMES: &'static [&'static str] = &["Dashboard", "Models", "System", "Logs"];

    pub fn next(self) -> Self {
        match self {
            Tab::Dashboard => Tab::Models,
            Tab::Models    => Tab::System,
            Tab::System    => Tab::Logs,
            Tab::Logs      => Tab::Dashboard,
        }
    }
    pub fn prev(self) -> Self {
        match self {
            Tab::Dashboard => Tab::Logs,
            Tab::Models    => Tab::Dashboard,
            Tab::System    => Tab::Models,
            Tab::Logs      => Tab::System,
        }
    }
    pub fn index(self) -> usize { self as usize }
}

pub struct App {
    pub should_quit: bool,
    pub active_tab: Tab,

    // Shared state refreshed periodically
    pub resource_stats: ResourceStats,
    pub running_models: Vec<RunningModel>,
    pub recommendations: Vec<ModelRecommendation>,
    pub system_info: Option<SystemInfo>,
    pub registry: Registry,

    // Server status (polled from server.json + /health)
    pub server_running: bool,
    pub server_port: u16,
    pub server_active_model: String,

    // Models tab
    pub models_table: TableState,
    pub search_query: String,
    pub search_mode: bool,

    // Logs tab
    pub logs: Vec<String>,
    pub logs_scroll: u16,

    // Status message shown in status bar
    pub status_msg: Option<(String, Instant)>,

    pub config: Config,
    pub last_refresh: Instant,
    pub refresh_interval: Duration,
}

impl App {
    pub fn new(config: Config, registry: Registry) -> Self {
        Self {
            should_quit: false,
            active_tab: Tab::Dashboard,
            resource_stats: ResourceStats::default(),
            running_models: vec![],
            recommendations: vec![],
            system_info: None,
            registry,
            server_running: false,
            server_port: 8888,
            server_active_model: String::new(),
            models_table: TableState::default(),
            search_query: String::new(),
            search_mode: false,
            logs: vec!["DeepSage started.".into()],
            logs_scroll: 0,
            status_msg: None,
            config,
            last_refresh: Instant::now() - Duration::from_secs(99),
            refresh_interval: Duration::from_secs(3),
        }
    }

    pub fn status(&mut self, msg: impl Into<String>) {
        self.status_msg = Some((msg.into(), Instant::now()));
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        self.logs.push(msg.into());
        // keep at most 500 lines
        if self.logs.len() > 500 {
            self.logs.drain(..self.logs.len() - 500);
        }
    }

    pub fn should_refresh(&self) -> bool {
        self.last_refresh.elapsed() >= self.refresh_interval
    }

    // ── Key handling ──────────────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: KeyEvent) {
        if self.search_mode {
            self.handle_search_key(key);
            return;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Tab => self.active_tab = self.active_tab.next(),
            KeyCode::BackTab => self.active_tab = self.active_tab.prev(),
            KeyCode::Char('1') => self.active_tab = Tab::Dashboard,
            KeyCode::Char('2') => self.active_tab = Tab::Models,
            KeyCode::Char('3') => self.active_tab = Tab::System,
            KeyCode::Char('4') => self.active_tab = Tab::Logs,
            KeyCode::Down | KeyCode::Char('j') => self.next_row(),
            KeyCode::Up   | KeyCode::Char('k') => self.prev_row(),
            KeyCode::Char('/') => {
                self.search_mode = true;
                self.search_query.clear();
            }
            KeyCode::Char('r') => self.action_run_selected(),
            KeyCode::Char('s') => self.action_stop_selected(),
            KeyCode::Char('p') => self.action_pull_selected(),
            KeyCode::Char('d') => self.action_delete_selected(),
            KeyCode::PageDown  => self.logs_scroll = self.logs_scroll.saturating_add(10),
            KeyCode::PageUp    => self.logs_scroll = self.logs_scroll.saturating_sub(10),
            _ => {}
        }
    }

    fn handle_search_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Enter => self.search_mode = false,
            KeyCode::Backspace => { self.search_query.pop(); }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.search_query.push(c);
            }
            _ => {}
        }
    }

    fn next_row(&mut self) {
        let max = match self.active_tab {
            Tab::Models => self.filtered_models().len(),
            Tab::Dashboard => self.recommendations.len(),
            _ => 0,
        };
        if max == 0 { return; }
        let i = self.models_table.selected().map(|i| (i + 1) % max).unwrap_or(0);
        self.models_table.select(Some(i));
    }

    fn prev_row(&mut self) {
        let max = match self.active_tab {
            Tab::Models => self.filtered_models().len(),
            Tab::Dashboard => self.recommendations.len(),
            _ => 0,
        };
        if max == 0 { return; }
        let i = self.models_table.selected()
            .map(|i| if i == 0 { max - 1 } else { i - 1 })
            .unwrap_or(0);
        self.models_table.select(Some(i));
    }

    pub fn filtered_models(&self) -> Vec<&ModelEntry> {
        let q = self.search_query.to_lowercase();
        self.registry.models.iter().filter(|m| {
            q.is_empty() || m.name.to_lowercase().contains(&q) || m.source.to_string().contains(&q)
        }).collect()
    }

    // ── Actions (triggered by key; actual async work done in commands) ────────

    fn selected_model_name(&self) -> Option<String> {
        let idx = self.models_table.selected()?;
        self.filtered_models().get(idx).map(|m| m.name.clone())
    }

    fn action_run_selected(&mut self) {
        if let Some(name) = self.selected_model_name() {
            self.status(format!("Run {}  →  deepsage run {}", name, name));
            self.log(format!("[run] {}", name));
        }
    }

    fn action_stop_selected(&mut self) {
        if let Some(name) = self.selected_model_name() {
            self.status(format!("Stop {}  →  deepsage stop {}", name, name));
            self.log(format!("[stop] {}", name));
        }
    }

    fn action_pull_selected(&mut self) {
        if let Some(name) = self.selected_model_name() {
            self.status(format!("Pull {}  →  deepsage pull {}", name, name));
            self.log(format!("[pull] {}", name));
        }
    }

    fn action_delete_selected(&mut self) {
        if let Some(name) = self.selected_model_name() {
            self.status(format!("Delete {}  →  deepsage delete {}", name, name));
            self.log(format!("[delete] {}", name));
        }
    }
}
