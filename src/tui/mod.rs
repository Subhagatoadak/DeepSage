mod app;
mod ui;

pub use app::App;

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{io, time::Duration};

use crate::{config::Config, monitor, registry};

pub async fn run(config: Config) -> Result<()> {
    let reg = registry::load().unwrap_or_default();
    let mut app = App::new(config, reg);

    // Initial data load before entering TUI
    refresh_data(&mut app).await;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = event_loop(&mut terminal, &mut app).await;

    // Always restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        // Poll for events with 16ms timeout (~60fps)
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                app.handle_key(key);
            }
        }

        if app.should_quit {
            break;
        }

        // Periodic data refresh
        if app.should_refresh() {
            refresh_data(app).await;
        }

        // Clear stale status messages after 4 seconds
        if let Some((_, ts)) = &app.status_msg {
            if ts.elapsed() > Duration::from_secs(4) {
                app.status_msg = None;
            }
        }
    }
    Ok(())
}

async fn refresh_data(app: &mut App) {
    app.resource_stats = monitor::collect();

    // Reload registry from disk (may have changed via CLI)
    if let Ok(reg) = registry::load() {
        app.registry = reg;
    }

    // Fetch llmfit recommendations — optional, skip if not installed
    if crate::hardware::check(&app.config.llmfit_path) {
        if let Ok(recs) = crate::hardware::recommendations(5, &app.config.llmfit_path) {
            app.recommendations = recs;
        }
        if app.system_info.is_none() {
            if let Ok(info) = crate::hardware::system_info(&app.config.llmfit_path) {
                app.system_info = Some(info);
            }
        }
    }

    // Check DeepSage server status via state file + live health probe
    app.server_running = false;
    app.running_models.clear();

    if let Some(state) = crate::server::read_serve_state() {
        app.server_port = state.port;
        app.server_active_model = state.active_model.clone();

        // Confirm it is actually alive
        let url = format!("http://127.0.0.1:{}/health", state.port);
        if let Ok(client) = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(800))
            .build()
        {
            if client
                .get(&url)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                app.server_running = true;
                app.running_models = vec![crate::backends::RunningModel {
                    name: state.active_model.clone(),
                    backend: "llamacpp".into(),
                    pid: Some(state.pid),
                    vram_gb: 0.0,
                    ram_gb: 0.0,
                    endpoint: Some(format!("http://127.0.0.1:{}/v1", state.port)),
                }];
            }
        }
    }

    app.last_refresh = std::time::Instant::now();
}
