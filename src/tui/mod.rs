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
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
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

        if app.should_quit { break; }

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

    // Fetch llmfit recommendations (non-blocking — skip on error)
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

    // Fetch running models from Ollama
    let ollama = crate::backends::ollama::OllamaBackend::new(&app.config.ollama.url);
    if let Ok(running) = ollama.running_models().await {
        app.running_models = running;
    }

    app.last_refresh = std::time::Instant::now();
}
