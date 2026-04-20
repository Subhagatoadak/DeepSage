use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, Gauge, Paragraph, Row, Table, Tabs, Wrap,
    },
};

use super::app::{App, Tab};

const ACCENT: Color  = Color::Cyan;
const GOOD:   Color  = Color::Green;
const WARN:   Color  = Color::Yellow;
const BAD:    Color  = Color::Red;
const DIM:    Color  = Color::DarkGray;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title bar
            Constraint::Length(3), // tabs
            Constraint::Min(0),    // content
            Constraint::Length(1), // status bar
        ])
        .split(area);

    draw_title(frame, chunks[0], app);
    draw_tabs(frame, chunks[1], app);
    match app.active_tab {
        Tab::Dashboard => draw_dashboard(frame, chunks[2], app),
        Tab::Models    => draw_models(frame, chunks[2], app),
        Tab::System    => draw_system(frame, chunks[2], app),
        Tab::Logs      => draw_logs(frame, chunks[2], app),
    }
    draw_statusbar(frame, chunks[3], app);
}

// ── Title bar ────────────────────────────────────────────────────────────────

fn draw_title(frame: &mut Frame, area: Rect, app: &App) {
    let hw = app.system_info.as_ref().map(|s| {
        format!("{}  RAM {:.0}GB  VRAM {:.0}GB", s.gpu_name, s.ram_gb, s.vram_gb)
    }).unwrap_or_else(|| "hardware unknown — run `deepsage system`".into());

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" DeepSage ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("│ ", Style::default().fg(DIM)),
        Span::styled(hw, Style::default().fg(Color::White)),
    ]));
    frame.render_widget(title, area);
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

fn draw_tabs(frame: &mut Frame, area: Rect, app: &App) {
    let titles: Vec<Line> = Tab::NAMES.iter().enumerate().map(|(i, &name)| {
        Line::from(format!(" {} [{}] ", name, i + 1))
    }).collect();
    let tabs = Tabs::new(titles)
        .select(app.active_tab.index())
        .block(Block::default().borders(Borders::BOTTOM))
        .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
        .style(Style::default().fg(DIM));
    frame.render_widget(tabs, area);
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

fn draw_dashboard(frame: &mut Frame, area: Rect, app: &mut App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(0)])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(rows[0]);

    draw_running_models(frame, top[0], app);
    draw_resources(frame, top[1], app);
    draw_recommendations(frame, rows[1], app);
}

fn draw_running_models(frame: &mut Frame, area: Rect, app: &App) {
    let header = Row::new(vec!["Model", "Backend", "VRAM", "Endpoint"])
        .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD));
    let rows: Vec<Row> = app.running_models.iter().map(|m| {
        Row::new(vec![
            Cell::from(m.name.clone()),
            Cell::from(m.backend.clone()).style(Style::default().fg(WARN)),
            Cell::from(format!("{:.1}GB", m.vram_gb)),
            Cell::from(m.endpoint.clone().unwrap_or_default()).style(Style::default().fg(DIM)),
        ])
    }).collect();

    let placeholder: Vec<Row> = if rows.is_empty() {
        vec![Row::new(vec![Cell::from("— none running —").style(Style::default().fg(DIM))])]
    } else { vec![] };

    let table = Table::new(
        rows.into_iter().chain(placeholder).collect::<Vec<_>>(),
        [Constraint::Min(16), Constraint::Length(9), Constraint::Length(7), Constraint::Min(20)],
    )
    .header(header)
    .block(Block::default().title(" Running Models ").borders(Borders::ALL));
    frame.render_widget(table, area);
}

fn draw_resources(frame: &mut Frame, area: Rect, app: &App) {
    let s = &app.resource_stats;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(2), Constraint::Length(2), Constraint::Length(2), Constraint::Min(0)])
        .split(area.inner(ratatui::layout::Margin { vertical: 1, horizontal: 1 }));

    let block = Block::default().title(" Resources ").borders(Borders::ALL);
    frame.render_widget(block, area);

    let cpu_label = format!(" CPU  {:.1}%", s.cpu_pct);
    let gauge_cpu = Gauge::default()
        .label(cpu_label)
        .ratio((s.cpu_pct / 100.0).clamp(0.0, 1.0) as f64)
        .gauge_style(Style::default().fg(gauge_color(s.cpu_pct / 100.0)));
    frame.render_widget(gauge_cpu, chunks[1]);

    let ram_label = format!(" RAM  {:.1}/{:.1}GB", s.ram_used_gb, s.ram_total_gb);
    let gauge_ram = Gauge::default()
        .label(ram_label)
        .ratio(s.ram_pct().clamp(0.0, 1.0) as f64)
        .gauge_style(Style::default().fg(gauge_color(s.ram_pct())));
    frame.render_widget(gauge_ram, chunks[2]);

    if s.swap_total_gb > 0.0 {
        let swap_label = format!(" Swap {:.1}/{:.1}GB", s.swap_used_gb, s.swap_total_gb);
        let gauge_swap = Gauge::default()
            .label(swap_label)
            .ratio(s.swap_pct().clamp(0.0, 1.0) as f64)
            .gauge_style(Style::default().fg(gauge_color(s.swap_pct())));
        frame.render_widget(gauge_swap, chunks[3]);
    }
}

fn gauge_color(ratio: f32) -> Color {
    if ratio < 0.6 { GOOD } else if ratio < 0.85 { WARN } else { BAD }
}

fn draw_recommendations(frame: &mut Frame, area: Rect, app: &mut App) {
    let header = Row::new(vec!["#", "Model", "Fit", "Score", "VRAM", "Quant", "Backend"])
        .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = app.recommendations.iter().enumerate().map(|(i, r)| {
        let fit_color = match r.fit_level.to_lowercase().as_str() {
            s if s.contains("perfect") => GOOD,
            s if s.contains("good")    => Color::LightGreen,
            s if s.contains("ok")      => WARN,
            _                           => BAD,
        };
        Row::new(vec![
            Cell::from(format!("{}", i + 1)),
            Cell::from(r.name.clone()).style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from(r.fit_level.clone()).style(Style::default().fg(fit_color)),
            Cell::from(format!("{:.2}", r.score)),
            Cell::from(format!("{:.1}GB", r.vram_required_gb)),
            Cell::from(r.quantization.clone()),
            Cell::from(r.backend.clone()).style(Style::default().fg(DIM)),
        ])
    }).collect();

    let placeholder: Vec<Row> = if rows.is_empty() {
        vec![Row::new(vec![Cell::from(
            if crate::hardware::check(&app.config.llmfit_path) {
                "Loading recommendations…"
            } else {
                "llmfit not found — install with: brew install llmfit"
            }
        ).style(Style::default().fg(DIM))])]
    } else { vec![] };

    let table = Table::new(
        rows.into_iter().chain(placeholder).collect::<Vec<_>>(),
        [Constraint::Length(3), Constraint::Min(20), Constraint::Length(10),
         Constraint::Length(6), Constraint::Length(8), Constraint::Length(10), Constraint::Min(10)],
    )
    .header(header)
    .block(Block::default().title(" llmfit Recommendations ").borders(Borders::ALL));
    frame.render_widget(table, area);
}

// ── Models tab ────────────────────────────────────────────────────────────────

fn draw_models(frame: &mut Frame, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    // Search bar
    let search_style = if app.search_mode {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };
    let search_text = if app.search_mode {
        format!("/ {}_", app.search_query)
    } else if app.search_query.is_empty() {
        "/ search models (press /)".into()
    } else {
        format!("/ {}", app.search_query)
    };
    let search = Paragraph::new(search_text)
        .style(search_style)
        .block(Block::default().borders(Borders::ALL).title(" Search "));
    frame.render_widget(search, chunks[0]);

    // Models table
    let header = Row::new(vec!["Name", "Source", "Backend", "Quant", "VRAM", "RAM", "Active"])
        .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD));

    let filtered = app.filtered_models().into_iter().cloned().collect::<Vec<_>>();
    let rows: Vec<Row> = filtered.iter().map(|m| {
        let active_marker = if m.active { "●" } else { "○" };
        let active_color  = if m.active { GOOD } else { DIM };
        let alloc_note    = if m.alloc_auto { "auto".into() } else { format!("{:.1}G", m.vram_alloc_gb) };
        Row::new(vec![
            Cell::from(m.name.clone()).style(Style::default().add_modifier(Modifier::BOLD)),
            Cell::from(m.source.to_string()).style(Style::default().fg(DIM)),
            Cell::from(m.backend.clone()),
            Cell::from(m.quantization.clone()),
            Cell::from(alloc_note),
            Cell::from(format!("{:.1}G", m.ram_alloc_gb)),
            Cell::from(active_marker).style(Style::default().fg(active_color)),
        ])
    }).collect();

    let placeholder: Vec<Row> = if rows.is_empty() {
        vec![Row::new(vec![Cell::from("No models registered. Use: deepsage register <model>").style(Style::default().fg(DIM))])]
    } else { vec![] };

    let table = Table::new(
        rows.into_iter().chain(placeholder).collect::<Vec<_>>(),
        [Constraint::Min(18), Constraint::Min(22), Constraint::Length(9),
         Constraint::Length(10), Constraint::Length(7), Constraint::Length(6), Constraint::Length(7)],
    )
    .header(header)
    .row_highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
    .highlight_symbol("▶ ")
    .block(Block::default().title(" Registered Models ").borders(Borders::ALL));

    let mut state = app.models_table.clone();
    frame.render_stateful_widget(table, chunks[1], &mut state);
    app.models_table = state;
}

// ── System tab ────────────────────────────────────────────────────────────────

fn draw_system(frame: &mut Frame, area: Rect, app: &App) {
    let text = match &app.system_info {
        None => vec![Line::from(Span::styled(
            "No hardware info yet. Press r to refresh, or run: deepsage system",
            Style::default().fg(DIM),
        ))],
        Some(s) => vec![
            Line::from(vec![
                Span::styled("  CPU Cores  ", Style::default().fg(ACCENT)),
                Span::raw(format!("{}", s.cpu_cores)),
            ]),
            Line::from(vec![
                Span::styled("  RAM        ", Style::default().fg(ACCENT)),
                Span::raw(format!("{:.1} GB", s.ram_gb)),
            ]),
            Line::from(vec![
                Span::styled("  GPU        ", Style::default().fg(ACCENT)),
                Span::raw(if s.gpu_name.is_empty() { "none detected".into() } else { s.gpu_name.clone() }),
            ]),
            Line::from(vec![
                Span::styled("  VRAM       ", Style::default().fg(ACCENT)),
                Span::raw(format!("{:.1} GB", s.vram_gb)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "  Memory Allocation",
                Style::default().fg(ACCENT).add_modifier(Modifier::UNDERLINED),
            )),
            Line::from(vec![
                Span::styled("  Registered models  ", Style::default().fg(DIM)),
                Span::raw(format!("{}", app.registry.models.len())),
            ]),
            Line::from(vec![
                Span::styled("  Manual VRAM alloc  ", Style::default().fg(DIM)),
                Span::raw(format!("{:.1} GB", app.registry.total_vram_alloc())),
            ]),
            Line::from(vec![
                Span::styled("  Available VRAM     ", Style::default().fg(DIM)),
                Span::raw(format!(
                    "{:.1} GB",
                    (s.vram_gb - app.registry.total_vram_alloc()).max(0.0)
                )),
            ]),
        ],
    };

    let para = Paragraph::new(text)
        .block(Block::default().title(" System Hardware ").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(para, area);
}

// ── Logs tab ──────────────────────────────────────────────────────────────────

fn draw_logs(frame: &mut Frame, area: Rect, app: &App) {
    let lines: Vec<Line> = app.logs.iter().map(|l| Line::from(l.as_str())).collect();
    let para = Paragraph::new(lines)
        .block(Block::default().title(" Logs ").borders(Borders::ALL))
        .wrap(Wrap { trim: false })
        .scroll((app.logs_scroll, 0));
    frame.render_widget(para, area);
}

// ── Status bar ────────────────────────────────────────────────────────────────

fn draw_statusbar(frame: &mut Frame, area: Rect, app: &App) {
    let msg = app.status_msg.as_ref().map(|(m, _)| m.as_str()).unwrap_or("");

    let hints = " q:Quit  Tab:Next  r:Run  s:Stop  p:Pull  d:Del  /:Search  PgUp/Dn:Scroll ";
    let span = if !msg.is_empty() {
        Line::from(vec![
            Span::styled(hints, Style::default().fg(DIM)),
            Span::styled(" │ ", Style::default().fg(DIM)),
            Span::styled(msg, Style::default().fg(ACCENT)),
        ])
    } else {
        Line::from(Span::styled(hints, Style::default().fg(DIM)))
    };

    frame.render_widget(Paragraph::new(span), area);
}
