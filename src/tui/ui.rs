use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table, Tabs, Wrap},
    Frame,
};

use super::app::{App, Tab};

const ACCENT: Color = Color::Cyan;
const GOOD: Color = Color::Green;
const WARN: Color = Color::Yellow;
const BAD: Color = Color::Red;
const DIM: Color = Color::DarkGray;

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
        Tab::Models => draw_models(frame, chunks[2], app),
        Tab::System => draw_system(frame, chunks[2], app),
        Tab::Logs => draw_logs(frame, chunks[2], app),
        Tab::Chat => draw_chat(frame, chunks[2], app),
    }
    draw_statusbar(frame, chunks[3], app);
}

// ── Title bar ────────────────────────────────────────────────────────────────

fn draw_title(frame: &mut Frame, area: Rect, app: &App) {
    let hw = app
        .system_info
        .as_ref()
        .map(|s| {
            format!(
                "{}  RAM {:.0}GB  VRAM {:.0}GB",
                s.gpu_name, s.ram_gb, s.vram_gb
            )
        })
        .unwrap_or_else(|| "hardware unknown".into());

    let (srv_sym, srv_text, srv_color) = if app.server_running {
        (
            "●",
            format!(
                " http://127.0.0.1:{}/v1 [{}]",
                app.server_port, app.server_active_model
            ),
            GOOD,
        )
    } else {
        ("○", " not running".into(), DIM)
    };

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " DeepSage ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("│ ", Style::default().fg(DIM)),
        Span::styled(hw, Style::default().fg(Color::White)),
        Span::styled("  │ Server ", Style::default().fg(DIM)),
        Span::styled(
            srv_sym,
            Style::default().fg(srv_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(srv_text, Style::default().fg(srv_color)),
    ]));
    frame.render_widget(title, area);
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

fn draw_tabs(frame: &mut Frame, area: Rect, app: &App) {
    let titles: Vec<Line> = Tab::NAMES
        .iter()
        .enumerate()
        .map(|(i, &name)| Line::from(format!(" {} [{}] ", name, i + 1)))
        .collect();
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
    let title = if app.server_running {
        " Server ● Running "
    } else {
        " Server ○ Stopped "
    };

    let header = Row::new(vec!["Model", "Backend", "OpenAI Endpoint"])
        .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = app
        .running_models
        .iter()
        .map(|m| {
            let endpoint = m.endpoint.as_deref().unwrap_or("-");
            Row::new(vec![
                Cell::from(m.name.clone()).style(Style::default().add_modifier(Modifier::BOLD)),
                Cell::from(m.backend.clone()).style(Style::default().fg(GOOD)),
                Cell::from(format!("{endpoint}/chat/completions"))
                    .style(Style::default().fg(ACCENT)),
            ])
        })
        .collect();

    let placeholder: Vec<Row> = if rows.is_empty() {
        vec![Row::new(vec![Cell::from(
            "— stopped —  run: deepsage serve",
        )
        .style(Style::default().fg(DIM))])]
    } else {
        vec![]
    };

    let table = Table::new(
        rows.into_iter().chain(placeholder).collect::<Vec<_>>(),
        [
            Constraint::Min(18),
            Constraint::Length(10),
            Constraint::Min(30),
        ],
    )
    .header(header)
    .block(Block::default().title(title).borders(Borders::ALL));
    frame.render_widget(table, area);
}

fn draw_resources(frame: &mut Frame, area: Rect, app: &App) {
    let s = &app.resource_stats;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 1,
        }));

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
    if ratio < 0.6 {
        GOOD
    } else if ratio < 0.85 {
        WARN
    } else {
        BAD
    }
}

fn draw_recommendations(frame: &mut Frame, area: Rect, app: &mut App) {
    let header = Row::new(vec![
        "#", "Model", "Fit", "Score", "VRAM", "TPS", "Quant", "Runtime",
    ])
    .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = app
        .recommendations
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let fit_color = match r.fit_level.to_lowercase().as_str() {
                s if s.contains("perfect") => GOOD,
                s if s.contains("good") => Color::LightGreen,
                s if s.contains("ok") => WARN,
                _ => BAD,
            };
            let short_name = r.name.splitn(2, '/').last().unwrap_or(&r.name).to_string();
            Row::new(vec![
                Cell::from(format!("{}", i + 1)),
                Cell::from(short_name).style(Style::default().add_modifier(Modifier::BOLD)),
                Cell::from(r.fit_level.clone()).style(Style::default().fg(fit_color)),
                Cell::from(format!("{:.1}", r.score)),
                Cell::from(format!("{:.1}G", r.vram_required_gb)),
                Cell::from(format!("{:.1}", r.estimated_tps)),
                Cell::from(r.quantization.clone()),
                Cell::from(r.backend.clone()).style(Style::default().fg(DIM)),
            ])
        })
        .collect();

    let placeholder: Vec<Row> = if rows.is_empty() {
        vec![Row::new(vec![Cell::from(
            if crate::hardware::check(&app.config.llmfit_path) {
                "Loading recommendations…"
            } else {
                "llmfit not found — install with: brew install llmfit"
            },
        )
        .style(Style::default().fg(DIM))])]
    } else {
        vec![]
    };

    let table = Table::new(
        rows.into_iter().chain(placeholder).collect::<Vec<_>>(),
        [
            Constraint::Length(3),
            Constraint::Min(22),
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Min(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .title(" llmfit Recommendations ")
            .borders(Borders::ALL),
    );
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
    let header = Row::new(vec![
        "Name", "Source", "Backend", "Quant", "VRAM", "RAM", "Active",
    ])
    .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD));

    let filtered = app
        .filtered_models()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let rows: Vec<Row> = filtered
        .iter()
        .map(|m| {
            let active_marker = if m.active { "●" } else { "○" };
            let active_color = if m.active { GOOD } else { DIM };
            let alloc_note = if m.alloc_auto {
                "auto".into()
            } else {
                format!("{:.1}G", m.vram_alloc_gb)
            };
            Row::new(vec![
                Cell::from(m.name.clone()).style(Style::default().add_modifier(Modifier::BOLD)),
                Cell::from(m.source.to_string()).style(Style::default().fg(DIM)),
                Cell::from(m.backend.clone()),
                Cell::from(m.quantization.clone()),
                Cell::from(alloc_note),
                Cell::from(format!("{:.1}G", m.ram_alloc_gb)),
                Cell::from(active_marker).style(Style::default().fg(active_color)),
            ])
        })
        .collect();

    let placeholder: Vec<Row> = if rows.is_empty() {
        vec![Row::new(vec![Cell::from(
            "No models registered. Use: deepsage register <model>",
        )
        .style(Style::default().fg(DIM))])]
    } else {
        vec![]
    };

    let table = Table::new(
        rows.into_iter().chain(placeholder).collect::<Vec<_>>(),
        [
            Constraint::Min(18),
            Constraint::Min(22),
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(7),
            Constraint::Length(6),
            Constraint::Length(7),
        ],
    )
    .header(header)
    .row_highlight_style(
        Style::default()
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    )
    .highlight_symbol("▶ ")
    .block(
        Block::default()
            .title(" Registered Models ")
            .borders(Borders::ALL),
    );

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
                Span::styled("  CPU        ", Style::default().fg(ACCENT)),
                Span::raw(format!("{} ({} cores)", s.cpu_name, s.cpu_cores)),
            ]),
            Line::from(vec![
                Span::styled("  RAM        ", Style::default().fg(ACCENT)),
                Span::raw(format!("{:.1} GB", s.ram_gb)),
            ]),
            Line::from(vec![
                Span::styled("  GPU        ", Style::default().fg(ACCENT)),
                Span::raw(if s.gpu_name.is_empty() {
                    "none detected".into()
                } else if s.unified_memory {
                    format!("{} (unified memory)", s.gpu_name)
                } else {
                    s.gpu_name.clone()
                }),
            ]),
            Line::from(vec![
                Span::styled("  VRAM       ", Style::default().fg(ACCENT)),
                Span::raw(format!(
                    "{:.1} GB{}",
                    s.vram_gb,
                    if s.unified_memory {
                        " (shared with RAM)"
                    } else {
                        ""
                    }
                )),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "  Memory Allocation",
                Style::default()
                    .fg(ACCENT)
                    .add_modifier(Modifier::UNDERLINED),
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
        .block(
            Block::default()
                .title(" System Hardware ")
                .borders(Borders::ALL),
        )
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

// ── Chat tab ──────────────────────────────────────────────────────────────────

fn draw_chat(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),    // message history
            Constraint::Length(3), // input bar
        ])
        .split(area);

    // Build message history lines
    let mut lines: Vec<Line> = Vec::new();
    for msg in &app.chat_messages {
        let (label, color, dim_content) = match msg.role.as_str() {
            "user"        => ("You   ", ACCENT, false),
            "assistant"   => ("AI    ", GOOD,   false),
            "tool_call"   => ("Tool▶ ", WARN,   true),
            "tool_result" => ("Tool← ", Color::Magenta, true),
            _             => ("      ", DIM,    true),
        };
        lines.push(Line::from(Span::styled(
            format!("{label}:"),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        // Split multi-line content into separate lines
        for content_line in msg.content.lines() {
            let span = if dim_content {
                Span::styled(format!("  {content_line}"), Style::default().fg(DIM))
            } else {
                Span::raw(format!("  {content_line}"))
            };
            lines.push(Line::from(span));
        }
        // Blinking cursor while streaming into empty assistant bubble
        if msg.content.is_empty() && msg.role == "assistant" {
            lines.push(Line::from(Span::styled(
                "  ▋",
                Style::default().fg(GOOD),
            )));
        }
        lines.push(Line::from(""));
    }

    if lines.is_empty() {
        let active_model = app
            .registry
            .models
            .iter()
            .find(|m| m.active)
            .map(|m| m.name.as_str())
            .unwrap_or("none");
        lines.push(Line::from(Span::styled(
            format!("  Active model: {active_model}  — press i to start chatting"),
            Style::default().fg(DIM),
        )));
    }

    let history_title = if app.chat_waiting {
        " Chat ● Thinking… "
    } else {
        " Chat "
    };

    let history = Paragraph::new(lines)
        .block(
            Block::default()
                .title(history_title)
                .borders(Borders::ALL)
                .border_style(if app.chat_waiting {
                    Style::default().fg(WARN)
                } else {
                    Style::default()
                }),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.chat_scroll, 0));
    frame.render_widget(history, chunks[0]);

    // Input bar
    let (input_border_style, input_title, cursor) = if app.chat_active {
        (
            Style::default().fg(ACCENT),
            " Message (Enter to send, Esc to cancel) ",
            "_",
        )
    } else if app.chat_waiting {
        (Style::default().fg(DIM), " Waiting for response… ", "")
    } else {
        (Style::default().fg(DIM), " Message (press i to type) ", "")
    };

    let input_text = format!("{}{}", app.chat_input, cursor);
    let input = Paragraph::new(input_text)
        .style(if app.chat_active {
            Style::default().fg(Color::White)
        } else {
            Style::default().fg(DIM)
        })
        .block(
            Block::default()
                .title(input_title)
                .borders(Borders::ALL)
                .border_style(input_border_style),
        );
    frame.render_widget(input, chunks[1]);
}

// ── Status bar ────────────────────────────────────────────────────────────────

fn draw_statusbar(frame: &mut Frame, area: Rect, app: &App) {
    let msg = app
        .status_msg
        .as_ref()
        .map(|(m, _)| m.as_str())
        .unwrap_or("");

    let hints = match app.active_tab {
        Tab::Chat if app.chat_active => {
            " Enter:Send  Esc:Cancel input "
        }
        Tab::Chat => {
            " i:Type  q:Quit  Tab:Next  PgUp/Dn:Scroll "
        }
        _ => {
            " q:Quit  Tab:Next  r:Run  s:Stop  p:Pull  d:Del  /:Search  PgUp/Dn:Scroll "
        }
    };

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
