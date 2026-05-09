"""
LLM Pattern Trader — Futuristic Dark Theme
Neon Cyan (#00F0FF) + Electric Pink (#FF2D78) on deep black (#0D0F1A)
Glassmorphism panels with subtle glow effects.
"""

STYLESHEET = """
/* ═══════════════════════ BASE ═══════════════════════ */
QWidget {
    background-color: #0D0F1A;
    color: #C8D6E5;
    font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    font-size: 10pt;
}
QMainWindow {
    background-color: #0D0F1A;
}

/* ═══════════════════ GLASSMORPHISM PANELS ═══════════════════ */
QGroupBox {
    background-color: rgba(16, 20, 38, 0.85);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 12px;
    margin-top: 1.4em;
    padding: 14px 10px 10px 10px;
}
QGroupBox::title {
    color: #00F0FF;
    font-weight: bold;
    font-size: 10pt;
    letter-spacing: 0.5px;
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 14px;
    background-color: #0D0F1A;
    border-radius: 6px;
    border: 1px solid rgba(0, 240, 255, 0.3);
    margin-left: 12px;
}

/* ═══════════════════ BUTTONS ═══════════════════ */
QPushButton {
    background-color: rgba(0, 240, 255, 0.08);
    color: #00F0FF;
    border: 1px solid rgba(0, 240, 255, 0.4);
    padding: 8px 20px;
    font-weight: bold;
    font-size: 10pt;
    border-radius: 8px;
    letter-spacing: 0.5px;
}
QPushButton:hover {
    background-color: rgba(0, 240, 255, 0.18);
    border-color: #00F0FF;
}
QPushButton:pressed {
    background-color: #00F0FF;
    color: #0D0F1A;
}
QPushButton:disabled {
    background-color: rgba(40, 44, 60, 0.5);
    color: #404860;
    border-color: #252840;
}

/* Primary Action Button */
QPushButton#PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF2D78, stop:1 #FF6B9D);
    border: none;
    color: #FFFFFF;
    font-size: 11pt;
    padding: 10px 24px;
}
QPushButton#PrimaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF4D8E, stop:1 #FF8BB3);
}
QPushButton#PrimaryButton:pressed {
    background: #D92062;
}

/* Stop Button */
QPushButton#stop_button {
    background-color: rgba(239, 68, 68, 0.1);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
QPushButton#stop_button:hover {
    background-color: rgba(239, 68, 68, 0.2);
    border-color: #EF4444;
}

/* ═══════════════════ INPUTS ═══════════════════ */
QPlainTextEdit, QLineEdit {
    background-color: rgba(13, 15, 26, 0.9);
    color: #E2E8F0;
    border: 1px solid #1E2240;
    padding: 7px 10px;
    border-radius: 8px;
    selection-background-color: #00F0FF;
    selection-color: #0D0F1A;
}
QPlainTextEdit:focus, QLineEdit:focus {
    border-color: rgba(0, 240, 255, 0.6);
}

/* ═══════════════════ LOG BROWSER ═══════════════════ */
QTextBrowser {
    background-color: rgba(8, 10, 18, 0.95);
    color: #C8D6E5;
    border: 1px solid rgba(0, 240, 255, 0.15);
    padding: 8px;
    border-radius: 10px;
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 9pt;
}

/* ═══════════════════ LABELS ═══════════════════ */
QLabel {
    color: #8892B0;
    background-color: transparent;
}
QLabel#status_label_ok { color: #00F0FF; font-weight: bold; }
QLabel#status_label_error { color: #FF2D78; font-weight: bold; }
QLabel#status_label_pending { color: #FFB800; font-weight: bold; }
QLabel#status_label_stopped { color: #404860; font-weight: bold; }
QLabel#header_label {
    color: #00F0FF;
    font-size: 18pt;
    font-weight: bold;
    letter-spacing: 2px;
}
QLabel#subtitle_label {
    color: #404860;
    font-size: 8pt;
    letter-spacing: 1px;
}
QLabel#stat_value {
    color: #FFFFFF;
    font-size: 16pt;
    font-weight: bold;
}
QLabel#stat_label {
    color: #404860;
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLabel#live_dot {
    color: #10B981;
    font-size: 14pt;
}

/* ═══════════════════ COMBOS & SPINNERS ═══════════════════ */
QComboBox {
    background-color: rgba(16, 20, 38, 0.9);
    border: 1px solid #1E2240;
    border-radius: 8px;
    padding: 6px 10px;
    min-width: 6em;
    color: #E2E8F0;
}
QComboBox:on { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #1E2240;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}
QComboBox QAbstractItemView {
    background-color: #101426;
    border: 1px solid rgba(0, 240, 255, 0.3);
    color: #C8D6E5;
    selection-background-color: #FF2D78;
    outline: none;
}
QSpinBox, QDoubleSpinBox {
    background-color: rgba(16, 20, 38, 0.9);
    border: 1px solid #1E2240;
    border-radius: 8px;
    padding: 6px;
    color: #E2E8F0;
}

/* ═══════════════════ TABS ═══════════════════ */
QTabWidget::pane {
    border: 1px solid rgba(0, 240, 255, 0.15);
    border-radius: 8px;
    background-color: rgba(8, 10, 18, 0.6);
    top: -1px;
}
QTabBar::tab {
    background-color: rgba(16, 20, 38, 0.8);
    color: #506080;
    padding: 8px 18px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border: 1px solid transparent;
    border-bottom: none;
    margin-right: 3px;
    font-weight: 600;
    font-size: 9pt;
}
QTabBar::tab:selected {
    background-color: rgba(0, 240, 255, 0.08);
    color: #00F0FF;
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-bottom: none;
}
QTabBar::tab:hover:!selected {
    background-color: rgba(0, 240, 255, 0.04);
    color: #8892B0;
}

/* ═══════════════════ STATUS BAR ═══════════════════ */
QStatusBar {
    background-color: #080A12;
    border-top: 1px solid rgba(0, 240, 255, 0.15);
    color: #506080;
    font-size: 9pt;
    padding: 2px 8px;
}
QStatusBar::item { border: none; }

/* ═══════════════════ TABLES ═══════════════════ */
QTableWidget, QTableView {
    background-color: #080A12;
    alternate-background-color: #0F1220;
    color: #C8D6E5;
    gridline-color: #1A1D2E;
    border: none;
    font-size: 9pt;
    selection-background-color: rgba(0, 240, 255, 0.12);
    selection-color: #FFFFFF;
}
QTableWidget::item, QTableView::item {
    background-color: #080A12;
    padding: 4px 8px;
    border-bottom: 1px solid #12152A;
}
QTableWidget::item:alternate, QTableView::item:alternate {
    background-color: #0F1220;
}
QTableWidget::item:selected, QTableView::item:selected {
    background-color: rgba(0, 240, 255, 0.15);
    color: #FFFFFF;
}
QHeaderView::section {
    background-color: #0D1025;
    color: #00F0FF;
    font-weight: bold;
    font-size: 8pt;
    padding: 6px 8px;
    border: none;
    border-bottom: 2px solid rgba(0, 240, 255, 0.25);
    letter-spacing: 0.5px;
}
QTableCornerButton::section {
    background-color: #0D1025;
    border: none;
}

/* ═══════════════════ SCROLLBARS ═══════════════════ */
QScrollBar:vertical, QScrollBar:horizontal {
    border: none;
    background-color: #0D0F1A;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #1E2240;
    border-radius: 4px;
    min-height: 30px;
    min-width: 30px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: rgba(0, 240, 255, 0.4);
}
QScrollBar::add-line, QScrollBar::sub-line { border: none; background: none; height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ═══════════════════ PROGRESS ═══════════════════ */
QProgressBar {
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #FFFFFF;
    background-color: #1E2240;
    font-size: 8pt;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00F0FF, stop:1 #00A8E8);
    border-radius: 4px;
}

/* ═══════════════════ CHECKBOX ═══════════════════ */
QCheckBox::indicator:checked {
    background-color: #FF2D78;
    border: 2px solid #FF2D78;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked {
    background-color: #1E2240;
    border: 2px solid #2A2D4A;
    border-radius: 3px;
}

/* ═══════════════════ MISC ═══════════════════ */
QMessageBox { background-color: #101426; }
QMessageBox QLabel { color: #FFFFFF; background-color: transparent; }
"""
