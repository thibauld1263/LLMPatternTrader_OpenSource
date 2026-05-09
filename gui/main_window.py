import os
import re
import sys
import json
import logging
import asyncio
import markdown
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextBrowser,
    QComboBox, QMessageBox, QStatusBar, QSplitter, QDesktopWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QFrame,
    QSpacerItem, QSizePolicy, QSpinBox, QCheckBox, QScrollArea
)
from PyQt5.QtCore import pyqtSlot, Qt, QTimer, QThread, pyqtSignal as Signal
from PyQt5.QtGui import QIcon, QTextCursor, QColor, QFont
from .style import STYLESHEET
from .chart import ChartWidget
from services.config_manager import ConfigManager
import resources.resources_rc
import pyqtgraph as pg
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════ MT5 BACKGROUND WORKER ═══════════════════
class MT5PositionWorker(QThread):
    """Runs all blocking MT5 calls in a background thread."""
    result_ready = Signal(dict)

    def __init__(self, magic_number: int):
        super().__init__()
        self.magic_number = magic_number

    def run(self):
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                self.result_ready.emit({"connected": False})
                return
            info = mt5.terminal_info()
            if not info:
                self.result_ready.emit({"connected": False})
                return

            positions = mt5.positions_get()
            pos_list = []
            total_pnl = 0.0
            if positions:
                for pos in positions:
                    if pos.magic != self.magic_number:
                        continue
                    sym_info = mt5.symbol_info(pos.symbol)
                    point = sym_info.point if sym_info else 0.00001
                    pip_val = point * 10
                    tick = mt5.symbol_info_tick(pos.symbol)
                    current = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask if tick else pos.price_current
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        pnl_pips = (current - pos.price_open) / pip_val if pip_val > 0 else 0
                    else:
                        pnl_pips = (pos.price_open - current) / pip_val if pip_val > 0 else 0
                    total_pnl += pos.profit
                    pos_list.append({
                        "ticket": pos.ticket, "symbol": pos.symbol,
                        "direction": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                        "volume": pos.volume, "entry": pos.price_open,
                        "sl": pos.sl, "tp": pos.tp,
                        "pnl_pips": round(pnl_pips, 1), "pnl_usd": round(pos.profit, 2)
                    })

            account = mt5.account_info()
            balance = account.balance if account else None

            self.result_ready.emit({
                "connected": True, "positions": pos_list,
                "total_pnl": total_pnl, "balance": balance
            })
        except Exception:
            self.result_ready.emit({"connected": False})

class StatCard(QFrame):
    """A futuristic stat card with glowing value."""
    def __init__(self, icon, label_text, initial_value="—", color="#00F0FF"):
        super().__init__()
        self.setFixedHeight(80)
        self.setMinimumWidth(170)
        self.setStyleSheet("""
            StatCard {
                background-color: rgba(16, 20, 38, 0.85);
                border: 1px solid rgba(0, 240, 255, 0.15);
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(2)
        
        # Icon + Label row
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 14pt; background: transparent;")
        top_row.addWidget(icon_label)
        self.desc_label = QLabel(label_text.upper())
        self.desc_label.setStyleSheet("color: #506080; font-size: 7pt; letter-spacing: 1.5px; background: transparent; font-weight: bold;")
        top_row.addWidget(self.desc_label)
        top_row.addStretch(1)
        layout.addLayout(top_row)
        
        # Value
        self.value_label = QLabel(initial_value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 20pt; font-weight: bold; background: transparent; font-family: 'Segoe UI', 'Inter', sans-serif;")
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.value_label)
    
    def set_value(self, text, color=None):
        self.value_label.setText(str(text))
        if color:
            self.value_label.setStyleSheet(f"color: {color}; font-size: 20pt; font-weight: bold; background: transparent; font-family: 'Segoe UI', 'Inter', sans-serif;")

class MainWindow(QMainWindow):

    def __init__(self, engine, config_manager: ConfigManager, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.engine = engine
        self.config_manager = config_manager
        self.loop = loop
        self.trading_task = None
        self._loaded_training_cache = None  # Loaded from file, cleared after use
        
        self.setWindowTitle("LLM Pattern Trader")
        
        screen = QDesktopWidget().availableGeometry()
        w = int(screen.width() * 0.90)
        h = int(screen.height() * 0.90)
        x = int(screen.left() + (screen.width() - w) / 2)
        y = int(screen.top() + (screen.height() - h) / 2)
        
        self.setGeometry(x, y, w, h)
        self.setWindowIcon(QIcon(":/logo.png"))
        self.setStyleSheet(STYLESHEET)
        
        # Central Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 6, 10, 4)
        main_layout.setSpacing(6)
        
        # ══════════════════ TOP: STAT CARDS ══════════════════
        stats_bar = QHBoxLayout()
        stats_bar.setSpacing(10)
        
        self.stat_balance = StatCard("💰", "Account Balance", "—", "#FFFFFF")
        self.stat_positions = StatCard("⚡", "Open Positions", "0", "#00F0FF")
        self.stat_pnl = StatCard("📊", "Floating PnL", "$0.00", "#10B981")
        self.stat_patterns = StatCard("🎯", "Validated Patterns", "—", "#FFB800")
        
        for card in [self.stat_balance, self.stat_positions, self.stat_pnl, self.stat_patterns]:
            stats_bar.addWidget(card)
        
        main_layout.addLayout(stats_bar)
        
        # ══════════════════ MIDDLE SPLITTER: Config | Charts ══════════════════
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setHandleWidth(2)
        top_splitter.setStyleSheet("QSplitter::handle { background-color: rgba(0, 240, 255, 0.08); }")
        
        # ─── LEFT: Config Panel ───
        self.config_pane = QWidget()
        self.config_layout = QVBoxLayout(self.config_pane)
        self.config_layout.setContentsMargins(0, 0, 4, 0)
        self.config_layout.setSpacing(6)
        self.create_connection_group()
        self.create_pattern_group()
        self.create_llm_group()
        self.create_risk_group()
        self.create_control_group()
        self.config_layout.addStretch(1)
        
        # ─── RIGHT: Chart Tabs ───
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(0)
        
        self.chart_tabs = QTabWidget()
        
        # Tab 1: Chart + Selector
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 4, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        view_label = QLabel("◈ SYMBOL:")
        view_label.setStyleSheet("color: #00F0FF; font-weight: bold; font-size: 9pt;")
        toolbar.addWidget(view_label)
        self.chart_selector = QComboBox()
        self.chart_selector.addItem("— Select —")
        self.chart_selector.currentIndexChanged.connect(self._on_chart_selector_changed)
        self.chart_selector.setMinimumWidth(180)
        toolbar.addWidget(self.chart_selector)
        toolbar.addStretch(1)
        chart_layout.addLayout(toolbar)
        self.chart_widget = ChartWidget()
        chart_layout.addWidget(self.chart_widget)
        self.chart_tabs.addTab(chart_container, "📊  Candlestick")
        
        # Tab 2: Equity Curve
        self.equity_widget = pg.PlotWidget()
        self.equity_widget.showGrid(x=True, y=True, alpha=0.08)
        self.equity_widget.setBackground('#080A12')
        self.chart_tabs.addTab(self.equity_widget, "📈  Equity Curve")
        
        # Tab 3: Backtest Trades
        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(8)
        self.trade_table.setHorizontalHeaderLabels([
            "Date", "Symbol", "Pattern", "Direction", "Entry", "Exit", "PnL (pips)", "Outcome"
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trade_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.chart_tabs.addTab(self.trade_table, "📋  Backtest Trades")
        
        # Tab 4: LLM Signals
        self.signal_table = QTableWidget()
        self.signal_table.setColumnCount(9)
        self.signal_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Pattern", "Direction", "Entry", "SL", "TP", "Risk%", "LLM Verdict"
        ])
        self.signal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.signal_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.signal_table.setAlternatingRowColors(True)
        self.signal_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.signal_table.verticalHeader().setVisible(False)
        self.chart_tabs.addTab(self.signal_table, "🧠  LLM Signals")
        
        right_layout.addWidget(self.chart_tabs)
        
        # Wrap config pane in scroll area so it can shrink freely
        config_scroll = QScrollArea()
        config_scroll.setWidget(self.config_pane)
        config_scroll.setWidgetResizable(True)
        config_scroll.setMinimumHeight(0)
        config_scroll.setMinimumWidth(220)
        config_scroll.setFrameShape(QFrame.NoFrame)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        top_splitter.addWidget(config_scroll)
        top_splitter.addWidget(right_widget)
        top_splitter.setSizes([220, 800])
        
        # Vertical splitter: top (charts) | bottom (orders+log), both resizable
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.setHandleWidth(3)
        vertical_splitter.setStyleSheet("QSplitter::handle { background-color: rgba(0, 240, 255, 0.12); }")
        vertical_splitter.addWidget(top_splitter)
        # bottom_splitter will be added below after it's created
        
        # ══════════════════ BOTTOM: Open Orders + Execution Log ══════════════════
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.setHandleWidth(2)
        bottom_splitter.setStyleSheet("QSplitter::handle { background-color: rgba(0, 240, 255, 0.08); }")
        
        # Left Bottom: Open Orders Dashboard
        orders_group = QGroupBox("⚡ Open Positions")
        orders_layout = QVBoxLayout(orders_group)
        orders_layout.setContentsMargins(6, 14, 6, 6)
        
        self.orders_table = QTableWidget()
        self.orders_table.setColumnCount(9)
        self.orders_table.setHorizontalHeaderLabels([
            "Ticket", "Symbol", "Dir", "Volume", "Entry", "SL", "TP", "PnL (pips)", "PnL ($)"
        ])
        self.orders_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.orders_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.orders_table.setAlternatingRowColors(True)
        self.orders_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.orders_table.verticalHeader().setVisible(False)
        orders_layout.addWidget(self.orders_table)
        
        # Right Bottom: Execution Log
        log_group = QGroupBox("📋 Execution Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 14, 6, 6)
        self.log_browser = QTextBrowser()
        self.log_browser.setOpenExternalLinks(True)
        log_layout.addWidget(self.log_browser)
        
        # Pattern Results Tab
        pattern_group = QGroupBox("🎯 Pattern Results (pips)")
        pattern_layout = QVBoxLayout(pattern_group)
        pattern_layout.setContentsMargins(6, 14, 6, 6)
        self.pattern_results_table = QTableWidget()
        self.pattern_results_table.setColumnCount(8)
        self.pattern_results_table.setHorizontalHeaderLabels([
            "Symbol", "Pattern", "Trades", "Win Rate", "PF", "Cumul (pips)", "Avg Gain (pips)", "Max DD (pips)"
        ])
        self.pattern_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pattern_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pattern_results_table.setAlternatingRowColors(True)
        self.pattern_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pattern_results_table.verticalHeader().setVisible(False)
        self.pattern_results_table.setSortingEnabled(True)
        pattern_layout.addWidget(self.pattern_results_table)
        
        bottom_splitter.addWidget(orders_group)
        bottom_splitter.addWidget(log_group)
        bottom_splitter.addWidget(pattern_group)
        bottom_splitter.setSizes([400, 350, 350])
        
        vertical_splitter.addWidget(bottom_splitter)
        vertical_splitter.setChildrenCollapsible(False)
        vertical_splitter.setSizes([500, 300])
        main_layout.addWidget(vertical_splitter)
        
        # Status Bar
        self.create_status_bar()
        self.load_configuration_into_ui()
        
        # Engine Signals
        self.engine.status_update.connect(self.update_status_label)
        self.engine.connection_status.connect(self.update_connection_status)
        self.engine.finished.connect(self.on_engine_finished)
        self.engine.chart_update.connect(self._cache_chart_data)
        self.engine.equity_update.connect(self._update_equity_curve)
        self.engine.trade_list_update.connect(self._update_trade_list)
        self.engine.positions_update.connect(self._update_open_orders)
        self.engine.signal_update.connect(self._update_signal_table)
        self.engine.pattern_results_update.connect(self._update_pattern_results)
        self.engine.training_complete.connect(self._on_training_complete)
        self.llm_api_key_input.textChanged.connect(self._update_credentials_in_memory)
        
        self.chart_cache = {}
        
        # Live position refresh timer (every 3 seconds) — runs in background thread
        self._mt5_worker = None
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self._request_position_refresh)
        self.position_timer.start(3000)

    # ═══════════════════ CONFIG GROUPS ═══════════════════

    def create_connection_group(self):
        group = QGroupBox("MT5 Connection")
        layout = QGridLayout(group)
        layout.setSpacing(6)
        self.mt4_path_input = QLineEdit()
        self.mt4_path_input.setPlaceholderText("e.g. C:/Program Files/MetaTrader 5/terminal64.exe")
        layout.addWidget(QLabel("Terminal:"), 0, 0)
        layout.addWidget(self.mt4_path_input, 0, 1)
        self.config_layout.addWidget(group)

    def create_pattern_group(self):
        group = QGroupBox("Pattern Scanner")
        layout = QGridLayout(group)
        layout.setSpacing(6)
        
        self.symbols_input = QLineEdit()
        self.symbols_input.setToolTip("Comma-separated (e.g., EURUSD,GBPUSD)")
        self.timeframes_input = QLineEdit()
        self.timeframes_input.setToolTip("Comma-separated (e.g., H1, H4)")
        self.win_rate_input = QLineEdit()
        self.win_rate_input.setPlaceholderText("0.40")
        self.pf_input = QLineEdit()
        self.pf_input.setPlaceholderText("1.4")
        self.trades_input = QLineEdit()
        self.trades_input.setPlaceholderText("10")
        self.bars_input = QLineEdit()
        self.bars_input.setToolTip("Bars to fetch per symbol/TF")
        
        # Trading hours
        self.start_hour_spin = QSpinBox()
        self.start_hour_spin.setRange(0, 23)
        self.start_hour_spin.setValue(4)
        self.start_hour_spin.setSuffix(" Broker")
        self.end_hour_spin = QSpinBox()
        self.end_hour_spin.setRange(0, 23)
        self.end_hour_spin.setValue(22)
        self.end_hour_spin.setSuffix(" Broker")
        
        layout.addWidget(QLabel("Symbols:"),    0, 0); layout.addWidget(self.symbols_input,    0, 1)
        layout.addWidget(QLabel("Timeframes:"),  1, 0); layout.addWidget(self.timeframes_input, 1, 1)
        layout.addWidget(QLabel("Bars:"),        2, 0); layout.addWidget(self.bars_input,       2, 1)
        layout.addWidget(QLabel("Min WR:"),      3, 0); layout.addWidget(self.win_rate_input,   3, 1)
        layout.addWidget(QLabel("Min PF:"),      4, 0); layout.addWidget(self.pf_input,         4, 1)
        layout.addWidget(QLabel("Min Trades:"),  5, 0); layout.addWidget(self.trades_input,     5, 1)
        layout.addWidget(QLabel("Start Hour:"),  6, 0); layout.addWidget(self.start_hour_spin,  6, 1)
        layout.addWidget(QLabel("End Hour:"),    7, 0); layout.addWidget(self.end_hour_spin,    7, 1)
        
        self.config_layout.addWidget(group)

    def create_llm_group(self):
        group = QGroupBox("AI Risk Manager")
        layout = QGridLayout(group)
        layout.setSpacing(6)
        
        self.model_combo = QComboBox()
        self.llm_api_key_input = QLineEdit()
        self.llm_api_key_input.setEchoMode(QLineEdit.Password)
        self.llm_api_key_input.setPlaceholderText("API Key")
        
        layout.addWidget(QLabel("Model:"),   0, 0); layout.addWidget(self.model_combo,      0, 1)
        layout.addWidget(QLabel("API Key:"), 1, 0); layout.addWidget(self.llm_api_key_input, 1, 1)
        
        self.config_layout.addWidget(group)

    def create_risk_group(self):
        group = QGroupBox("Risk Management")
        layout = QGridLayout(group)
        layout.setSpacing(6)
        
        self.risk_mode_combo = QComboBox()
        self.risk_mode_combo.addItems(["Fix Lot Size", "Ultra Safe", "Conservative", "Moderate", "Aggressive", "Max Risk"])
        self.risk_mode_combo.setCurrentText("Moderate")
        self.risk_mode_combo.currentTextChanged.connect(self._on_risk_mode_changed)
        
        self.fixed_lot_input = QLineEdit()
        self.fixed_lot_input.setText("0.01")
        self.fixed_lot_input.setPlaceholderText("0.01")
        self.fixed_lot_input.setToolTip("Fixed lot size (only used in Fix Lot Size mode)")
        
        self.risk_label = QLabel("Safe: 0.15% | 0.35% | 0.5%")
        self.risk_label.setStyleSheet("color: #8892B0; font-size: 8pt;")
        
        layout.addWidget(QLabel("Mode:"),     0, 0); layout.addWidget(self.risk_mode_combo, 0, 1)
        layout.addWidget(QLabel("Fix Lots:"), 1, 0); layout.addWidget(self.fixed_lot_input, 1, 1)
        layout.addWidget(self.risk_label, 2, 0, 1, 2)
        
        # Initial state
        self._on_risk_mode_changed("Moderate")
        
        self.config_layout.addWidget(group)
    
    def _on_risk_mode_changed(self, mode):
        is_fixed = (mode == "Fix Lot Size")
        self.fixed_lot_input.setEnabled(is_fixed)
        self.fixed_lot_input.setStyleSheet("" if is_fixed else "background-color: #12152A; color: #404860;")
        
        labels = {
            "Fix Lot Size":  f"Fixed at {self.fixed_lot_input.text()} lots per trade",
            "Ultra Safe":    "Risk: 0.04% → 0.25% | Cap: 2%",
            "Conservative":  "Risk: 0.08% → 0.50% | Cap: 5%",
            "Moderate":      "Risk: 0.15% → 1.00% | Cap: 8%",
            "Aggressive":    "Risk: 0.25% → 1.50% | Cap: 12%",
            "Max Risk":      "Risk: 0.50% → 2.50% | Cap: 15%",
        }
        self.risk_label.setText(labels.get(mode, ""))

    def create_control_group(self):
        group = QGroupBox("Execution")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        
        # Magic Number + Comment row
        id_row = QGridLayout()
        id_row.setSpacing(4)
        id_row.addWidget(QLabel("Magic:"), 0, 0)
        self.magic_number_spin = QSpinBox()
        self.magic_number_spin.setRange(0, 999999999)
        self.magic_number_spin.setValue(123456)
        id_row.addWidget(self.magic_number_spin, 0, 1)
        id_row.addWidget(QLabel("Comment:"), 0, 2)
        self.trade_comment_input = QLineEdit("PT")
        self.trade_comment_input.setMaxLength(20)
        self.trade_comment_input.setPlaceholderText("e.g. PT")
        id_row.addWidget(self.trade_comment_input, 0, 3)
        layout.addLayout(id_row)

        # News buffer settings
        news_row = QGridLayout()
        news_row.setSpacing(4)
        news_row.addWidget(QLabel("News Buffer Before (min):"), 0, 0)
        self.news_before_spin = QSpinBox()
        self.news_before_spin.setRange(0, 480)
        self.news_before_spin.setValue(120)
        self.news_before_spin.setToolTip("Minutes before a news event to stop opening new trades")
        news_row.addWidget(self.news_before_spin, 0, 1)
        news_row.addWidget(QLabel("After (min):"), 0, 2)
        self.news_after_spin = QSpinBox()
        self.news_after_spin.setRange(0, 480)
        self.news_after_spin.setValue(120)
        self.news_after_spin.setToolTip("Minutes after a news event to resume opening new trades")
        news_row.addWidget(self.news_after_spin, 0, 3)
        layout.addLayout(news_row)
        
        self.signal_only_check = QCheckBox("Signal Only (no auto-execution)")
        self.signal_only_check.setChecked(True)
        self.signal_only_check.setStyleSheet("color: #FFB800; font-size: 9pt;")
        self.signal_only_check.setToolTip("When checked, the LLM evaluates signals but no orders are placed. Signals are shown in the LLM Signals tab.")
        layout.addWidget(self.signal_only_check)
        
        self.llm_filter_check = QCheckBox("LLM Filter (AI validates each signal)")
        self.llm_filter_check.setChecked(True)
        self.llm_filter_check.setStyleSheet("color: #00F0FF; font-size: 9pt;")
        self.llm_filter_check.setToolTip("When enabled, each pattern signal is sent to the LLM for approval before execution.\nWhen disabled, all validated patterns trade directly with default ATR risk parameters.")
        layout.addWidget(self.llm_filter_check)
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        self.run_button = QPushButton("▶  LAUNCH")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.start_trading)
        self.run_button.setFixedHeight(38)
        
        self.stop_button = QPushButton("■  STOP")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_trading)
        self.stop_button.setFixedHeight(38)
        
        btn_row.addWidget(self.run_button)
        btn_row.addWidget(self.stop_button)
        layout.addLayout(btn_row)
        
        # Save / Load Training State buttons
        training_row = QHBoxLayout()
        training_row.setSpacing(8)
        
        self.save_training_btn = QPushButton("💾  Save Training")
        self.save_training_btn.setFixedHeight(32)
        self.save_training_btn.setEnabled(False)
        self.save_training_btn.setToolTip("Save validated patterns to a file so you can resume later without re-training.")
        self.save_training_btn.setStyleSheet("QPushButton { background-color: #1A2744; color: #506080; border: 1px solid #1E2A4A; border-radius: 6px; font-size: 9pt; } QPushButton:enabled { color: #10B981; border-color: #10B981; } QPushButton:enabled:hover { background-color: #10B98122; }")
        self.save_training_btn.clicked.connect(self._save_training)
        
        self.load_training_btn = QPushButton("📂  Load Training")
        self.load_training_btn.setFixedHeight(32)
        self.load_training_btn.setToolTip("Load a previously saved training state. Already-trained pairs will skip training on next launch.")
        self.load_training_btn.setStyleSheet("QPushButton { background-color: #1A2744; color: #8892B0; border: 1px solid #1E2A4A; border-radius: 6px; font-size: 9pt; } QPushButton:hover { background-color: #00F0FF11; color: #00F0FF; border-color: #00F0FF; }")
        self.load_training_btn.clicked.connect(self._load_training)
        
        training_row.addWidget(self.save_training_btn)
        training_row.addWidget(self.load_training_btn)
        layout.addLayout(training_row)
        
        # Loaded state label
        self.loaded_state_label = QLabel("")
        self.loaded_state_label.setStyleSheet("color: #10B981; font-size: 8pt; padding: 2px;")
        self.loaded_state_label.setWordWrap(True)
        layout.addWidget(self.loaded_state_label)
        
        self.config_layout.addWidget(group)

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Stopped")
        self.status_label.setObjectName("status_label_stopped")
        self.status_bar.addPermanentWidget(QLabel("Status: "))
        self.status_bar.addPermanentWidget(self.status_label)
        
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setObjectName("status_label_error")
        self.status_bar.addPermanentWidget(QLabel("  │  MT5: "))
        self.status_bar.addPermanentWidget(self.connection_label)
        
        self.timer_text_label = QLabel("  │  Next Candle: ")
        self.status_bar.addPermanentWidget(self.timer_text_label)
        self.timer_label = QLabel("--:--:--")
        self.timer_label.setStyleSheet("color: #00F0FF; font-family: 'JetBrains Mono', 'Consolas', monospace; font-weight: bold; font-size: 12px;")
        self.status_bar.addPermanentWidget(self.timer_label)
        
        self.candle_timer = QTimer(self)
        self.candle_timer.timeout.connect(self._update_countdown)
        self.candle_timer.start(1000)
        self._update_countdown()

    # ═══════════════════ LIVE DATA ═══════════════════

    def _update_countdown(self):
        import datetime
        from config import TIMEFRAMES
        
        # Use local time — avoid MT5 calls on main thread
        now = datetime.datetime.now()
        tf_input = self.timeframes_input.text().upper()
        tfs = [s for s in re.split(r'\s*,\s*', tf_input) if s]
        min_minutes = 60
        active_found = False
        for tf in tfs:
            if tf in TIMEFRAMES:
                val = TIMEFRAMES[tf].get("minutes", 60)
                if not active_found or val < min_minutes:
                    min_minutes = val
                    active_found = True
                    
        total_minutes = now.hour * 60 + now.minute
        next_boundary_total = ((total_minutes // min_minutes) + 1) * min_minutes
        minutes_to_add = next_boundary_total - total_minutes
        
        next_target = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=minutes_to_add)
        diff = int((next_target - now).total_seconds())
        if diff < 0: diff = 0
        h, rem = divmod(diff, 3600)
        m, s = divmod(rem, 60)
        
        self.timer_text_label.setText(f"  │  Next M{min_minutes}: ")
        self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _request_position_refresh(self):
        """Start a background thread to refresh MT5 positions (non-blocking)."""
        # Don't start a new worker if one is still running
        if self._mt5_worker is not None and self._mt5_worker.isRunning():
            return
        self._mt5_worker = MT5PositionWorker(self.magic_number_spin.value())
        self._mt5_worker.result_ready.connect(self._on_mt5_result)
        self._mt5_worker.start()

    def _on_mt5_result(self, data: dict):
        """Handle MT5 results on the main thread (safe for UI updates)."""
        try:
            if not data.get("connected"):
                return
            pos_list = data.get("positions", [])
            total_pnl = data.get("total_pnl", 0.0)
            balance = data.get("balance")

            if balance is not None:
                self.stat_balance.set_value(f"${balance:,.2f}")

            self.stat_positions.set_value(str(len(pos_list)), "#00F0FF" if pos_list else "#506080")

            pnl_color = "#10B981" if total_pnl >= 0 else "#FF2D78"
            self.stat_pnl.set_value(f"${total_pnl:+,.2f}", pnl_color)

            self._update_open_orders(pos_list)
        except Exception:
            pass

    @pyqtSlot(dict)
    def _update_open_orders(self, positions_data):
        """Update the Open Positions table."""
        if isinstance(positions_data, dict):
            positions_data = positions_data.get("positions", [])
        
        self.orders_table.setRowCount(len(positions_data))
        for row, pos in enumerate(positions_data):
            ticket_item = QTableWidgetItem(str(pos.get("ticket", "")))
            ticket_item.setForeground(QColor("#8892B0"))
            self.orders_table.setItem(row, 0, ticket_item)
            
            sym_item = QTableWidgetItem(pos.get("symbol", ""))
            sym_item.setForeground(QColor("#FFFFFF"))
            sym_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.orders_table.setItem(row, 1, sym_item)
            
            dir_text = pos.get("direction", "")
            dir_item = QTableWidgetItem(dir_text)
            dir_item.setForeground(QColor("#10B981") if dir_text == "BUY" else QColor("#FF2D78"))
            dir_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.orders_table.setItem(row, 2, dir_item)
            
            vol_item = QTableWidgetItem(str(pos.get("volume", "")))
            vol_item.setForeground(QColor("#C8D6E5"))
            self.orders_table.setItem(row, 3, vol_item)
            
            entry_item = QTableWidgetItem(f"{pos.get('entry', 0):.5f}")
            entry_item.setForeground(QColor("#C8D6E5"))
            self.orders_table.setItem(row, 4, entry_item)
            
            sl_item = QTableWidgetItem(f"{pos.get('sl', 0):.5f}")
            sl_item.setForeground(QColor("#FF2D78"))
            self.orders_table.setItem(row, 5, sl_item)
            
            tp_item = QTableWidgetItem(f"{pos.get('tp', 0):.5f}")
            tp_item.setForeground(QColor("#10B981"))
            self.orders_table.setItem(row, 6, tp_item)
            
            pnl_pips = pos.get("pnl_pips", 0)
            pnl_item = QTableWidgetItem(f"{pnl_pips:+.1f}")
            pnl_item.setForeground(QColor("#10B981") if pnl_pips >= 0 else QColor("#FF2D78"))
            pnl_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.orders_table.setItem(row, 7, pnl_item)
            
            pnl_usd = pos.get("pnl_usd", 0)
            usd_item = QTableWidgetItem(f"${pnl_usd:+.2f}")
            usd_item.setForeground(QColor("#10B981") if pnl_usd >= 0 else QColor("#FF2D78"))
            usd_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.orders_table.setItem(row, 8, usd_item)

    # ═══════════════════ CONFIG LOAD / SAVE ═══════════════════

    def load_configuration_into_ui(self):
        try:
            agent_config = self.config_manager.get_agent_config()
            
            self.model_combo.clear()
            for model in self.config_manager.models_data:
                display = model.get("display_name", model.get("model_id"))
                self.model_combo.addItem(display, model)
                
            import config
            default_symbols = config.SYMBOLS if hasattr(config, 'SYMBOLS') else ["EURUSD", "GBPUSD", "USDJPY"]
            self.symbols_input.setText(", ".join(agent_config.get("trading_symbols", default_symbols)))
            self.timeframes_input.setText(", ".join(agent_config.get("trading_timeframes", ["H1"])))
            self.bars_input.setText(str(agent_config.get("bars_to_fetch", 20000)))
            self.win_rate_input.setText(str(agent_config.get("min_win_rate", 0.40)))
            self.pf_input.setText(str(agent_config.get("min_profit_factor", 1.4)))
            self.trades_input.setText(str(agent_config.get("min_trades", 10)))
            self.start_hour_spin.setValue(agent_config.get("trade_start_hour", 4))
            self.end_hour_spin.setValue(agent_config.get("trade_end_hour", 22))
            self.risk_mode_combo.setCurrentText(agent_config.get("risk_mode", "Safe"))
            self.fixed_lot_input.setText(str(agent_config.get("fixed_lot", 0.01)))
            
            saved_path = os.getenv("MT4_DATA_PATH", "")
            if not saved_path:
                saved_path = r"C:\Program Files\Fusion Markets MetaTrader 5\terminal64.exe"
            self.mt4_path_input.setText(saved_path)
            
            if (last_model_id := agent_config.get("last_selected_model_id")):
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemData(i).get('model_id') == last_model_id:
                        self.model_combo.setCurrentIndex(i)
                        break
            
            self.model_combo.currentIndexChanged.connect(self._on_model_changed)
            if self.model_combo.count() > 0:
                self._on_model_changed(self.model_combo.currentIndex())
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load UI config: {e}")

    @pyqtSlot(int)
    def _on_model_changed(self, index):
        if index < 0: return
        model_data = self.model_combo.itemData(index)
        model_id = model_data.get("model_id")
        creds = self.config_manager.model_credentials.get(model_id, {})
        key = creds.get("api_key", "")
        if len(key) > 8:
            masked = key[:4] + "*" * (len(key)-8) + key[-4:]
            self.llm_api_key_input.setText(masked)
        else:
            self.llm_api_key_input.setText(key)
            
    @pyqtSlot(str)
    def _update_credentials_in_memory(self, text):
        if "*" in text: return
        idx = self.model_combo.currentIndex()
        if idx >= 0:
            m_id = self.model_combo.itemData(idx).get("model_id")
            if m_id not in self.config_manager.model_credentials:
                self.config_manager.model_credentials[m_id] = {}
            self.config_manager.model_credentials[m_id]["api_key"] = text

    def save_configuration(self):
        try:
            model_index = self.model_combo.currentIndex()
            agent_config = {
                "last_selected_model_id": self.model_combo.itemData(model_index).get('model_id') if model_index >= 0 else None,
                "trading_symbols": [s for s in re.split(r'\s*,\s*', self.symbols_input.text().strip().upper()) if s],
                "trading_timeframes": [s for s in re.split(r'\s*,\s*', self.timeframes_input.text().strip().upper()) if s],
                "bars_to_fetch": int(self.bars_input.text() or 20000),
                "min_win_rate": float(self.win_rate_input.text() or 0.40),
                "min_profit_factor": float(self.pf_input.text() or 1.4),
                "min_trades": int(self.trades_input.text() or 10),
                "trade_start_hour": self.start_hour_spin.value(),
                "trade_end_hour": self.end_hour_spin.value(),
                "risk_mode": self.risk_mode_combo.currentText(),
                "fixed_lot": float(self.fixed_lot_input.text() or 0.01)
            }
            self.config_manager.save_agent_config(agent_config)
            env_updates = {"MT4_DATA_PATH": self.mt4_path_input.text()}
            self.config_manager.save_env_and_credentials(env_updates, self.config_manager.model_credentials)
        except Exception as e:
            logger.error(f"Save Config Error: {e}")

    def get_current_config_from_ui(self):
        llm_api_key = self.llm_api_key_input.text()
        if llm_api_key and "*" not in llm_api_key:
            os.environ['TEMP_LLM_API_KEY'] = llm_api_key
            
        return {
            "mt4_data_path": self.mt4_path_input.text(),
            "enabled_model": self.model_combo.itemData(self.model_combo.currentIndex()) if self.model_combo.currentIndex() >= 0 else None,
            "trading_symbols": [s for s in re.split(r'\s*,\s*', self.symbols_input.text().strip()) if s],
            "trading_timeframes": [s for s in re.split(r'\s*,\s*', self.timeframes_input.text().strip().upper()) if s],
            "bars_to_fetch": int(self.bars_input.text() or 20000),
            "min_win_rate": float(self.win_rate_input.text() or 0.40),
            "min_profit_factor": float(self.pf_input.text() or 1.4),
            "min_trades": int(self.trades_input.text() or 10),
            "trade_start_hour": self.start_hour_spin.value(),
            "trade_end_hour": self.end_hour_spin.value(),
            "risk_mode": self.risk_mode_combo.currentText(),
            "fixed_lot": float(self.fixed_lot_input.text() or 0.01),
            "signal_only": self.signal_only_check.isChecked(),
            "llm_filter": self.llm_filter_check.isChecked(),
            "magic_number": self.magic_number_spin.value(),
            "trade_comment": self.trade_comment_input.text().strip() or "PT",
            "news_buffer_before": self.news_before_spin.value(),
            "news_buffer_after": self.news_after_spin.value(),
        }

    # ═══════════════════ ENGINE HANDLING ═══════════════════

    def start_trading(self):
        logger.info("MainWindow: 'RUN' button clicked.")
        if self.trading_task and not self.trading_task.done():
            return

        config = self.get_current_config_from_ui()
        if not config:
            return

        self.set_controls_enabled(False)
        self.log_browser.clear()
        
        self.chart_cache.clear()
        self.chart_selector.blockSignals(True)
        self.chart_selector.clear()
        self.chart_selector.addItem("— Select —")
        self.chart_selector.blockSignals(False)
        self.chart_widget.plot_widget.clear()
        self.equity_widget.clear()
        self.signal_table.setRowCount(0)

        # Inject loaded training cache if available
        if self._loaded_training_cache:
            config["loaded_training_cache"] = self._loaded_training_cache
            self._loaded_training_cache = None  # Clear after use — one-shot
        
        self.trading_task = self.loop.create_task(self.engine.start_with_loop(config, self.loop))

    def stop_trading(self):
        if self.trading_task and not self.trading_task.done():
            self.engine.stop()
            self.update_status_label("Stopping...", "status_label_pending")
            self.stop_button.setEnabled(False)

    @pyqtSlot()
    def on_engine_finished(self):
        self.set_controls_enabled(True)
        self.update_status_label("Stopped", "status_label_stopped")

    @pyqtSlot(dict)
    def _on_training_complete(self, validated_cache):
        """Enable Save Training button once training finishes."""
        n_pairs = len(validated_cache)
        n_patterns = sum(len(pats) for pats in validated_cache.values())
        self.save_training_btn.setEnabled(True)
        logger.info(f"Training complete: {n_pairs} pairs, {n_patterns} patterns. Save Training enabled.")

    def _save_training(self):
        """Save current validated patterns to a JSON file."""
        from PyQt5.QtWidgets import QFileDialog
        from engine import TradingEngine
        
        cache = self.engine._validated_cache
        if not cache:
            QMessageBox.warning(self, "No Data", "No training data to save. Run training first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Training State", "training_state.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        
        # Build session info for metadata
        session_info = {
            "symbols": [s for s in re.split(r'\s*,\s*', self.symbols_input.text().strip().upper()) if s],
            "timeframes": [s for s in re.split(r'\s*,\s*', self.timeframes_input.text().strip().upper()) if s],
            "bars_to_fetch": int(self.bars_input.text() or 20000),
            "min_win_rate": float(self.win_rate_input.text() or 0.40),
            "min_profit_factor": float(self.pf_input.text() or 1.4),
            "min_trades": int(self.trades_input.text() or 10),
        }
        
        try:
            TradingEngine.save_training_state(filepath, cache, session_info)
            n_pairs = len(cache)
            n_patterns = sum(len(p) for p in cache.values())
            QMessageBox.information(self, "Saved", f"Training state saved!\n\n{n_pairs} symbol/TF pairs\n{n_patterns} validated patterns\n\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save training state:\n{e}")

    def _load_training(self):
        """Load a previously saved training state from a JSON file."""
        from PyQt5.QtWidgets import QFileDialog
        from engine import TradingEngine
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Training State", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        
        try:
            loaded_cache, session_info = TradingEngine.load_training_state(filepath)
            if not loaded_cache:
                QMessageBox.warning(self, "Empty File", "The file contains no training data.")
                return
            
            self._loaded_training_cache = loaded_cache
            
            # Show summary
            symbols = set()
            tfs = set()
            total_patterns = 0
            for (sym, tf), pats in loaded_cache.items():
                symbols.add(sym)
                tfs.add(tf)
                total_patterns += len(pats)
            
            saved_at = session_info.get("saved_at", "Unknown") if isinstance(session_info, dict) else "Unknown"
            summary = f"✅ Loaded: {len(symbols)} symbols, {len(tfs)} TFs, {total_patterns} patterns"
            self.loaded_state_label.setText(summary)
            
            QMessageBox.information(
                self, "Training Loaded",
                f"Loaded training state from:\n{filepath}\n\n"
                f"Symbols: {', '.join(sorted(symbols))}\n"
                f"Timeframes: {', '.join(sorted(tfs))}\n"
                f"Validated Patterns: {total_patterns}\n\n"
                f"These pairs will skip training on next LAUNCH.\n"
                f"Add new symbols/TFs and they will be trained fresh."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load training state:\n{e}")

    @pyqtSlot(str, str, object, object, list)
    def _cache_chart_data(self, symbol, tf, df, patterns, trades=None):
        key = f"{symbol} ({tf})"
        self.chart_cache[key] = (df, patterns, symbol, trades or [])
        if self.chart_selector.findText(key) == -1:
            self.chart_selector.addItem(key)
        if self.chart_selector.currentText() == key:
            self.chart_widget.update_chart(df, patterns, symbol=symbol, trades=trades)
        elif self.chart_selector.currentIndex() == 0:
            self.chart_selector.setCurrentText(key)

    @pyqtSlot(int)
    def _on_chart_selector_changed(self, index):
        if index <= 0: return
        key = self.chart_selector.itemText(index)
        if key in self.chart_cache:
            df, patterns, sym, trades = self.chart_cache[key]
            self.chart_widget.update_chart(df, patterns, symbol=sym, trades=trades)

    @pyqtSlot(list, list)
    def _update_equity_curve(self, trade_indices, cum_pnls):
        self.equity_widget.clear()
        if not trade_indices or not cum_pnls:
            return
        
        total_pnl = cum_pnls[-1]
        n_trades = len(cum_pnls)
        color = '#00F0FF' if total_pnl >= 0 else '#FF2D78'
        
        self.equity_widget.setTitle(
            f"Cumulative: {total_pnl:+.1f} pips │ {n_trades} trades",
            color='#C8D6E5', size='11pt'
        )
        self.equity_widget.setLabel('left', 'PnL (pips)', color='#506080')
        self.equity_widget.setLabel('bottom', 'Trade #', color='#506080')
            
        curve = pg.PlotDataItem(trade_indices, cum_pnls, pen=pg.mkPen(color, width=2))
        fill_color = (0, 240, 255, 30) if total_pnl >= 0 else (255, 45, 120, 30)
        curve.setFillBrush(pg.mkBrush(color=fill_color))
        curve.setFillLevel(0)
        
        self.equity_widget.addItem(curve)
        baseline = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#1E2240', style=Qt.DashLine))
        self.equity_widget.addItem(baseline)

    @pyqtSlot(list)
    def _update_trade_list(self, trades_data):
        trades_data.sort(key=lambda t: t.get("date", ""), reverse=True)
        
        self.trade_table.setRowCount(len(trades_data))
        for row, t in enumerate(trades_data):
            self.trade_table.setItem(row, 0, QTableWidgetItem(str(t.get("date", ""))))
            self.trade_table.setItem(row, 1, QTableWidgetItem(str(t.get("symbol", ""))))
            self.trade_table.setItem(row, 2, QTableWidgetItem(str(t.get("pattern", ""))))
            self.trade_table.setItem(row, 3, QTableWidgetItem("BUY" if t.get("direction", 1) > 0 else "SELL"))
            self.trade_table.setItem(row, 4, QTableWidgetItem(f"{t.get('entry', 0):.5f}"))
            self.trade_table.setItem(row, 5, QTableWidgetItem(f"{t.get('exit', 0):.5f}"))
            
            pnl = t.get("pnl", 0)
            pnl_item = QTableWidgetItem(f"{pnl:+.1f}")
            pnl_item.setForeground(QColor('#10B981') if pnl >= 0 else QColor('#FF2D78'))
            self.trade_table.setItem(row, 6, pnl_item)
            
            outcome = t.get("outcome", "")
            outcome_item = QTableWidgetItem(outcome)
            if outcome == "TP":
                outcome_item.setForeground(QColor('#10B981'))
            elif outcome == "SL":
                outcome_item.setForeground(QColor('#FF2D78'))
            self.trade_table.setItem(row, 7, outcome_item)
        
        total_pnl = sum(t.get("pnl", 0) for t in trades_data)
        self.chart_tabs.setTabText(2, f"📋  Backtest ({len(trades_data)}) │ {total_pnl:+.0f} pips")
        self.stat_patterns.set_value(str(len(trades_data)))

    @pyqtSlot(list)
    def _update_signal_table(self, signals_data):
        """Update the LLM Signals tab with approved/rejected signals."""
        # Append new signals to existing rows
        for sig in signals_data:
            row = self.signal_table.rowCount()
            self.signal_table.insertRow(row)
            
            time_item = QTableWidgetItem(str(sig.get("time", "")))
            time_item.setForeground(QColor("#8892B0"))
            self.signal_table.setItem(row, 0, time_item)
            
            sym_item = QTableWidgetItem(sig.get("symbol", ""))
            sym_item.setForeground(QColor("#FFFFFF"))
            sym_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.signal_table.setItem(row, 1, sym_item)
            
            pat_item = QTableWidgetItem(sig.get("pattern", ""))
            pat_item.setForeground(QColor("#C8D6E5"))
            self.signal_table.setItem(row, 2, pat_item)
            
            direction = sig.get("direction", "")
            dir_item = QTableWidgetItem(direction)
            dir_item.setForeground(QColor("#10B981") if direction == "BUY" else QColor("#FF2D78"))
            dir_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.signal_table.setItem(row, 3, dir_item)
            
            entry_item = QTableWidgetItem(f"{sig.get('entry', 0):.5f}")
            entry_item.setForeground(QColor("#C8D6E5"))
            self.signal_table.setItem(row, 4, entry_item)
            
            sl_item = QTableWidgetItem(f"{sig.get('sl', 0):.5f}")
            sl_item.setForeground(QColor("#FF2D78"))
            self.signal_table.setItem(row, 5, sl_item)
            
            tp_item = QTableWidgetItem(f"{sig.get('tp', 0):.5f}")
            tp_item.setForeground(QColor("#10B981"))
            self.signal_table.setItem(row, 6, tp_item)
            
            risk_item = QTableWidgetItem(f"{sig.get('risk_pct', 0):.1f}%")
            risk_item.setForeground(QColor("#FFB800"))
            self.signal_table.setItem(row, 7, risk_item)
            
            verdict = sig.get("verdict", "")
            is_approved = any(verdict.upper().startswith(v) for v in ["APPROVED", "BUY", "SELL", "TAKE"])
            verdict_item = QTableWidgetItem(verdict)
            verdict_item.setForeground(QColor("#10B981") if is_approved else QColor("#FF2D78"))
            verdict_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.signal_table.setItem(row, 8, verdict_item)
        
        # Update tab title with counts
        total = self.signal_table.rowCount()
        self.chart_tabs.setTabText(3, f"🧠  Signals ({total})")
        
        # Forward approved signals to the chart for arrow markers
        approved_signals = [s for s in signals_data
                            if s.get("direction", "").upper() in ("BUY", "SELL")]
        if approved_signals:
            self.chart_widget.draw_llm_signals(approved_signals)

    def set_controls_enabled(self, enabled):
        self.run_button.setEnabled(enabled)
        self.stop_button.setEnabled(not enabled)
        for groupbox in self.config_pane.findChildren(QGroupBox):
            if groupbox.title() != "Execution":
                groupbox.setEnabled(enabled)
        if enabled:
            self.update_status_label("Stopped", "status_label_stopped")
    
    @pyqtSlot(list)
    def _update_pattern_results(self, results_data):
        """Populate Pattern Results table with validated pattern stats."""
        for r in results_data:
            row = self.pattern_results_table.rowCount()
            self.pattern_results_table.insertRow(row)
            
            items = [
                r.get("symbol", ""),
                r.get("pattern", ""),
                str(r.get("trades", 0)),
                f"{r.get('win_rate', 0):.1%}",
                f"{r.get('profit_factor', 0):.2f}",
                f"{r.get('cumul_balance', 0):.1f}",
                f"{r.get('avg_gain', 0):.2f}",
                f"{r.get('max_dd', 0):.1f}"
            ]
            
            wr = r.get('win_rate', 0)
            if wr >= 0.55:
                row_color = QColor("#10B981")
            elif wr >= 0.50:
                row_color = QColor("#F59E0B")
            else:
                row_color = QColor("#FF2D78")
            
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setForeground(row_color)
                if col >= 2:  # Numeric columns — right align
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.pattern_results_table.setItem(row, col, item)

    @pyqtSlot(str, str)
    def update_status_label(self, message, style_class):
        self.status_label.setText(message)
        self.status_label.setObjectName(style_class)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    @pyqtSlot(bool)
    def update_connection_status(self, connected):
        if connected:
            self.connection_label.setText("Connected")
            self.connection_label.setObjectName("status_label_ok")
        else:
            self.connection_label.setText("Disconnected")
            self.connection_label.setObjectName("status_label_error")
        self.connection_label.style().unpolish(self.connection_label)
        self.connection_label.style().polish(self.connection_label)

    @pyqtSlot(list)
    def append_log_message(self, logs):
        for log_data in logs:
            msg = log_data.get('message', '')
            log_type = log_data.get('type')
            content = log_data.get('content', {})
            
            html = ""
            if log_type == "observation":
                text = content.get("message", msg)
                html = f"<div style='margin-bottom:5px;'><font color='#00F0FF'><b>▸ SYS</b></font> <font color='#C8D6E5'>{text}</font></div>"
            elif log_type == "trade_action":
                action = content.get('action', 'N/A')
                symbol = content.get('tool_args', {}).get('symbol', 'N/A')
                lots = content.get('tool_args', {}).get('lots', 'N/A')
                html = (
                    f"<div style='margin:4px 0; padding:8px 12px; background-color: rgba(16,185,129,0.1); "
                    f"border-radius: 6px; border-left: 3px solid #10B981;'>"
                    f"<font color='#10B981' size='3'><b>⚡ {action}</b></font><br/>"
                    f"<font color='#E2E8F0'>{symbol}</font> │ "
                    f"<font color='#8892B0'>{lots}</font></div>"
                )
            elif log_type == "final_answer":
                text = content.get("text", msg)
                html = (
                    f"<div style='margin:3px 0; padding:6px 12px; background-color: rgba(0,240,255,0.06); "
                    f"border-radius: 4px; border-left: 2px solid #00F0FF;'>"
                    f"<font color='#00F0FF'>🧠</font> <font color='#C8D6E5'>{text}</font></div>"
                )
            else:
                level = log_data.get("level", "INFO").upper()
                if level == "INFO":
                    html = f"<div style='margin-bottom:3px;'><font color='#8892B0'>ℹ {msg}</font></div>"
                elif level == "WARNING":
                    html = f"<div style='margin-bottom:3px;'><font color='#FFB800'>⚠ {msg}</font></div>"
                elif level in ["ERROR", "CRITICAL"]:
                    html = f"<div style='margin-bottom:3px;'><font color='#FF2D78'>✖ {msg}</font></div>"
            
            if html:
                self.log_browser.append(html)
        self.log_browser.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        self.save_configuration()
        if self.trading_task and not self.trading_task.done():
            self.stop_trading()
        event.accept()
