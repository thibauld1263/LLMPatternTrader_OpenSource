import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPicture, QPainter, QColor
import pandas as pd
import numpy as np

# Global pyqtgraph options for dark mode
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', '#1A1D2A') # API Trader Background
pg.setConfigOption('foreground', '#D0D0D0') # API Trader Text

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data: pd.DataFrame):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.picture = QPicture()
        self.generate_picture()

    def generate_picture(self):
        p = QPainter(self.picture)
        if hasattr(p, 'setRenderHint'):
            p.setRenderHint(QPainter.Antialiasing)
            
        w = 0.3 # Width of candelstick body
        
        # We assume data index is an integer sequence for X-axis plotting
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            
            # Colors
            if row['close'] >= row['open']:
                # Bullish - Original Teal/Cyan (#25F4EE)
                p.setPen(pg.mkPen('#25F4EE'))
                p.setBrush(pg.mkBrush('#25F4EE'))
            else:
                # Bearish - Original Pink/Red (#FE2C55)
                p.setPen(pg.mkPen('#FE2C55'))
                p.setBrush(pg.mkBrush('#FE2C55'))
                
            # Draw wick
            p.drawLine(pg.QtCore.QPointF(i, row['low']), pg.QtCore.QPointF(i, row['high']))
            
            # Draw body
            if row['open'] == row['close']:
                p.drawLine(pg.QtCore.QPointF(i - w, row['open']), pg.QtCore.QPointF(i + w, row['close']))
            else:
                rect = pg.QtCore.QRectF(i - w, min(row['open'], row['close']), w * 2, abs(row['open'] - row['close']))
                p.drawRect(rect)
                
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class ChartWidget(QWidget):
    """
    Custom embedded PySide/PyQt candlestick chart using pyqtgraph
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        # We use a custom string-based Axis to collapse weekends/gaps
        class TimeAxisItem(pg.AxisItem):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.x_values = []
                self.x_strings = []
                
            def set_mapping(self, x_values, x_strings):
                self.x_values = x_values
                self.x_strings = x_strings
                
            def tickStrings(self, values, scale, spacing):
                strings = []
                for v in values:
                    idx = int(v)
                    if 0 <= idx < len(self.x_strings):
                        strings.append(self.x_strings[idx])
                    else:
                        strings.append("")
                return strings
                
        self.time_axis = TimeAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': self.time_axis})
        self.plot_widget.showGrid(x=True, y=True, alpha=0.1)
        
        # Hide default labels
        self.plot_widget.hideAxis('left')
        self.plot_widget.showAxis('right')
        
        layout.addWidget(self.plot_widget)
        self.candlestick_item = None
        self.pattern_markers = []

    def update_chart(self, df: pd.DataFrame, patterns: dict = None, symbol: str = None, trades: list = None):
        """
        Expects a pandas DataFrame with datetime index and open/high/low/close columns.
        Patterns should be a dict of {pattern_name: pd.Series(booleans)}
        Trades should be a list of trade dicts with entry_time, exit_time, entry_price, exit_price, etc.
        """
        self.plot_widget.clear()
        self.pattern_markers.clear()
        
        self._current_symbol = symbol  # Track which symbol this chart shows
        
        if df is None or df.empty:
            return
            
        # Slice to the last 5000 bars for full context and visibility
        display_bars = 5000
        if len(df) > display_bars:
            df = df.iloc[-display_bars:].copy()
        else:
            df = df.copy()
        
        self._last_df = df
        self._last_trades = trades  # Cache trades for redraws
            
        if patterns:
            # Slice patterns dict to match the dataframe length exactly
            sliced_patterns = {}
            for pat_name, signals in patterns.items():
                if len(signals) > display_bars:
                    sliced_patterns[pat_name] = signals.iloc[-display_bars:]
                else:
                    sliced_patterns[pat_name] = signals
            patterns = sliced_patterns
        
        # Build mapping for custom axis to collapse weekends
        x_indices = list(range(len(df)))
        x_strings = df.index.strftime('%m-%d %H:%M').tolist()
        self.time_axis.set_mapping(x_indices, x_strings)

        class DateCandlestickItem(pg.GraphicsObject):
            def __init__(self, df_data, indices):
                pg.GraphicsObject.__init__(self)
                self.data = df_data
                self.ts = indices
                self.picture = QPicture()
                self._generate()

            def _generate(self):
                p = QPainter(self.picture)
                # Width is just 0.4 relative to the integer index spacing (1.0)
                w = 0.4
                    
                for i in range(len(self.data)):
                    row = self.data.iloc[i]
                    x = self.ts[i] # This is now just the integer index
                    
                    if row['close'] >= row['open']:
                        p.setPen(pg.mkPen('#25F4EE'))
                        p.setBrush(pg.mkBrush('#1A1D2A')) # Hollow for bull (optional, but looks good)
                    else:
                        p.setPen(pg.mkPen('#25F4EE'))
                        p.setBrush(pg.mkBrush('#25F4EE')) # Solid for bear
                        
                    p.drawLine(pg.QtCore.QPointF(x, row['low']), pg.QtCore.QPointF(x, row['high']))
                    if row['open'] == row['close']:
                        p.drawLine(pg.QtCore.QPointF(x - w, row['open']), pg.QtCore.QPointF(x + w, row['close']))
                    else:
                        rect = pg.QtCore.QRectF(x - w, min(row['open'], row['close']), w * 2, abs(row['open'] - row['close']))
                        p.drawRect(rect)
                p.end()

            def paint(self, p, *args):
                p.drawPicture(0, 0, self.picture)

            def boundingRect(self):
                return pg.QtCore.QRectF(self.picture.boundingRect())

        # Load Candles
        self.candlestick_item = DateCandlestickItem(df, x_indices)
        self.plot_widget.addItem(self.candlestick_item)

        # --- Indicator Overlays (same as sent to LLM) ---
        try:
            from ta_utils import utils
            
            # EMA lines
            ema20 = utils.ema(df['close'], 20)
            ema50 = utils.ema(df['close'], 50)
            ema200 = utils.ema(df['close'], 200)
            
            self.plot_widget.addItem(pg.PlotDataItem(
                x_indices, ema20.values, pen=pg.mkPen('#FFD700', width=1), name='EMA 20'))
            self.plot_widget.addItem(pg.PlotDataItem(
                x_indices, ema50.values, pen=pg.mkPen('#FF8C00', width=1), name='EMA 50'))
            self.plot_widget.addItem(pg.PlotDataItem(
                x_indices, ema200.values, pen=pg.mkPen('#FFFFFF', width=1, style=Qt.DashLine), name='EMA 200'))
            
            # Bollinger Bands
            bb_up, bb_mid, bb_low = utils.bollinger(df, 20, 2.0)
            self.plot_widget.addItem(pg.PlotDataItem(
                x_indices, bb_up.values, pen=pg.mkPen('#888888', width=1, style=Qt.DashLine)))
            self.plot_widget.addItem(pg.PlotDataItem(
                x_indices, bb_low.values, pen=pg.mkPen('#888888', width=1, style=Qt.DashLine)))
        except Exception:
            pass  # Don't crash chart if indicators fail

        # Draw Pattern Markers
        if patterns:
            candle_range = (df['high'] - df['low']).mean()
            offset = candle_range * 0.6
            
            for pat_name, signals in patterns.items():
                # Get indices where signal != 0
                signal_idxs = signals[signals != 0].index
                
                for idx in signal_idxs:
                    if idx in df.index:
                        i = df.index.get_loc(idx)
                        x_val = x_indices[i]
                        sig_val = signals.loc[idx]
                        
                        if sig_val > 0:
                            # BUY: Green arrow pointing UP, placed BELOW the candle
                            y_val = df['low'].iloc[i] - offset
                            color = '#10B981'
                            arrow = pg.ArrowItem(pos=(x_val, y_val), angle=90, brush=color, pen=color)
                            text_anchor = (0.5, 0)
                            text_y = y_val - offset * 0.4
                        else:
                            # SELL: Red arrow pointing DOWN, placed ABOVE the candle
                            y_val = df['high'].iloc[i] + offset
                            color = '#F43F5E'
                            arrow = pg.ArrowItem(pos=(x_val, y_val), angle=-90, brush=color, pen=color)
                            text_anchor = (0.5, 1)
                            text_y = y_val + offset * 0.4
                        
                        self.plot_widget.addItem(arrow)
                        
                        text = pg.TextItem(pat_name, anchor=text_anchor, color=color)
                        text.setPos(x_val, text_y)
                        self.plot_widget.addItem(text)
        
        # Draw Backtest Trade Overlays (entry → exit lines)
        if trades:
            self._draw_backtest_trades(df, x_indices, trades)
        
        # Calculate Y-Range
        min_y = df['low'].min()
        max_y = df['high'].max()
        padding = (max_y - min_y) * 0.15
        
        self.plot_widget.setYRange(min_y - padding, max_y + padding)
        self.plot_widget.setXRange(x_indices[0], x_indices[-1] + 5)
        
        # Redraw any cached LLM signals for this symbol
        if hasattr(self, '_signals_by_symbol') and self._signals_by_symbol:
            self._draw_signal_markers(df, x_indices)

    def _draw_backtest_trades(self, df: pd.DataFrame, x_indices: list, trades: list):
        """Draw backtest trade entry/exit markers with connecting lines.
        
        Visual encoding:
        - Green (TP): winning trade, dashed line connecting entry → exit
        - Red (SL): losing trade, dashed line connecting entry → exit
        - Amber (TIMEOUT/OPEN): neutral outcome
        - Entry: circle marker with direction arrow
        - Exit: square marker
        - PnL label at exit point
        """
        if not trades:
            return

        df_index = df.index
        y_range = df['high'].max() - df['low'].min()
        label_offset = y_range * 0.015  # Offset for PnL labels

        for trade in trades:
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            direction = trade.get("direction", 0)
            outcome = trade.get("outcome", "")
            pnl = trade.get("pnl_pips", 0)

            if entry_time is None or entry_price <= 0:
                continue

            # Find X index for entry
            try:
                entry_loc = df_index.get_loc(entry_time)
                if isinstance(entry_loc, slice):
                    entry_loc = entry_loc.start
                elif isinstance(entry_loc, np.ndarray):
                    entry_loc = int(np.where(entry_loc)[0][0])
            except (KeyError, TypeError, IndexError):
                continue

            if entry_loc >= len(x_indices):
                continue

            # Pick color based on outcome
            if outcome == "TP":
                color = '#10B981'       # Green — winner
                line_alpha = 80
            elif outcome == "SL":
                color = '#FF2D78'       # Red — loser
                line_alpha = 80
            elif outcome == "TIMEOUT":
                color = '#F59E0B'       # Amber — timeout
                line_alpha = 60
            else:
                color = '#8892B0'       # Gray — open/unknown
                line_alpha = 50

            entry_x = x_indices[entry_loc]

            # Entry marker: small colored dot
            entry_scatter = pg.ScatterPlotItem(
                [entry_x], [entry_price],
                pen=pg.mkPen(color, width=1.5),
                brush=pg.mkBrush(color),
                size=8,
                symbol='t' if direction == 1 else 't3'  # upward / downward triangle
            )
            self.plot_widget.addItem(entry_scatter)

            # Draw connecting line and exit marker
            if exit_time is not None and exit_price > 0:
                try:
                    exit_loc = df_index.get_loc(exit_time)
                    if isinstance(exit_loc, slice):
                        exit_loc = exit_loc.start
                    elif isinstance(exit_loc, np.ndarray):
                        exit_loc = int(np.where(exit_loc)[0][0])
                except (KeyError, TypeError, IndexError):
                    continue

                if exit_loc >= len(x_indices):
                    continue

                exit_x = x_indices[exit_loc]

                # Connecting line (dashed, semi-transparent)
                line_color = QColor(color)
                line_color.setAlpha(line_alpha)
                line = pg.PlotDataItem(
                    [entry_x, exit_x], [entry_price, exit_price],
                    pen=pg.mkPen(line_color, width=1.2, style=Qt.DashLine)
                )
                self.plot_widget.addItem(line)

                # Exit marker: small square
                exit_scatter = pg.ScatterPlotItem(
                    [exit_x], [exit_price],
                    pen=pg.mkPen(color, width=1.5),
                    brush=pg.mkBrush(color),
                    size=7,
                    symbol='s'  # square
                )
                self.plot_widget.addItem(exit_scatter)

                # PnL label at exit
                pnl_text = f"{pnl:+.0f}"
                label = pg.TextItem(pnl_text, anchor=(0.5, 1.0 if pnl >= 0 else 0.0), color=color)
                label.setFont(pg.QtGui.QFont('Segoe UI', 7, pg.QtGui.QFont.Bold))
                label_y = exit_price + (label_offset if pnl >= 0 else -label_offset)
                label.setPos(exit_x, label_y)
                self.plot_widget.addItem(label)

    def draw_llm_signals(self, signals_data, symbol: str = None):
        """Cache and draw LLM signal markers. Called from main_window when signals arrive."""
        if not hasattr(self, '_signals_by_symbol'):
            self._signals_by_symbol = {}
        
        for sig in signals_data:
            sym = sig.get('symbol', '')
            if sym not in self._signals_by_symbol:
                self._signals_by_symbol[sym] = []
            self._signals_by_symbol[sym].append(sig)
        
        # If we have chart data, draw immediately for the current symbol
        if self.candlestick_item is not None:
            if hasattr(self, '_last_df') and self._last_df is not None:
                self._draw_signal_markers(self._last_df, list(range(len(self._last_df))))

    def _draw_signal_markers(self, df, x_indices):
        """Draw large glowing arrows for LLM-approved signals on the chart."""
        if not hasattr(self, '_signals_by_symbol'):
            return
        
        current_sym = getattr(self, '_current_symbol', None)
        if not current_sym:
            return
        
        signals_for_sym = self._signals_by_symbol.get(current_sym, [])
        if not signals_for_sym:
            return
            
        candle_range = (df['high'] - df['low']).mean()
        offset = candle_range * 1.2
        
        for sig in signals_for_sym:
            direction = sig.get('direction', '').upper()
            if direction not in ('BUY', 'SELL'):
                continue
                
            entry_price = sig.get('entry', 0)
            if entry_price <= 0:
                continue
            
            # Find the closest bar index to the entry price (place at latest bar)
            x_val = x_indices[-1] if x_indices else 0
            
            if direction == 'BUY':
                y_val = df['low'].iloc[-1] - offset
                color = '#10B981'
                arrow = pg.ArrowItem(pos=(x_val, y_val), angle=90, 
                                     brush=color, pen=pg.mkPen(color, width=2),
                                     headLen=20, headWidth=16, tailLen=10)
            else:
                y_val = df['high'].iloc[-1] + offset
                color = '#F43F5E'
                arrow = pg.ArrowItem(pos=(x_val, y_val), angle=-90,
                                     brush=color, pen=pg.mkPen(color, width=2),
                                     headLen=20, headWidth=16, tailLen=10)
            
            self.plot_widget.addItem(arrow)
            
            label = f"LLM: {sig.get('pattern', '')}"
            text = pg.TextItem(label, anchor=(0.5, 0.5), color=color)
            text.setFont(pg.QtGui.QFont('Segoe UI', 8, pg.QtGui.QFont.Bold))
            text.setPos(x_val, y_val - (offset * 0.3 if direction == 'BUY' else -offset * 0.3))
            self.plot_widget.addItem(text)

