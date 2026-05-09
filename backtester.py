"""
PatternTrader - Backtester
ATR-based SL/TP entries with real pip PnL tracking.
Saves combined equity curve per symbol/TF as PNG.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import config
import ta_utils as utils
from data_fetcher import MT5DataFetcher

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp = None
    direction: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl_pips: float = 0.0
    pattern: str = ""
    outcome: str = ""

@dataclass
class BacktestResult:
    symbol: str
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_gain: float = 0.0
    trade_count: int = 0
    max_drawdown: float = 0.0

class Backtester:

    def __init__(self, sl_mult=None, tp_mult=None, max_hold=50):
        self.sl_mult = sl_mult or config.SL_ATR_MULT
        self.tp_mult = tp_mult or config.TP_ATR_MULT
        self.max_hold = max_hold

    def run(self, df: pd.DataFrame, signals: pd.Series,
            symbol: str, pattern_name: str = "") -> BacktestResult:
        """Backtest a single signal series. Returns real pip PnL trades.
        Strict Next-Open Execution:
        - Signal at i-1 triggers trade entry at open[i].
        - ATR is also pulled from i-1 to avoid lookahead.
        - Exits (SL/TP) can be triggered on the same entry candle i.
        """
        pip_size, _ = MT5DataFetcher.pip_info(symbol)
        atr_vals = utils.atr(df)

        trades: List[Trade] = []
        in_trade = False
        trade = None
        trade_entry_i = -1

        open_vals = df["open"].values
        close_vals = df["close"].values
        high_vals = df["high"].values
        low_vals = df["low"].values
        idx = df.index
        sig_vals = signals.values
        atr_arr = atr_vals.values

        for i in range(1, len(df)):
            # 1. Open new trade based on PREVIOUS closed candle (i-1)
            if not in_trade and sig_vals[i-1] != 0:
                atr_val = atr_arr[i-1]
                if pd.notna(atr_val) and atr_val > 0:
                    direction = int(sig_vals[i-1])
                    entry = open_vals[i]
                    
                    if direction == 1:
                        sl = entry - atr_val * self.sl_mult
                        tp = entry + atr_val * self.tp_mult
                        # Enforce minimum 10-pip SL
                        if (entry - sl) < pip_size * 10:
                            sl = entry - pip_size * 10
                    else:
                        sl = entry + atr_val * self.sl_mult
                        tp = entry - atr_val * self.tp_mult
                        # Enforce minimum 10-pip SL
                        if (sl - entry) < pip_size * 10:
                            sl = entry + pip_size * 10
                        
                    trade = Trade(
                        entry_time=idx[i], direction=direction,
                        entry_price=entry, stop_loss=sl,
                        take_profit=tp, pattern=pattern_name,
                    )
                    in_trade = True
                    trade_entry_i = i

            # 2. Check exits (can occur on the SAME candle we just entered)
            if in_trade:
                if trade.direction == 1:
                    # Check SL first (pessimistic approach)
                    if low_vals[i] <= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = idx[i]
                        trade.outcome = "SL"
                        trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                        trades.append(trade)
                        in_trade = False
                        continue
                    if high_vals[i] >= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = idx[i]
                        trade.outcome = "TP"
                        trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                        trades.append(trade)
                        in_trade = False
                        continue
                else:
                    # Check SL first (pessimistic approach)
                    if high_vals[i] >= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = idx[i]
                        trade.outcome = "SL"
                        trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                        trades.append(trade)
                        in_trade = False
                        continue
                    if low_vals[i] <= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = idx[i]
                        trade.outcome = "TP"
                        trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                        trades.append(trade)
                        in_trade = False
                        continue

                # 3. Timeout check
                hold_bars = i - trade_entry_i
                if hold_bars >= self.max_hold:
                    trade.exit_price = close_vals[i]
                    trade.exit_time = idx[i]
                    trade.outcome = "TIMEOUT"
                    if trade.direction == 1:
                        trade.pnl_pips = (close_vals[i] - trade.entry_price) / pip_size
                    else:
                        trade.pnl_pips = (trade.entry_price - close_vals[i]) / pip_size
                    trades.append(trade)
                    in_trade = False
                    continue

        # Close any active open trade at the end of the data series
        if in_trade and trade is not None:
            trade.exit_price = close_vals[-1]
            trade.exit_time = idx[-1]
            trade.outcome = "OPEN"
            if trade.direction == 1:
                trade.pnl_pips = (close_vals[-1] - trade.entry_price) / pip_size
            else:
                trade.pnl_pips = (trade.entry_price - close_vals[-1]) / pip_size
            trades.append(trade)

        return self._summarize(trades, symbol)

    def run_all_patterns(self, df: pd.DataFrame, signals_df: pd.DataFrame,
                         symbol: str) -> Dict[str, BacktestResult]:
        import sys
        import itertools
        spinner = itertools.cycle(['-', '\\', '|', '/'])
        total_cols = len(signals_df.columns)
        
        results = {}
        for i, col in enumerate(signals_df.columns):
            sig = signals_df[col]
            if (sig != 0).sum() > 0:
                results[col] = self.run(df, sig, symbol, pattern_name=col)
            
            if sys.stdout:
                sys.stdout.write(f"\r[TRAINING/SCAN] {symbol} Testing Pattern: {col} ({i+1}/{total_cols}) {next(spinner)}   ")
                sys.stdout.flush()
            
        if sys.stdout:
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
        return results

    def _summarize(self, trades: List[Trade], symbol: str) -> BacktestResult:
        result = BacktestResult(symbol=symbol, trades=trades)
        if not trades:
            return result
        pnls = [t.pnl_pips for t in trades]
        result.trade_count = len(trades)
        result.total_pnl = sum(pnls)
        result.avg_gain = np.mean(pnls)
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        result.win_rate = len(winners) / len(pnls) if pnls else 0
        gp = sum(winners) if winners else 0
        gl = sum(abs(l) for l in losers) if losers else 0
        result.profit_factor = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0)
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        result.max_drawdown = (peak - equity).max() if len(pnls) > 0 else 0
        return result

    @staticmethod
    def results_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
        rows = []
        for name, r in results.items():
            rows.append({
                "Pattern": name,
                "Trades": r.trade_count,
                "Win Rate": round(r.win_rate, 4),
                "Profit Factor": round(r.profit_factor, 2),
                "Cumulative Balance": round(r.total_pnl, 1),
                "Avg Gain": round(r.avg_gain, 2),
                "Max DD": round(r.max_drawdown, 1),
            })
        return pd.DataFrame(rows).sort_values("Profit Factor", ascending=False)

    @staticmethod
    def combine_signals(signals_df: pd.DataFrame,
                        pattern_scores: dict = None) -> tuple:
        """Merge multiple pattern signal columns into ONE combined signal
        series with at most 1 signal per bar.

        If multiple patterns fire on the same bar, the one with the highest
        score (e.g. profit factor from backtest) wins. Ties broken by
        column order.

        Args:
            signals_df: DataFrame where each column is a pattern's signal
                        series (+1, -1, or 0).
            pattern_scores: Optional dict of {pattern_name: score}.
                           Higher score = higher priority.
                           If None, all patterns are equal priority.

        Returns:
            Tuple of (combined_signal: pd.Series, winning_pattern: pd.Series)
            - combined_signal: at most one non-zero value per bar
            - winning_pattern: name of the pattern that won each bar (empty string if no signal)
        """
        combined = pd.Series(0, index=signals_df.index, dtype=int)
        winner = pd.Series("", index=signals_df.index, dtype=str)

        if signals_df.empty:
            return combined, winner

        # Build priority-ordered list of columns
        cols = list(signals_df.columns)
        if pattern_scores:
            cols.sort(key=lambda c: pattern_scores.get(c, 0), reverse=True)

        # For each bar, take the first (highest-priority) non-zero signal
        for col in cols:
            mask = (combined == 0) & (signals_df[col] != 0)
            combined[mask] = signals_df[col][mask].astype(int)
            winner[mask] = col

        return combined, winner

    def run_combined(self, df: pd.DataFrame, signals_df: pd.DataFrame,
                     symbol: str, pattern_scores: dict = None) -> BacktestResult:
        """Backtest with a 1-order-per-candle-per-pair limit.

        Merges all pattern signals into a single series (highest PF wins
        when multiple fire on the same bar), then runs a standard backtest.
        Each trade is tagged with the pattern that actually triggered it.

        Args:
            df: OHLCV DataFrame.
            signals_df: DataFrame of all pattern signals.
            symbol: Trading symbol.
            pattern_scores: {pattern_name: profit_factor} for priority.

        Returns:
            BacktestResult for the combined signal.
        """
        combined, winners = self.combine_signals(signals_df, pattern_scores)
        result = self.run(df, combined, symbol, pattern_name="COMBINED")
        
        # Tag each trade with the actual winning pattern name
        for trade in result.trades:
            if trade.entry_time is not None:
                try:
                    idx = df.index.get_loc(trade.entry_time)
                    # Signal was on previous bar, entry on next open
                    if idx > 0:
                        pat_name = winners.iloc[idx - 1]
                        if pat_name:
                            trade.pattern = pat_name
                except (KeyError, IndexError):
                    pass
        
        return result

    @staticmethod
    def plot_combined_equity(results: Dict[str, BacktestResult],
                             symbol: str, tf_name: str,
                             save_dir: str = None) -> str:
        """Merge all pattern trades, plot cumulative PnL + drawdown, save PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = save_dir or config.RESULTS_DIR
        os.makedirs(save_dir, exist_ok=True)

        # Collect all trades sorted by exit time
        all_trades = []
        for _, r in results.items():
            for t in r.trades:
                if t.exit_time is not None:
                    all_trades.append(t)
        if not all_trades:
            return ""

        all_trades.sort(key=lambda t: t.exit_time)
        exits = [t.exit_time for t in all_trades]
        pnls = [t.pnl_pips for t in all_trades]
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum

        # Stats
        total = cum[-1]
        n = len(all_trades)
        w = sum(1 for p in pnls if p > 0)
        wr = w / n if n else 0
        gp = sum(p for p in pnls if p > 0)
        gl = sum(abs(p) for p in pnls if p < 0)
        pf = gp / gl if gl > 0 else float("inf")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        height_ratios=[3, 1],
                                        gridspec_kw={"hspace": 0.15})

        ax1.plot(exits, cum, color="#2196F3", lw=1.2, label="Cumulative PnL (pips)")
        ax1.fill_between(exits, 0, cum,
                         where=[c >= 0 for c in cum], color="#4CAF50", alpha=0.15)
        ax1.fill_between(exits, 0, cum,
                         where=[c < 0 for c in cum], color="#F44336", alpha=0.15)
        ax1.axhline(0, color="gray", ls="--", lw=0.5)
        txt = (f"PnL: {total:+,.0f} pips | Trades: {n} | "
               f"WR: {wr:.1%} | PF: {pf:.2f} | MaxDD: {dd.max():,.0f} pips")
        ax1.set_title(f"{symbol} {tf_name} - Combined Equity", fontsize=13, fontweight="bold")
        ax1.text(0.5, 0.02, txt, transform=ax1.transAxes, fontsize=9,
                 ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        ax1.set_ylabel("Cumulative PnL (pips)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(exits, 0, -dd, color="#F44336", alpha=0.4)
        ax2.plot(exits, -dd, color="#D32F2F", lw=0.8)
        ax2.set_ylabel("Drawdown (pips)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f"{symbol}_{tf_name}_equity.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    [OK] Equity: {path}")
        return path
