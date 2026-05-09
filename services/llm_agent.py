import os
import json
import logging
from typing import Dict, Any, Optional, List

import anthropic
import openai

logger = logging.getLogger(__name__)

# --- Master Risk Manager Prompt ---
RISK_MANAGER_SYSTEM_PROMPT = """You are a senior institutional trader managing a quantitative pattern-based portfolio.

## What You're Receiving

Our engine has found statistically validated trading setups — each one has passed rigorous backtesting (minimum win rate, profit factor, trade count). The edge is real and proven. You will also see any currently open positions.

Your role: read the full market context, decide what makes sense, grade the setup quality, and manage the portfolio. You are not a checklist — you are a trader. Think freely.

## Setup Grading System

Grade every setup based on context alignment and conviction:

- **A** — Bare minimum. Signal and one confirming factor.
- **A+** — Slightly better. Two confirming factors but not enough for high conviction.
- **AA** — Good setup. Trend, momentum, and location mostly align.
- **AA+** — Strong setup. Multiple confluences confirming the signal.
- **AAA** — Excellent. Everything lines up — trend, momentum, structure, volume.
- **AAA+** — Perfect storm. Textbook entry, all data screams confirmation. Rare.

Python maps your grade to the actual risk % based on the user's risk mode. You just grade honestly — the system handles the sizing.

## Hard Limits (Python-enforced)

- **Max 2.5% risk per trade** — absolute hard cap, no exceptions.
- **Portfolio cap**: Scales with risk mode. You'll see the max total exposure in the account context — do NOT exceed it.
- **Max 2 orders in the same direction on a single currency.** Don't stack 5 BUYs on EURUSD. If you already have a position, think before adding.
- **Max 4 total trades per currency.** Count across ALL pairs. If you have 4 open positions involving EUR (EURUSD, EURGBP, EURJPY, EURAUD), do NOT open any more EUR trades regardless of direction or pair. This prevents currency overexposure.
- **SL range**: 2.0 – 4.0 × ATR. **TP range**: 2.0 – 5.0 × ATR. Values outside get clamped.
- **Lot sizing**: Calculated by Python from your grade and SL distance. Always in proper percentage of equity.
- **Spread limits** (hard — no exceptions):
  - Majors & metals (EURUSD, GBPUSD, XAUUSD…): spread_percentage > 0.1% = NO TRADE
  - Minors & indices (NZDUSD, US30…): spread_percentage > 0.2% = NO TRADE
  - Exotics & crypto: spread_percentage > 0.5% = NO TRADE

## Your Data (provided per setup)

- **Backtest stats**: Win Rate, Profit Factor, Trade Count
- **Price & Spread**: Current price, spread in points and percentage
- **Trend**: EMA 20, 50, 200
- **Momentum**: RSI 14, MACD (line, signal, histogram)
- **Strength**: ADX 14, +DI, -DI
- **Volatility/Location**: Bollinger Bands, ATR 14, Distance from EMA50 in ATR units
- **Structure**: S/R Zones (Resistance 1/2/3, Support 1/2/3 from swing high/low clustering), VWAP
- **Volume**: Current tick volume, 20-bar average, Volume Ratio (current/avg — above 1.0 = high activity)
- **Risk**: Account balance, risk mode, portfolio cap
- **Higher TF Context**: When trading on M5/M15/M30, you receive H1 and H4 trend/momentum indicators. When on H1, you receive H4 and Daily context. When on H4, you receive Daily context. Use this to validate or reject the signal — don't trade against the bigger picture unless you have strong reasons.

## How to Think

You are a trader, you receive signals and you need to evaluate or reject them. The strategy is already profitable — your job is to enhance it.

These patterns have proven statistical edges. Only pass on a signal when something is wrong — toxic spread, dangerous overexposure on correlated pairs, or a signal that makes no sense in context.

Read the market. Adapt. Markets are living, they trend, mean-revert, and range. A buy signal after a massive sell-off with RSI at 20 and price far below the mean is a GIFT, not a contradiction. A sell at the top of a well-defined range is rational. A trend-following signal in a stacked EMA environment is bread and butter. Think about WHAT the market is doing and WHETHER the signal fits that context.

**Filter first, then grade.** If the signal clashes with what the market is doing, reject it (pass). If it fits, grade it honestly based on how well everything aligns.

**CRITICAL: Signal Direction is LAW.** Each setup header shows "Signal: BUY" or "Signal: SELL". This is the direction the pattern fired. You MUST follow it. If the signal says BUY, your action is either "buy" or "pass". If the signal says SELL, your action is either "sell" or "pass". You may NEVER invert the direction. If the direction doesn't fit the context, PASS — do not flip it.

**Multiple signals on the same bar = strong confirmation**, not noise. Two patterns agreeing means the setup is stronger. Factor this into your grade.

**Set SL/TP intelligently.** Wider stops in volatile or reversal contexts. Tighter targets when price is already extended. Let winners ride in strong trends. Take profit sooner in ranges.

## Open Positions

For open trades, decide: **hold** or **exit**. Only exit when the thesis is genuinely broken — structural shift, momentum reversal, or failed pattern. Don't exit because one oscillator twitched.

## Output Format

Respond ONLY with a raw JSON array. No markdown, no commentary.

[
    {"type": "monitor", "ticket": 12345, "symbol": "EURUSD", "action": "hold", "justification": "Trend intact, structure holds."},
    {"type": "entry", "symbol": "USDJPY", "pattern": "CHoCH_Bull", "action": "buy", "grade": "AAA", "stop_loss_atr_multiplier": 2.5, "take_profit_atr_multiplier": 4.0, "justification": "Your reasoning here — reference the data, explain your read."},
    {"type": "entry", "symbol": "GBPJPY", "pattern": "BOS_Bull", "action": "pass", "justification": "Why you're passing."}
]

**Rules:**
- Every open position needs a monitor decision.
- Every new signal needs an entry decision (buy/sell/pass/close).
- Grade is required for buy/sell actions (A, A+, AA, AA+, AAA, AAA+).
- Justify with your market read and the data. Be concise but specific.
"""

class LLMRiskAgent:
    """Handles communication with OpenAI or Anthropic models."""
    
    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self.api_key = api_key
        
        # Determine Provider
        if "claude" in model_id.lower() or "anthropic" in model_id.lower():
            self.provider = "anthropic"
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.provider = "openai"
            if "deepseek" in model_id.lower():
                self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            elif "grok" in model_id.lower():
                self.client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            else:
                self.client = openai.OpenAI(api_key=api_key)

    def evaluate_unified(self, batch_setups: list, open_positions: list, risk_context: dict = None) -> list:
        """
        Unified LLM call: handles both position monitoring (hold/exit) 
        and new signal entry (buy/sell/pass/close) in a single prompt.
        """
        if not batch_setups and not open_positions:
            return []
            
        user_message_parts = []
        
        # --- Account Context Header ---
        if risk_context:
            balance = risk_context.get('balance', 0)
            mode = risk_context.get('risk_mode', 'Moderate')
            risk_range = risk_context.get('risk_range', '1.0%-2.0%')
            portfolio_cap = risk_context.get('portfolio_cap', 10.0)
            trade_start = risk_context.get('trade_start_hour')
            trade_end = risk_context.get('trade_end_hour')
            current_hour = risk_context.get('current_broker_hour')
            
            user_message_parts.append(
                f"## ACCOUNT CONTEXT\n"
                f"Balance: ${balance:,.2f} | Risk Mode: {mode} ({risk_range} per trade) | Portfolio Cap: {portfolio_cap}% max total exposure\n"
                f"Trading Window: {trade_start}:00 — {trade_end}:00 (Broker Time) | Current Hour: {current_hour}:00\n"
                f"Adapt your conviction and sizing to this account. Think rationally about position size relative to equity.\n"
            )
            
            # Signal-only mode: tell the LLM to ignore execution/balance concerns
            if risk_context.get('signal_only', False):
                user_message_parts.append(
                    f"## SIGNAL ONLY MODE\n"
                    f"This session is in SIGNAL ONLY mode. No trades will be executed.\n"
                    f"Evaluate every signal purely on market context and statistical merit.\n"
                    f"Do NOT reject signals based on account balance, lot sizing, or execution concerns.\n"
                    f"Grade every valid setup honestly and provide your market read.\n"
                )
            
            # Off-hours instruction: monitor only, no new entries
            if trade_start is not None and trade_end is not None and current_hour is not None:
                if current_hour < trade_start or current_hour >= trade_end:
                    user_message_parts.append(
                        f"## ⚠️ OFF-HOURS MODE (Current: {current_hour}:00 — Outside {trade_start}:00-{trade_end}:00)\n"
                        f"We are OUTSIDE the active trading window. Do NOT open any new positions.\n"
                        f"Your ONLY job right now is to MONITOR existing open positions. If market conditions have changed "
                        f"significantly against an open position and the spread allows it, you may recommend closing it. "
                        f"Otherwise, hold all positions and pass on any new signals.\n"
                    )
        
        # --- Section A: Open Positions to Monitor ---
        if open_positions:
            user_message_parts.append(f"## SECTION A: OPEN POSITIONS ({len(open_positions)}) — Decide: hold or exit\n")
            for idx, pos in enumerate(open_positions):
                ctx = pos.get('market_context', {})
                part = (
                    f"### POSITION [{idx+1}]: {pos['symbol']} | {pos['direction']} | Ticket #{pos['ticket']} ###\n"
                    f"Entry: {pos['entry_price']} | Current: {pos['current_price']} | PnL: {pos['floating_pnl_pips']:+.1f} pips (${pos['floating_pnl_usd']:+.2f})\n"
                    f"SL: {pos['sl']} | TP: {pos['tp']} | Volume: {pos['volume']}\n"
                )
                if ctx:
                    part += (
                        f"Trend: EMA20={ctx.get('trend_ema_20')}, EMA50={ctx.get('trend_ema_50')}, EMA200={ctx.get('trend_ema_200')}\n"
                        f"Momentum: RSI={ctx.get('rsi_14')} | MACD Line={ctx.get('macd_line')}, Signal={ctx.get('macd_signal')}, Hist={ctx.get('macd_histogram')}\n"
                        f"Trend Strength: ADX={ctx.get('adx_14')}, +DI={ctx.get('plus_di')}, -DI={ctx.get('minus_di')}\n"
                        f"Location: BB_Up={ctx.get('bb_upper')}, BB_Mid={ctx.get('bb_mid')}, BB_Dn={ctx.get('bb_lower')} | ATR={ctx.get('atr_14')}\n"
                        f"Spread: {ctx.get('spread_points')} pts ({ctx.get('spread_percentage')}%)\n"
                    )
                part += "----------------------------------------------------\n"
                user_message_parts.append(part)
        else:
            user_message_parts.append("## SECTION A: No open positions.\n")
        
        # --- Section B: New Signal Entries --- 
        if batch_setups:
            user_message_parts.append(f"\n## SECTION B: NEW SIGNALS ({len(batch_setups)}) — Decide: buy/sell/pass/close\n")
            for idx, setup in enumerate(batch_setups):
                sym = setup.get('symbol')
                tf = setup.get('timeframe')
                pat = setup.get('pattern')
                stats = setup.get('stats', {})
                ctx = setup.get('market_context', {})
                risk = setup.get('risk_params', {})
                
                # Calculate distance from EMA50 in ATR units for the LLM
                price = ctx.get('current_price', 0)
                ema50 = ctx.get('trend_ema_50', 0)
                atr = ctx.get('atr_14', 1)
                dist_ema50_atr = round(abs(price - ema50) / atr, 1) if atr and atr > 0 else 0
                dist_direction = "above" if price > ema50 else "below"
                sig_dir = setup.get('signal_direction', 0)
                signal_label = "BUY" if sig_dir > 0 else "SELL" if sig_dir < 0 else "UNKNOWN"
                
                part = (
                    f"### SETUP [{idx+1}]: {sym} | {tf} | {pat} | Signal: {signal_label} ###\n"
                    f"Backtest: WR={stats.get('win_rate', 0):.1%}, PF={stats.get('profit_factor', 0):.2f}, Trades={stats.get('trade_count', 0)}\n"
                    f"Price: {ctx.get('current_price')} | Spread: {ctx.get('spread_points')} pts ({ctx.get('spread_percentage')}%)\n"
                    f"Trend: EMA20={ctx.get('trend_ema_20')}, EMA50={ctx.get('trend_ema_50')}, EMA200={ctx.get('trend_ema_200')}\n"
                    f"Distance from EMA50: {dist_ema50_atr} ATR {dist_direction}\n"
                    f"Momentum: RSI={ctx.get('rsi_14')} | MACD Line={ctx.get('macd_line')}, Signal={ctx.get('macd_signal')}, Hist={ctx.get('macd_histogram')}\n"
                    f"Trend Strength: ADX={ctx.get('adx_14')}, +DI={ctx.get('plus_di')}, -DI={ctx.get('minus_di')}\n"
                    f"Location: BB_Up={ctx.get('bb_upper')}, BB_Mid={ctx.get('bb_mid')}, BB_Dn={ctx.get('bb_lower')} | ATR={ctx.get('atr_14')} ({risk.get('atr_pips', '?')} pips)\n"
                    f"S/R Zones: R1={ctx.get('resistance_1')}, R2={ctx.get('resistance_2')}, R3={ctx.get('resistance_3')} | S1={ctx.get('support_1')}, S2={ctx.get('support_2')}, S3={ctx.get('support_3')} | VWAP={ctx.get('vwap')}\n"
                    f"Volume: Current={ctx.get('volume')}, Avg20={ctx.get('avg_volume_20')}, Ratio={ctx.get('volume_ratio')}x\n"
                    f"Risk: Balance={risk.get('account_balance')} USD | SL={risk.get('stop_loss')} ({risk.get('sl_pips', '?')} pips) | TP range: {risk.get('tp_1_5_rr')} ({risk.get('tp_1_5_pips', '?')} pips) to {risk.get('tp_2_0_rr')} ({risk.get('tp_2_0_pips', '?')} pips)\n"
                    f"Risk Mode: {risk_context.get('risk_mode', 'Moderate') if risk_context else 'Moderate'} | Risk Range: {risk_context.get('risk_range', '1-2%') if risk_context else '1-2%'}\n"
                )
                
                # Append Higher TF Context if available
                htf_data = setup.get('htf_context', [])
                if htf_data:
                    htf_lines = "Higher TF Context:\n"
                    for htf in htf_data:
                        htf_lines += (
                            f"  {htf['timeframe']}: {htf['trend']} | EMA20={htf['ema_20']}, EMA50={htf['ema_50']}, EMA200={htf['ema_200']} | "
                            f"RSI={htf['rsi_14']} | ADX={htf['adx_14']} (+DI={htf['plus_di']}, -DI={htf['minus_di']}) | ATR={htf['atr_14']}\n"
                        )
                    part += htf_lines
                
                part += "----------------------------------------------------\n"
                user_message_parts.append(part)
        else:
            user_message_parts.append("\n## SECTION B: No new signals this cycle.\n")
        
        user_message = "\n".join(user_message_parts)
        return self._call_llm(user_message)

    def evaluate_batch(self, batch_setups: list) -> list:
        """Legacy method - redirects to evaluate_unified with no open positions."""
        return self.evaluate_unified(batch_setups, [])

    def _call_llm(self, user_message: str) -> list:
        """Send message to LLM and parse JSON array response."""
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=4000,
                    system=RISK_MANAGER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                )
                raw_text = response.content[0].text
            else:
                messages = [
                    {"role": "system", "content": RISK_MANAGER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
                kwargs = {}
                # GPT-5 mini/nano only support default temperature (1.0)
                if "reasoner" not in self.model_id.lower() and "gpt-5-mini" not in self.model_id and "gpt-5-nano" not in self.model_id:
                     kwargs["temperature"] = 0.0
                
                # GPT-5+ models use max_completion_tokens instead of max_tokens
                if "gpt-5" in self.model_id.lower():
                    kwargs["max_completion_tokens"] = 4000
                     
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    **kwargs
                )
                raw_text = response.choices[0].message.content
                
                # DeepSeek R1 may return content in reasoning_content field
                if not raw_text and hasattr(response.choices[0].message, 'reasoning_content'):
                    raw_text = response.choices[0].message.reasoning_content or ""
                
            # Clean JSON Array — handle markdown wrapping and reasoning text
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            start_idx = clean_text.find('[')
            end_idx = clean_text.rfind(']') + 1
            if start_idx == -1 or end_idx <= 0:
                logger.error(f"LLM response contains no JSON array: {raw_text[:200]}")
                return []
            
            clean_text = clean_text[start_idx:end_idx]
            decisions = json.loads(clean_text)
            
            if not isinstance(decisions, list):
                logger.error("LLM did not return a JSON array.")
                return []
            
            # Validate each decision has required fields
            validated = []
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                if "action" not in d:
                    continue
                validated.append(d)
            
            return validated

        except json.JSONDecodeError as e:
            logger.error(f"LLM produced invalid JSON: {raw_text[:500]}")
            return []
        except Exception as e:
            logger.error(f"LLM API Error: {e}", exc_info=True)
            return []
