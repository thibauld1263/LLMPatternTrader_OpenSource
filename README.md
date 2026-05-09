<div align="center">
  <h1>LLM Pattern Trader</h1>
  <p><strong>A trading engine bridging MetaTrader 5 with Large Language Models.</strong></p>

  <a href="https://buymeacoffee.com/omegatradinghub" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" >
  </a>
  <br><br>
</div>

## Overview

LLM Pattern Trader is a desktop trading application built with PyQt5 and Python. It connects directly to your local MetaTrader 5 terminal to scan markets for custom algorithmic patterns, evaluates them against technical indicators, and sends this context to an LLM to execute risk assessments and trade sizing.

If you find this open-source project useful, please consider supporting the development using the link above.

---

## Core Features

*   **MetaTrader 5 Integration:** Fetches OHLCV, historical data, and symbol properties directly from MT5.
*   **Pattern Recognition:** Scans the market continuously for custom candlestick and sequential patterns.
*   **Technical Context Engine:** Generates a market report for every signal including EMA, RSI, MACD, Bollinger Bands, ADX, ATR, and volume analytics.
*   **LLM Trade Validation:** The engine asks an AI to evaluate the technical context of a pattern before placing the trade, mapping the LLM's grade to dynamic fractional lot sizing.
*   **Session Filtering:** Disables patterns that statistically lose money during specific trading sessions and performs multi-timeframe trend alignments.
*   **News Blackout:** Pulls the economic calendar and suspends entries during high-impact events.

---

## Prerequisites

1.  **Windows OS** (Required for MetaTrader 5 integration).
2.  **MetaTrader 5** installed and logged into a broker (live or demo).
3.  **Python 3.9+** installed on your system.

---

## Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/OmegaTradingHub/LLMPatternTrader_OpenSource.git
cd LLMPatternTrader_OpenSource
```

**2. Create a Virtual Environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Environment Variables:**
Rename `.env.example` to `.env` or edit the existing `.env` file to include your own API keys.
```env
MT4_DATA_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"
OPENAI_API_KEY="your_openai_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
DEEPSEEK_API_KEY="your_deepseek_api_key"
XAI_API_KEY="your_xai_api_key"
```

**5. Prepare MT5:**
*   Open MetaTrader 5.
*   Go to `Tools -> Options -> Expert Advisors` and allow `Automated Trading` and `WebRequest`.
*   Ensure your MT5 is open.

---

## How to Run

Launch the trading engine UI by running:
```bash
python main_llm.py
```

### The Workflow:
1.  **Select Symbols & Timeframes:** Use the GUI to define the assets and durations you want to trade.
2.  **Scan & Validate:** The engine evaluates patterns against minimum Profit Factor & Win Rate thresholds.
3.  **Live Trading Phase:** When a whitelisted pattern fires, it compiles the technical context, sends it to the LLM API, sizes the lot, and pushes the trade to MT5.

---

## Disclaimer
**Trading involves significant risk to your capital.** This software is provided "as is" without warranty of any kind. This is a research and educational tool. Do not trade with money you cannot afford to lose.

---

<div align="center">
  <br>
  <b>Support the project:</b> <a href="https://buymeacoffee.com/omegatradinghub">buymeacoffee.com/omegatradinghub</a>
</div>
