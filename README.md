# weatherdecisions

Probabilistic weather forecasting (temperature, HDD/CDD, precipitation, wind) mapped to Kalshi prediction market contracts to generate trade signals for Austin, TX weather markets.

Built by Malachy O'Donnabhain, Mehdi Touhami, Zain Radwan, and Neel Dutta Gupta.

---

## What it does

Prediction markets like Kalshi price binary weather outcomes (e.g. "Will the high in Austin exceed 80°F today?"). Volume on these markets is low and participants are mostly retail — which means mispricings exist.

This system builds a probabilistic weather model that outputs a **fair value** for each contract. Rather than taking directional bets, it acts as a passive **market maker**: posting limit orders on both sides of fair value, and closing positions when price reverts.

When the market price deviates from our model's fair value, a limit order fills. When it returns, we close for a gain. The strategy is directionally neutral — it profits from the spread in either direction.

---

## How it works

**1. Probabilistic forecast**
Ingest NWS ensemble data and fit a normal distribution over the day's temperature outcomes. Estimate mu (expected temperature) and sigma (uncertainty). Derive HDD/CDD, precipitation probability, and wind probability.

**2. Fair value computation**
Use the normal CDF to compute the exact probability that temperature falls within each Kalshi contract's range. This probability is the model's fair value price for that contract.

**3. Signal generation**
Compare model fair value to Kalshi's current market price. If the deviation exceeds a threshold (accounting for costs and slippage), generate a trade signal.

**4. Market making**
Post passive limit orders in a grid around fair value. Size positions smaller when sigma is large (high forecast uncertainty). Close on return to fair value or at a gain threshold.

---

## Repo structure
```
weatherdecisions/
├── data/           # Raw and processed weather data
├── models/         # Trained model artifacts
├── scripts/        # Pipeline scripts (ingestion, modeling, signal gen)
├── Thesis.txt      # Strategy writeup and market-making logic
├── notes.txt       # Research notes and modeling approach
├── requirements.txt
└── README.md
```

---

## Setup
```bash
git clone https://github.com/neeldg/weatherdecisions.git
cd weatherdecisions
pip install -r requirements.txt
```

You will need a Kalshi API key set as an environment variable:
```bash
export KALSHI_API_KEY=your_key_here
```

---

## Dependencies

| Package | Use |
|---|---|
| `pandas` / `numpy` | Data wrangling, HDD/CDD calculation |
| `scikit-learn` | Distribution fitting, regression |
| `scipy` | Normal CDF for probability computation |
| `requests` | NWS and Kalshi API calls |
| `yfinance` | Energy proxy validation |

---

## Key design decisions

**Why market-make instead of directionally bet?**
A ±1°F temperature range contract is a lumpy bet. Being right 60% of the time still produces a volatile equity curve. Passive market making captures the spread in both directions and smooths returns.

**Why assume a normal distribution?**
Minute-to-minute temperature changes are approximately normal. The approximation is reasonable for D+1 forecasts but may underestimate fat tails during extreme weather events. A t-distribution or empirical quantile approach can be substituted if needed.

**Update frequency**
The pipeline is designed to re-run every 10–30 minutes, updating mu, sigma, and limit order prices as new data arrives.

---

## Limitations

- Normality is an approximation. Extreme heat or cold events can produce fatter tails than the model expects.
- Kalshi volumes are thin. Large positions will move the market against you.
- NWS ensemble data has a fixed release cadence — intraday model updates are limited by data availability.
- This is a research project, not a live trading system. No guarantees on execution quality or PnL.

---

## Not financial advice

This project is for research and educational purposes only.