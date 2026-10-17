"""Event returns and compact descriptive event-study statistics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = (1, 2, 3, 4, 5)

def build_event_returns(events: pd.DataFrame, prices: pd.DataFrame, benchmark: str | None = None) -> pd.DataFrame:
    """Forward close-to-close returns from event-date close to +1..+5 trading sessions."""
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p["ticker"] = p["ticker"].astype(str).str.upper()
    p["adj_close"] = pd.to_numeric(p["adj_close"], errors="coerce")
    p = p.dropna(subset=["date", "ticker", "adj_close"]).sort_values(["ticker", "date"])
    series = {t: g.set_index("date")["adj_close"].sort_index() for t, g in p.groupby("ticker")}
    benchmark_returns = None
    if benchmark:
        b = series.get(benchmark.upper())
        if b is not None:
            benchmark_returns = b.pct_change()
    out = []
    for _, event in events.iterrows():
        ticker = str(event["ticker"]).upper()
        date = pd.Timestamp(event["event_date"]).normalize()
        s = series.get(ticker)
        row = event.to_dict()
        row["ticker"] = ticker
        row["event_date"] = date
        row["price_event_date"] = pd.NaT
        if s is None or s.empty:
            for h in HORIZONS: row[f"return_{h}d"] = np.nan
            out.append(row); continue
        # If the event date is not a trading date, use the next available date. This is
        # explicit and reproducible; users should align event_date to reaction date.
        eligible = s.index[s.index >= date]
        if len(eligible) == 0:
            for h in HORIZONS: row[f"return_{h}d"] = np.nan
            out.append(row); continue
        event_trade_date = eligible[0]
        row["price_event_date"] = event_trade_date
        base = float(s.loc[event_trade_date])
        for h in HORIZONS:
            future = s.index[s.index > event_trade_date]
            if len(future) >= h and base:
                target_date = future[h - 1]
                ret = float(s.loc[target_date] / base - 1)
                row[f"return_{h}d"] = ret
                if benchmark_returns is not None:
                    # Approximate market-adjusted cumulative return by subtracting
                    # benchmark cumulative return over matching sessions.
                    b = series.get(benchmark.upper())
                    if b is not None and event_trade_date in b.index:
                        bf = b.index[b.index > event_trade_date]
                        if len(bf) >= h:
                            row[f"market_adjusted_return_{h}d"] = ret - float(b.loc[bf[h-1]] / b.loc[event_trade_date] - 1)
                        else: row[f"market_adjusted_return_{h}d"] = np.nan
                    else: row[f"market_adjusted_return_{h}d"] = np.nan
            else:
                row[f"return_{h}d"] = np.nan
                if benchmark: row[f"market_adjusted_return_{h}d"] = np.nan
        out.append(row)
    return pd.DataFrame(out)

def summarize_event_study(df: pd.DataFrame, score_columns=("finbert_score", "lm_score"),
                          adjusted: bool = False) -> pd.DataFrame:
    results = []
    for score_col in score_columns:
        if score_col not in df.columns: continue
        for h in HORIZONS:
            ret_col = (f"market_adjusted_return_{h}d" if adjusted else f"return_{h}d")
            if ret_col not in df.columns: continue
            sub = df[[score_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty: continue
            vals = sub[ret_col].astype(float)
            tstat, pval = stats.ttest_1samp(vals, 0.0) if len(vals) >= 2 else (np.nan, np.nan)
            results.append({"score": score_col, "horizon": f"{h}d", "n": len(vals),
                            "mean_return": vals.mean(), "median_return": vals.median(),
                            "hit_rate": (vals > 0).mean(), "t_stat": tstat, "p_value": pval,
                            "spearman_corr": sub[score_col].corr(vals, method="spearman") if len(vals) >= 3 else np.nan})
    return pd.DataFrame(results)
