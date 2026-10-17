"""CLI pipeline for transcript sentiment and earnings event-study analysis."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .sentiment import score_lm, score_finbert
from .event_study import build_event_returns, summarize_event_study

def read_transcript(data_dir: Path, filename: str) -> str:
    path = data_dir / "transcripts" / filename
    if not path.exists():
        # also allow a transcript path relative to data_dir
        path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {filename} (looked in {data_dir / 'transcripts'})")
    return path.read_text(encoding="utf-8", errors="replace")

def download_prices(events: pd.DataFrame, output_path: Path, benchmark: str | None = None):
    import yfinance as yf
    tickers = sorted(set(events["ticker"].astype(str).str.upper()))
    if benchmark: tickers.append(benchmark.upper())
    start = (pd.to_datetime(events["event_date"]).min() - pd.Timedelta(days=10)).date().isoformat()
    end = (pd.to_datetime(events["event_date"]).max() + pd.Timedelta(days=20)).date().isoformat()
    rows = []
    for ticker in sorted(set(tickers)):
        hist = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if hist.empty: continue
        col = "Close"
        close = hist[col]
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        rows.extend({"date": idx.date().isoformat(), "ticker": ticker, "adj_close": float(value)}
                    for idx, value in close.dropna().items())
    pd.DataFrame(rows).to_csv(output_path, index=False)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs/run")
    ap.add_argument("--skip-finbert", action="store_true", help="Skip transformer scoring; useful for smoke tests")
    ap.add_argument("--model-name", default="ProsusAI/finbert")
    ap.add_argument("--benchmark", default=None, help="Benchmark ticker present in prices, e.g. SPY")
    ap.add_argument("--download-prices", action="store_true", help="Download adjusted prices via yfinance")
    args = ap.parse_args()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = data_dir / "events.csv"
    events = pd.read_csv(events_path)
    required = {"ticker", "event_date", "transcript_file"}
    if not required.issubset(events.columns):
        raise ValueError(f"{events_path} must include columns {sorted(required)}")
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    dict_path = data_dir / "lm_dictionary.csv"
    score_rows = []
    for _, ev in events.iterrows():
        text = read_transcript(data_dir, str(ev["transcript_file"]))
        row = ev.to_dict()
        row.update(score_lm(text, dict_path))
        if not args.skip_finbert:
            row.update(score_finbert(text, model_name=args.model_name))
        else:
            row.update({"finbert_positive": float("nan"), "finbert_negative": float("nan"),
                        "finbert_neutral": float("nan"), "finbert_score": float("nan"), "finbert_chunks": 0})
        score_rows.append(row)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(output_dir / "call_scores.csv", index=False)
    prices_path = data_dir / "prices.csv"
    if args.download_prices:
        download_prices(events, prices_path, args.benchmark)
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing {prices_path}. Supply date,ticker,adj_close or use --download-prices.")
    prices = pd.read_csv(prices_path)
    returns = build_event_returns(scores, prices, benchmark=args.benchmark)
    returns.to_csv(output_dir / "event_returns.csv", index=False)
    summary = summarize_event_study(returns)
    summary.to_csv(output_dir / "event_study_summary.csv", index=False)
    # Diagnostic scatter: score vs 5-session return, if both columns exist.
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for col, label in (("finbert_score", "FinBERT"), ("lm_score", "LM dictionary")):
        if col in returns.columns and "return_5d" in returns.columns:
            subset = returns[[col, "return_5d"]].dropna()
            if not subset.empty:
                ax.scatter(subset[col], subset["return_5d"], alpha=.75, label=label)
                plotted = True
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Call sentiment score")
    ax.set_ylabel("Forward 5-session return")
    ax.set_title("Earnings-call sentiment vs forward returns")
    if plotted: ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sentiment_vs_return.png", dpi=160)
    plt.close(fig)
    report = [
        "# Earnings Call Sentiment Analyzer — Run Report", "",
        f"- Events processed: **{len(scores)}**",
        f"- FinBERT enabled: **{not args.skip_finbert}**",
        f"- Benchmark: **{args.benchmark or 'not used'}**", "",
        "## Event-study summary", "",
        summary.to_markdown(index=False) if not summary.empty else "No summary rows available; check prices and sentiment columns.", "",
        "## Interpretation cautions", "",
        "- This is an observational event study, not evidence of causality.",
        "- Check event-time alignment and missing-price exclusions.",
        "- The reported one-sample t-tests do not correct for overlapping windows, clustering, or multiple testing.",
        "- Do not treat these results as investment advice."
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Done. Outputs saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
