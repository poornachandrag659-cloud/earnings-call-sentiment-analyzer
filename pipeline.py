
"""CLI pipeline for transcript sentiment and earnings event-study analysis."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.event_study import build_event_returns, summarize_event_study
from src.sentiment import read_transcript, score_finbert, score_lm
from src.data import download_prices


def main():
    parser = argparse.ArgumentParser(
        description="Analyze earnings-call sentiment and event-study returns."
    )

    parser.add_argument(
        "--data-dir",
        default="data/sample",
        help="Directory containing events.csv, prices.csv and transcripts.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sample",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--skip-finbert",
        action="store_true",
        help="Skip FinBERT scoring.",
    )
    parser.add_argument(
        "--model-name",
        default="ProsusAI/finbert",
        help="Hugging Face FinBERT model name.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark ticker present in prices, e.g. SPY.",
    )
    parser.add_argument(
        "--download-prices",
        action="store_true",
        help="Download adjusted prices using yfinance.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate event metadata.
    events_path = data_dir / "events.csv"

    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")

    events = pd.read_csv(events_path)

    required = {"ticker", "event_date", "transcript_file"}

    if not required.issubset(events.columns):
        raise ValueError(
            f"{events_path} must include columns {sorted(required)}"
        )

    if events.empty:
        raise ValueError(f"No events found in {events_path}")

    events["event_date"] = pd.to_datetime(
        events["event_date"], errors="raise"
    ).dt.normalize()

    dict_path = data_dir / "lm_dictionary.csv"

    # Score each transcript.
    score_rows = []

    for _, event in events.iterrows():
        transcript = read_transcript(
            data_dir, str(event["transcript_file"])
        )

        row = event.to_dict()
        row.update(score_lm(transcript, dict_path))

        if not args.skip_finbert:
            row.update(
                score_finbert(
                    transcript,
                    model_name=args.model_name,
                )
            )
        else:
            row.update(
                {
                    "finbert_positive": float("nan"),
                    "finbert_negative": float("nan"),
                    "finbert_neutral": float("nan"),
                    "finbert_score": float("nan"),
                    "finbert_chunks": 0,
                }
            )

        score_rows.append(row)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(output_dir / "call_scores.csv", index=False)

    # Load or download price data.
    prices_path = data_dir / "prices.csv"

    if args.download_prices:
        data_dir.mkdir(parents=True, exist_ok=True)
        download_prices(events, prices_path, args.benchmark)

    if not prices_path.exists():
        raise FileNotFoundError(
            f"Missing {prices_path}. Supply date,ticker,adj_close "
            "or use --download-prices."
        )

    prices = pd.read_csv(prices_path)

    required_prices = {"date", "ticker", "adj_close"}

    if not required_prices.issubset(prices.columns):
        raise ValueError(
            f"{prices_path} must include columns "
            f"{sorted(required_prices)}"
        )

    # Calculate event-window returns.
    returns = build_event_returns(
        scores,
        prices,
        benchmark=args.benchmark,
    )

    returns.to_csv(output_dir / "event_returns.csv", index=False)

    # Generate the standard summary.
    summary = summarize_event_study(returns)
    summary.to_csv(
        output_dir / "event_study_summary.csv",
        index=False,
    )

    # Diagnostic plot: sentiment versus five-session returns.
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for column, label in (
        ("finbert_score", "FinBERT"),
        ("lm_score", "Loughran-McDonald"),
    ):
        if column in returns.columns and "return_5d" in returns.columns:
            subset = returns[[column, "return_5d"]].dropna()

            if not subset.empty:
                ax.scatter(
                    subset[column],
                    subset["return_5d"],
                    alpha=0.75,
                    label=label,
                )
                plotted = True

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Call sentiment score")
    ax.set_ylabel("Forward five-session return")
    ax.set_title("Earnings-call sentiment vs. forward returns")

    if plotted:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No valid observations available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    fig.savefig(
        output_dir / "sentiment_vs_return.png",
        dpi=160,
    )
    plt.close(fig)

    # Generate the run report.
    report = [
        "# Earnings Call Sentiment Analyzer — Run Report",
        "",
        f"- Events processed: **{len(scores)}**",
        f"- FinBERT enabled: **{not args.skip_finbert}**",
        f"- Benchmark: **{args.benchmark or 'not used'}**",
        "",
        "## Event-study summary",
        "",
        (
            summary.to_markdown(index=False)
            if not summary.empty
            else "No summary rows available; check prices and sentiment columns."
        ),
        "",
        "## Interpretation cautions",
        "",
        "- This is an observational event study, not evidence of causality.",
        "- Verify event-date alignment and missing-price exclusions.",
        "- Check overlapping windows, clustering and multiple testing.",
        "- Treat small-sample statistical tests with caution.",
        "- These results are not investment advice.",
    ]

    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
