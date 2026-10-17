# Earnings Call Sentiment Analyzer

A reproducible research project that compares **FinBERT contextual sentiment** with a **Loughran–McDonald (LM) dictionary baseline** and tests whether call sentiment is associated with post-earnings stock returns over 1–5 trading days.

> **Research prototype, not investment advice.** Sentiment is not a trading signal by itself. Results can be noisy, sensitive to event timing, and affected by confounders. The pipeline deliberately does not claim alpha unless your own out-of-sample results support it.

## Features

- Scores transcript text in chunks using `ProsusAI/finbert` (positive / negative / neutral probabilities).
- Computes a transparent LM-style dictionary baseline from a lexicon CSV you supply.
- Aggregates sentence/chunk scores to call-level metrics.
- Calculates event-window returns for trading days +1 through +5 from adjusted prices.
- Optionally calculates market-adjusted returns using a benchmark ticker.
- Produces CSV outputs, summary metrics, plots, and a research report.
- Includes a synthetic sample dataset and tests so the pipeline can be smoke-tested without downloading a model or market data.

## Research design

1. **Event definition:** `events.csv` has one row per earnings call and an `event_date`. Set this to the first trading session when the call's information was available. For after-market calls, the first reaction day is normally the next trading session. Record the convention and apply it consistently.
2. **Text:** transcripts are supplied locally; this repository does not scrape paywalled transcripts or redistribute copyrighted calls.
3. **FinBERT:** score transcript chunks using the Hugging Face model. The signed score is `P(positive) - P(negative)`. Neutral probability is retained.
4. **LM baseline:** count positive and negative dictionary matches. Baseline score is `(positive_count - negative_count) / total_tokens`; also retain the positive/negative counts. Use the official LM dictionary obtained under its terms; do not commit the lexicon unless its licence permits redistribution.
5. **Returns:** use adjusted close prices and calculate close-to-close returns from event close to the close 1–5 trading sessions later. For a cleaner event study, define event date as the first tradable reaction date and measure returns from the previous close to +1, or adapt the implementation consistently. The included convention measures from event-date close to future closes; it is a post-event drift window, not the immediate announcement reaction.
6. **Inference:** report sample size, mean, median, hit rate, and a one-sample t-test against zero. Small samples and overlapping windows make p-values fragile. Avoid look-ahead bias; split chronologically for any predictive model and use robust/clustered inference for serious research.

## Quick start

### 1. Create environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Smoke test with bundled synthetic data

```bash
python -m src.pipeline --data-dir data/sample --output-dir outputs/sample --skip-finbert
pytest
```

The sample transcripts and prices are **synthetic examples**, not actual company disclosures or market data. They are only for testing file formats and the pipeline.

### 3. Run with real transcripts

Put one UTF-8 `.txt` file per event in `data/transcripts/`, then create `data/events.csv`:

```csv
ticker,event_date,transcript_file,call_time
AAPL,2025-02-01,aapl_2025q1.txt,after_market
```

`event_date` must match the event-study convention described above. Provide prices in `data/prices.csv` with columns `date,ticker,adj_close`; include the benchmark as a ticker (for example `SPY`) if using market-adjusted returns.

```bash
python -m src.pipeline --data-dir data --output-dir outputs/run
```

FinBERT downloads model files on first run and needs network access. To score only the dictionary baseline, add `--skip-finbert`. To fetch prices automatically, use `--download-prices` (requires yfinance and internet); manually downloaded prices are preferable for reproducibility.

### 4. Supply the LM dictionary

Create `data/lm_dictionary.csv` with at least these columns:

```csv
word,negative,positive
abandon,1,0
accomplish,0,1
```

Use the official Loughran–McDonald dictionary and map its `Negative` and `Positive` columns to the schema above. Keep the downloaded dictionary out of public commits unless the terms explicitly permit redistribution. The code normalizes words to lowercase and matches whole tokens.

### Outputs

- `call_scores.csv`: per-call FinBERT and LM scores.
- `event_returns.csv`: per-event forward 1–5 session returns and optional benchmark-adjusted returns.
- `event_study_summary.csv`: sample sizes, mean/median returns, hit rates, and t-tests by score type and horizon.
- `sentiment_vs_return.png`: diagnostic scatter plot.
- `report.md`: generated run summary and caveats.

## Repository layout

```text
src/
  sentiment.py       FinBERT and LM dictionary scoring
  event_study.py     Forward-return construction and summary statistics
  pipeline.py        CLI orchestration and output generation
data/
  sample/            Synthetic demo inputs
tests/               Unit tests
```

## Methodological caveats

- **Not causal:** association between tone and returns does not prove sentiment caused returns.
- **Event timing matters:** earnings calls can occur before open, during market hours, or after close. A wrong event date can reverse the interpretation.
- **Leakage:** do not include later filings, analyst commentary, or future text in a call transcript. Avoid random train/test splits for time series.
- **Transcript length:** chunking avoids the model's sequence-length limit, but long transcripts may have many repetitive/boilerplate passages. Consider management Q&A separately from prepared remarks.
- **Aggregation:** averaging chunk probabilities weights short and long chunks equally. For a more rigorous variant, weight by token count and report section-level results.
- **LM lexicon:** the LM dictionary is designed for financial text; generic sentiment lexicons can misclassify financial terms.
- **Returns:** the simple t-test assumes independent observations. Event windows may overlap and market events cluster; use HAC or clustered standard errors for publishable work.
- **Survivorship and coverage:** document ticker selection, transcript availability, delisted firms, missing prices, and sample exclusions.
- **Multiple testing:** testing many tickers, horizons, score variants, and strategies inflates false positives. Pre-register a primary hypothesis or adjust for multiple comparisons.
- **Licensing:** review model, transcript, dictionary, and data-provider terms before redistribution.

## Research questions worth testing

1. Does FinBERT score correlate with LM net tone?
2. Is sentiment more informative in Q&A than prepared remarks?
3. Does tone add information after controlling for earnings surprise, market returns, size, and volatility?
4. Is the relationship stable by sector, year, and market regime?
5. Does performance persist after transaction costs and a strict chronological holdout?

## References

- Yang, Y., Uy, M. C. S., & Huang, A. (2020). *FinBERT: A Pretrained Language Model for Financial Communications*. https://arxiv.org/abs/2006.08097
- Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks*. Journal of Finance, 66(1), 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x
- ProsusAI FinBERT: https://github.com/ProsusAI/finBERT
- ML4T word embeddings chapter: https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/16_word_embeddings

## License

Project code is provided under the MIT License. Third-party model weights, dictionaries, transcripts, and market data may have separate terms.
