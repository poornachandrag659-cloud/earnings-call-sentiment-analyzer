"""Sentiment scoring helpers: FinBERT and Loughran–McDonald dictionary baseline."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
import pandas as pd

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

def score_lm(text: str, dictionary_path: str | Path) -> dict:
    """Score text against a user-supplied CSV with word,negative,positive columns."""
    path = Path(dictionary_path)
    if not path.exists():
        return {"lm_score": float("nan"), "lm_positive": 0, "lm_negative": 0,
                "lm_token_count": len(tokenize(text)), "lm_coverage": 0.0}
    lex = pd.read_csv(path)
    required = {"word", "negative", "positive"}
    if not required.issubset(lex.columns):
        raise ValueError(f"Dictionary must contain columns: {sorted(required)}")
    lex["word"] = lex["word"].astype(str).str.lower().str.strip()
    negative = set(lex.loc[pd.to_numeric(lex["negative"], errors="coerce").fillna(0).astype(int) > 0, "word"])
    positive = set(lex.loc[pd.to_numeric(lex["positive"], errors="coerce").fillna(0).astype(int) > 0, "word"])
    tokens = tokenize(text)
    pos = sum(t in positive for t in tokens)
    neg = sum(t in negative for t in tokens)
    denom = len(tokens)
    return {"lm_score": (pos - neg) / denom if denom else 0.0,
            "lm_positive": pos, "lm_negative": neg, "lm_token_count": denom,
            "lm_coverage": (pos + neg) / denom if denom else 0.0}

def chunks(text: str, max_words: int = 180) -> list[str]:
    words = (text or "").split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [""]

def score_finbert(text: str, model_name: str = "ProsusAI/finbert", max_words: int = 180,
                  batch_size: int = 8) -> dict:
    """Score transcript chunks. Requires transformers/torch and model download access."""
    from transformers import pipeline
    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name,
                          top_k=None, truncation=True)
    pieces = chunks(text, max_words=max_words)
    predictions = []
    for start in range(0, len(pieces), batch_size):
        predictions.extend(classifier(pieces[start:start + batch_size]))
    # transformers pipeline with top_k=None returns list-of-list label scores.
    rows = []
    for pred in predictions:
        mapping = {str(x["label"]).lower(): float(x["score"]) for x in pred}
        rows.append({"positive": mapping.get("positive", 0.0),
                     "negative": mapping.get("negative", 0.0),
                     "neutral": mapping.get("neutral", 0.0)})
    df = pd.DataFrame(rows)
    avg = df.mean(numeric_only=True)
    return {"finbert_positive": float(avg["positive"]),
            "finbert_negative": float(avg["negative"]),
            "finbert_neutral": float(avg["neutral"]),
            "finbert_score": float(avg["positive"] - avg["negative"]),
            "finbert_chunks": len(rows)}
