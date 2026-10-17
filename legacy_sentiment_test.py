import pandas as pd
from src.sentiment import tokenize, score_lm, chunks

def test_tokenize():
    assert tokenize("Profit rose; losses fell.") == ["profit", "rose", "losses", "fell"]

def test_lm_score(tmp_path):
    p = tmp_path / "dict.csv"
    pd.DataFrame([{"word":"profit","negative":0,"positive":1},
                  {"word":"loss","negative":1,"positive":0}]).to_csv(p, index=False)
    result = score_lm("profit profit loss", p)
    assert result["lm_positive"] == 2
    assert result["lm_negative"] == 1
    assert abs(result["lm_score"] - 1/3) < 1e-8

def test_chunks():
    assert len(chunks("one two three four", max_words=2)) == 2
