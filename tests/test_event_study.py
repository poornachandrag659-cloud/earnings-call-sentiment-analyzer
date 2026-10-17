import pandas as pd
from src.event_study import build_event_returns, summarize_event_study

def test_returns_and_summary():
    events = pd.DataFrame([{"ticker":"ABC","event_date":"2024-01-02","finbert_score":0.2,"lm_score":0.1}])
    prices = pd.DataFrame([
        {"date":f"2024-01-0{i}","ticker":"ABC","adj_close":100+i*10} for i in range(2,9)
    ])
    out = build_event_returns(events, prices)
    assert abs(out.loc[0, "return_1d"] - (130/120 - 1)) < 1e-8
    summary = summarize_event_study(out)
    assert not summary.empty
