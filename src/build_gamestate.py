"""캐시된 문자중계에서 '타석마다의 경기 상태'가 붙은 데이터셋을 만든다.

검증 결과는 경기별로 같이 저장한다. 점수 추적은 720경기 전부 공식 최종 점수와
일치하지만 아웃·주자 추적은 드문 주루 상황에서 어긋나므로, 분석에서 변수별로
믿을 수 있는 경기만 골라 쓸 수 있어야 한다.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from cache_livetext import load_cached
from kbo_gamestate import PA_COLS, parse_game_with_state, validate

DATA = Path(__file__).resolve().parent.parent / "data"


def build(season: int) -> None:
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    sched = pd.read_parquet(DATA / f"kbo_schedule_{season}.parquet").set_index("game_id")
    starters = box[box["is_starter"]]

    frames, checks = [], []
    for i, gid in enumerate(sorted(box["game_id"].unique()), start=1):
        html = load_cached(gid, season)
        if html is None:
            continue
        g = starters[starters["game_id"] == gid]
        if len(g) < 2 or gid not in sched.index:
            continue

        away = g[g["team_side"] == "away"]["선수명"].iloc[0]
        home = g[g["team_side"] == "home"]["선수명"].iloc[0]
        df = parse_game_with_state(html, gid, away, home)
        if df.empty:
            continue

        s = sched.loc[gid]
        v = validate(df, int(s["away_score"]), int(s["home_score"]))

        # 박스스코어 투구수와도 맞는지(선발 기준) 함께 본다.
        totals = df.attrs.get("pitch_total", {})
        ref = g.set_index("선수명")["투구수"]
        v["starter_ok"] = all(totals.get(n) == int(c) for n, c in ref.items())
        v["game_id"] = gid

        df["date"] = s.name[:8]
        df["is_starter"] = df["pitcher"].isin([away, home])
        frames.append(df.reindex(columns=PA_COLS + ["date"]))
        checks.append(v)

        if i % 200 == 0:
            print(f"  [{season}] {i}경기 처리", file=sys.stderr)

    pa = pd.concat(frames, ignore_index=True)
    chk = pd.DataFrame(checks)
    pa.to_parquet(DATA / f"kbo_pa_state_{season}.parquet", index=False)
    chk.to_parquet(DATA / f"kbo_state_check_{season}.parquet", index=False)

    print(f"[{season}] 타석 {len(pa):,}건 / {chk['game_id'].nunique()}경기")
    print(f"  점수 일치 {chk['score_ok'].sum()} · 아웃 정합 {chk['outs_ok'].sum()} "
          f"· 주자 정합 {chk['runner_ok'].sum()} · 선발 투구수 일치 {chk['starter_ok'].sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    a = p.parse_args()
    build(a.season)
