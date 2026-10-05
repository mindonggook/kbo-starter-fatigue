"""투구 하나 = 행 하나인 데이터셋을 캐시에서 만든다.

타석 단위 성적(피안타율)은 '타자가 익숙해져서'와 '투수가 지쳐서'에 모두 반응해
두 축을 가를 수 없었다(발견 08). 투구 단위에는 그 둘을 다르게 건드리는 지표가 있다.

  볼 비율   — 제구. 타자가 그 투수를 세 번째 본다고 투수가 볼을 더 던지지는 않는다.
  헛스윙률  — 구위. 익숙함도 영향을 주지만 결과 지표보다는 덜하다.

그래서 '볼 비율이 누적 투구수와 함께 오르는가'는 익숙함으로 설명되지 않는,
피로에 가장 가까운 신호다.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from cache_livetext import load_cached
from kbo_livetext import (RE_BATTER, RE_BATTER_SUB, RE_HALF, RE_PITCH,
                          RE_PITCHER_CHANGE, RE_RESULT, _events_in_order,
                          classify_result)
from kbo_gamestate import RE_RUNNER, RUNNER_OUT_TOKENS, RUNNER_SAFE_TOKENS

DATA = Path(__file__).resolve().parent.parent / "data"

COLS = ["game_id", "date", "pitcher", "is_starter", "inning", "half",
        "batting_order", "타순회전", "투수_누적투구수", "타석_투구번호",
        "결과", "주자수", "점수차_투수팀기준"]


def parse_pitches(html: str, game_id: str, away_starter: str, home_starter: str) -> pd.DataFrame:
    events = _events_in_order(html)
    rows: list[dict] = []

    inning = half = None
    current_pitcher = {"초": home_starter, "말": away_starter}
    pitch_total: dict[str, int] = {}
    tto_seen: dict[tuple[str, int], int] = {}
    runs = {"초": 0, "말": 0}
    other = {"초": "말", "말": "초"}

    reached = scored = runner_outs = 0
    slot = batter = None
    pa_pitch_no = 0
    # 타순회전은 타석이 '끝날 때' 확정되지만 투구는 그 전에 던져지므로,
    # 타석이 열릴 때 미리 계산해 그 타석의 모든 투구에 같은 값을 붙인다.
    current_tto = None

    for text in events:
        m = RE_HALF.match(text)
        if m:
            inning, half = int(m.group(1)), m.group(2)
            reached = scored = runner_outs = 0
            slot = batter = None
            pa_pitch_no = 0
            current_tto = None
            continue

        m = RE_PITCHER_CHANGE.match(text)
        if m and half is not None:
            current_pitcher[half] = m.group(2)
            if slot is not None:
                key = (current_pitcher[half], slot)
                current_tto = tto_seen.get(key, 0) + 1
            continue

        m = RE_BATTER_SUB.match(text)
        if m:
            if slot == int(m.group(1)):
                batter = m.group(3)
            continue

        m = RE_BATTER.match(text)
        if m:
            slot, batter = int(m.group(1)), m.group(2)
            pa_pitch_no = 0
            if half is not None:
                key = (current_pitcher[half], slot)
                current_tto = tto_seen.get(key, 0) + 1
            continue

        m = RE_PITCH.match(text)
        if m and half is not None:
            mound = current_pitcher[half]
            pitch_total[mound] = pitch_total.get(mound, 0) + 1
            pa_pitch_no += 1
            rows.append({
                "game_id": game_id, "pitcher": mound,
                "inning": inning, "half": half, "batting_order": slot,
                "타순회전": current_tto, "투수_누적투구수": pitch_total[mound],
                "타석_투구번호": pa_pitch_no, "결과": m.group(2),
                "주자수": reached - scored - runner_outs,
                "점수차_투수팀기준": runs[other[half]] - runs[half],
            })
            continue

        m = RE_RUNNER.match(text)
        if m and half is not None:
            what = m.group(1)
            if "교체" in what:
                continue
            if "홈인" in what:
                runs[half] += 1
                scored += 1
            elif (any(t in what for t in RUNNER_OUT_TOKENS)
                  and not any(t in what for t in RUNNER_SAFE_TOKENS)):
                runner_outs += 1
            continue

        m = RE_RESULT.match(text)
        if m and batter is not None and m.group(1) == batter:
            result_text = m.group(2)
            if half is not None and slot is not None:
                key = (current_pitcher[half], slot)
                tto_seen[key] = tto_seen.get(key, 0) + 1
            _, _, kind = classify_result(result_text)
            if kind in ("안타", "볼넷·사구", "실책 출루", "야수선택", "낫아웃 출루"):
                reached += 1
            if "홈런" in result_text and half is not None:
                runs[half] += 1
                reached += 1
                scored += 1
            slot = batter = None
            pa_pitch_no = 0
            current_tto = None

    return pd.DataFrame(rows)


def build(season: int) -> None:
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    chk_path = DATA / f"kbo_state_check_{season}.parquet"
    ok = None
    if chk_path.exists():
        chk = pd.read_parquet(chk_path)
        ok = set(chk.loc[chk["score_ok"] & chk["starter_ok"], "game_id"])

    starters = box[box["is_starter"]]
    frames = []
    for i, gid in enumerate(sorted(box["game_id"].unique()), start=1):
        if ok is not None and gid not in ok:
            continue
        html = load_cached(gid, season)
        if html is None:
            continue
        g = starters[starters["game_id"] == gid]
        if len(g) < 2:
            continue
        away = g[g["team_side"] == "away"]["선수명"].iloc[0]
        home = g[g["team_side"] == "home"]["선수명"].iloc[0]
        df = parse_pitches(html, gid, away, home)
        if df.empty:
            continue
        df["date"] = box.loc[box["game_id"] == gid, "date"].iloc[0]
        df["is_starter"] = df["pitcher"].isin([away, home])
        frames.append(df.reindex(columns=COLS))
        if i % 200 == 0:
            print(f"  [{season}] {i}경기", file=sys.stderr)

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DATA / f"kbo_pitches_{season}.parquet", index=False)
    print(f"[{season}] 투구 {len(out):,}개 / {out['game_id'].nunique()}경기 "
          f"(선발 {int(out['is_starter'].sum()):,}개)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seasons", nargs="+", type=int)
    a = p.parse_args()
    for s in a.seasons:
        build(s)
