"""시즌 전체 문자중계(타석 단위) 데이터셋 구축.

경기마다 문자중계를 받아 타석 하나 = 행 하나로 펼치고, 그 결과를 박스스코어의
투수별 투구수와 대조해 경기별 검증 플래그를 같이 남긴다. KBO 중계 텍스트는
간혹 투수 교체를 통째로 빠뜨리기 때문에(확인된 사례 있음), 분석 때 어긋난 경기를
걸러낼 수 있어야 한다.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from kbo_livetext import fetch_game_text, parse_game

DATA = Path(__file__).resolve().parent.parent / "data"

PA_COLS = [
    "game_id", "date", "inning", "half", "batting_team", "pitcher", "is_starter",
    "batting_order", "batter", "result", "종류", "타수", "안타",
    "타석_투구수", "투수_누적투구수", "타순회전",
]


def build(season: int, checkpoint_every: int = 40, sleep: float = 0.35) -> None:
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    starters = box[box["is_starter"]]

    pa_path = DATA / f"kbo_plate_appearances_{season}.parquet"
    check_path = DATA / f"kbo_livetext_check_{season}.parquet"

    frames: list[pd.DataFrame] = []
    checks: list[dict] = []
    done: set[str] = set()
    if pa_path.exists() and check_path.exists():
        frames.append(pd.read_parquet(pa_path))
        prev_check = pd.read_parquet(check_path)
        checks.extend(prev_check.to_dict("records"))
        done = set(prev_check["game_id"])
        print(f"[{season}] 체크포인트에서 {len(done)}경기 재사용", file=sys.stderr)

    game_ids = [g for g in sorted(box["game_id"].unique()) if g not in done]
    print(f"[{season}] 남은 {len(game_ids)}경기", file=sys.stderr)

    session = requests.Session()
    t0 = time.time()
    for i, gid in enumerate(game_ids, start=1):
        g = starters[starters["game_id"] == gid]
        if len(g) < 2:
            checks.append({"game_id": gid, "n_pa": 0, "starter_ok": False, "note": "선발 정보 없음"})
            continue

        away = g[g["team_side"] == "away"]["선수명"].iloc[0]
        home = g[g["team_side"] == "home"]["선수명"].iloc[0]
        meta = box[box["game_id"] == gid].iloc[0]

        try:
            html = fetch_game_text(gid, season, session)
            pa = parse_game(html, gid, away, home)
        except Exception as e:
            checks.append({"game_id": gid, "n_pa": 0, "starter_ok": False, "note": f"실패: {e}"})
            time.sleep(sleep)
            continue

        totals = pa.attrs.get("pitch_total", {})
        ref = box[(box["game_id"] == gid) & box["is_starter"]].set_index("선수명")["투구수"]
        starter_ok = all(totals.get(name) == int(cnt) for name, cnt in ref.items())

        if not pa.empty:
            pa["date"] = meta["date"]
            pa["is_starter"] = pa["pitcher"].isin([away, home])
            frames.append(pa.reindex(columns=PA_COLS))

        checks.append({"game_id": gid, "n_pa": len(pa), "starter_ok": bool(starter_ok), "note": ""})

        if i % checkpoint_every == 0 or i == len(game_ids):
            pd.concat(frames, ignore_index=True).to_parquet(pa_path, index=False)
            pd.DataFrame(checks).to_parquet(check_path, index=False)
            print(f"  [{season}] {i}/{len(game_ids)}경기, {time.time()-t0:.0f}초 -> 저장", file=sys.stderr)

        time.sleep(sleep)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(pa_path, index=False)
        pd.DataFrame(checks).to_parquet(check_path, index=False)
        chk = pd.DataFrame(checks)
        print(f"[{season}] 타석 {len(combined)}건 / {chk['game_id'].nunique()}경기 "
              f"(선발 투구수 검증 통과 {int(chk['starter_ok'].sum())}경기)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    a = p.parse_args()
    build(a.season)
