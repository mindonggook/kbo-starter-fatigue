"""투구 하나에 추적값과 결과를 함께 붙인다.

발견 13에 설명되지 않은 모순이 있다 — 투구수가 쌓이면 볼 비율이 오히려 줄었는데
(-0.90%p), 발견 17에서는 릴리스 포인트가 두 배로 흔들린다고 했다.
폼이 무너지는데 제구가 좋아진다는 것은 앞뒤가 안 맞는다.

둘을 직접 이으려면 같은 투구에 '릴리스 좌표'와 '볼/스트라이크'가 함께 있어야 한다.
네이버 중계에서 ptsOptions의 ballcount와 textOptions의 'N구 …' 번호가 같은 순번이므로
그 번호로 맞붙이면 된다.
"""
import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
FT_S_TO_KMH = 0.3048 * 3.6

RE_PITCH_TEXT = re.compile(r"^(\d+)구\s+(\S+)")


def parse_game(raw: dict) -> pd.DataFrame:
    rows = []
    roster = raw["roster"]
    for inn_str in sorted(raw["innings"], key=int):
        # 이닝 안은 역순이므로 뒤집어야 시간순이 된다.
        for item in reversed(raw["innings"][inn_str]):
            pts = item.get("ptsOptions") or []
            if not pts:
                continue

            pcode = None
            outcomes = {}
            for t in (item.get("textOptions") or []):
                cs = t.get("currentGameState") or {}
                if pcode is None and cs.get("pitcher"):
                    pcode = str(cs["pitcher"])
                m = RE_PITCH_TEXT.match((t.get("text") or "").strip())
                if m:
                    outcomes[int(m.group(1))] = m.group(2)

            for q in pts:
                try:
                    speed = math.sqrt(q["vx0"] ** 2 + q["vy0"] ** 2 + q["vz0"] ** 2) * FT_S_TO_KMH
                except (KeyError, TypeError):
                    continue
                no = q.get("ballcount")
                rows.append({
                    "game_id": raw["game_id"], "season": raw["season"],
                    "inning": int(inn_str), "pitcher_code": pcode,
                    "pitcher": roster.get(pcode),
                    "투구번호": no, "결과": outcomes.get(no),
                    "구속": speed,
                    "release_x": q.get("x0"), "release_z": q.get("z0"),
                    "plate_x": q.get("crossPlateX"), "plate_z": q.get("crossPlateY"),
                    "sz_top": q.get("topSz"), "sz_bot": q.get("bottomSz"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["투수_누적투구수"] = df.groupby("pitcher_code").cumcount() + 1
    return df


def build(seasons: list[int]) -> None:
    for season in seasons:
        cache = DATA / f"naver_pts_cache_{season}"
        frames = []
        for path in sorted(cache.glob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                df = parse_game(json.load(f))
            if not df.empty:
                frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(DATA / f"naver_pitch_{season}.parquet", index=False)
        matched = out["결과"].notna().mean() * 100
        print(f"[{season}] 투구 {len(out):,}개 / {out['game_id'].nunique()}경기 "
              f"· 결과 매칭 {matched:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seasons", nargs="+", type=int)
    a = p.parse_args()
    build(a.seasons)
