"""네이버 캐시에서 '타석 단위' 표를 만든다 — 구속과 결과를 같은 행에 묶기 위해.

발견 17에서 구속이 1.6km/h 떨어진다는 것은 확인했지만, 그 저하가 성적 저하의
몇 퍼센트를 설명하는지는 모른다. 두 값이 따로 놀고 있어서다.

네이버 중계는 타석 하나를 textRelay 항목 하나로 묶어준다 — 제목에 타자,
textOptions에 투구 하나하나와 결과, ptsOptions에 그 투구들의 추적값이 들어 있다.
그래서 타석마다 (구속, 결과)를 한 행에 담을 수 있다.

주의: 이닝 안의 항목은 역순(최신이 먼저)이다. 누적 투구수를 세려면 뒤집어야 한다.
"""
import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from kbo_livetext import classify_result

DATA = Path(__file__).resolve().parent.parent / "data"
FT_S_TO_KMH = 0.3048 * 3.6

RE_BATTER = re.compile(r"^(\d+)번타자\s+(\S+)$")
RE_RESULT = re.compile(r"^(\S+)\s*:\s*(.+)$")


def parse_game(raw: dict) -> pd.DataFrame:
    rows = []
    roster = raw["roster"]
    for inn_str in sorted(raw["innings"], key=int):
        # 이닝 안은 역순으로 쌓여 있으므로 뒤집어야 시간순이 된다.
        for item in reversed(raw["innings"][inn_str]):
            pts = item.get("ptsOptions") or []
            if not pts:
                continue

            pcode = None
            texts = []
            state = None
            for t in (item.get("textOptions") or []):
                cs = t.get("currentGameState") or {}
                if pcode is None and cs.get("pitcher"):
                    pcode = str(cs["pitcher"])
                # 타석이 열릴 때의 상태 — 주자·점수·아웃이 모두 여기 들어 있다.
                if state is None and cs:
                    state = cs
                if t.get("text"):
                    texts.append(t["text"])
            state = state or {}

            def _i(key):
                try:
                    return int(state.get(key) or 0)
                except (TypeError, ValueError):
                    return 0

            # base1/2/3에는 '그 루에 있는 주자의 타순 번호'가 들어간다(0이면 빈 루).
            # 그러므로 더하는 게 아니라 0이 아닌 루의 개수를 세야 한다.
            runners = sum(1 for b in ("base1", "base2", "base3") if _i(b) > 0)

            slot = batter = None
            m = RE_BATTER.match(item.get("title") or "")
            if m:
                slot, batter = int(m.group(1)), m.group(2)

            result_text = None
            for tx in texts:
                m2 = RE_RESULT.match(tx)
                if m2 and batter and m2.group(1) == batter:
                    result_text = m2.group(2)
                    break

            speeds = []
            rel_x, rel_z = [], []
            for q in pts:
                try:
                    speeds.append(math.sqrt(q["vx0"] ** 2 + q["vy0"] ** 2 + q["vz0"] ** 2) * FT_S_TO_KMH)
                    rel_x.append(q.get("x0"))
                    rel_z.append(q.get("z0"))
                except (KeyError, TypeError):
                    continue
            if not speeds:
                continue

            is_ab = is_hit = None
            kind = None
            if result_text:
                is_ab, is_hit, kind = classify_result(result_text)

            rows.append({
                "game_id": raw["game_id"], "season": raw["season"],
                "inning": int(inn_str), "pitcher_code": pcode,
                "pitcher": roster.get(pcode), "batting_order": slot, "batter": batter,
                "투구수": len(speeds),
                "최고구속": max(speeds), "평균구속": float(np.mean(speeds)),
                "release_x": float(np.mean([v for v in rel_x if v is not None])) if any(v is not None for v in rel_x) else np.nan,
                "release_z": float(np.mean([v for v in rel_z if v is not None])) if any(v is not None for v in rel_z) else np.nan,
                "result": result_text, "종류": kind,
                "타수": is_ab, "안타": is_hit,
                "주자수": runners, "아웃카운트": _i("out"),
                "home_score": _i("homeScore"), "away_score": _i("awayScore"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["타석번호"] = df.groupby("pitcher_code").cumcount() + 1
    df["투수_누적투구수"] = df.groupby("pitcher_code")["투구수"].cumsum()
    # 그 타석 '이전'까지의 누적 — 결과가 나오기 전 상태가 설명변수여야 한다.
    df["이전_누적투구수"] = df["투수_누적투구수"] - df["투구수"]
    # 타순 회전: 같은 투수가 같은 타순을 몇 번째 상대하는가
    df["타순회전"] = df.groupby(["pitcher_code", "batting_order"]).cumcount() + 1
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
        out.to_parquet(DATA / f"naver_pa_{season}.parquet", index=False)
        ok = out["result"].notna().mean() * 100
        print(f"[{season}] 타석 {len(out):,}건 / {out['game_id'].nunique()}경기 "
              f"· 결과 파싱 {ok:.1f}% · 최고구속 중앙 {out['최고구속'].median():.1f}km/h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seasons", nargs="+", type=int)
    a = p.parse_args()
    build(a.seasons)
