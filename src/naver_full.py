"""투구 하나에 구종·실측 구속·결과·추적값을 모두 붙인다.

앞서 ptsOptions(추적값)와 textOptions의 '1구 스트라이크' 텍스트만 썼는데,
textOptions의 투구 항목에는 더 많은 것이 들어 있었다.

    stuff        구종 (직구 / 슬라이더 / 커브 / 체인지업 / 포크 / 커터 …)
    speed        실측 구속 (계산값이 아니다)
    pitchResult  투구 결과 코드
    ptsPitchId   추적값(ptsOptions.pitchId)과 이어주는 열쇠

구종이 생기면 앞선 두 발견을 다시 봐야 한다.
  발견 17 — '최고 구속 하락'이 직구가 느려진 것인지, 변화구를 더 던진 것인지
  발견 22 — '릴리스 흔들림'이 폼이 무너진 것인지, 구종마다 릴리스가 달라서인지
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

# PITCHf/x는 (x0, y0, z0)를 홈플레이트에서 50피트 지점으로 준다 — 실제 릴리스가 아니다.
# 거기서의 좌표 차이에는 이미 날아온 궤적이 섞이므로, 투구 폼을 재려면 되돌려야 한다.
RELEASE_Y = 55.0      # 실제 릴리스에 가까운 평면(익스텐션 ~5.5피트)
PLATE_Y = 17 / 24     # crossPlateY가 주는 플레이트 앞면 거리(0.7083피트)


def _project(q, y_target):
    """등가속도 궤적을 따라 y = y_target 평면에서의 (x, z)를 구한다.

    y(t) = y0 + vy0·t + ½·ay·t²  를 t에 대해 풀고 그 t를 x, z에 넣는다.
    공은 y가 줄어드는 방향으로 날아가므로, 50피트보다 뒤(55피트)는 t < 0이다.
    """
    try:
        y0, vy0, ay = float(q["y0"]), float(q["vy0"]), float(q["ay"])
        a = 0.5 * ay
        b = vy0
        c = y0 - y_target
        if abs(a) < 1e-9:
            t = -c / b
        else:
            disc = b * b - 4 * a * c
            if disc < 0:
                return np.nan, np.nan
            root = math.sqrt(disc)
            t1, t2 = (-b + root) / (2 * a), (-b - root) / (2 * a)
            # y0에서 가장 가까운 시점을 고른다(물리적으로 맞는 쪽).
            t = t1 if abs(t1) < abs(t2) else t2
        x = q["x0"] + q["vx0"] * t + 0.5 * q["ax"] * t * t
        z = q["z0"] + q["vz0"] * t + 0.5 * q["az"] * t * t
        return x, z
    except (KeyError, TypeError, ValueError):
        return np.nan, np.nan


def parse_game(raw: dict) -> pd.DataFrame:
    rows = []
    roster = raw["roster"]
    for inn_str in sorted(raw["innings"], key=int):
        # 이닝 안은 역순으로 쌓여 있다.
        for item in reversed(raw["innings"][inn_str]):
            pts = {q.get("pitchId"): q for q in (item.get("ptsOptions") or [])}
            if not pts:
                continue

            pcode = state = None
            texts, pitches = [], []
            for t in (item.get("textOptions") or []):
                cs = t.get("currentGameState") or {}
                if pcode is None and cs.get("pitcher"):
                    pcode = str(cs["pitcher"])
                if state is None and cs:
                    state = cs
                if t.get("text"):
                    texts.append(t["text"])
                if t.get("ptsPitchId") or t.get("stuff"):
                    pitches.append(t)
            state = state or {}

            def _i(key):
                try:
                    return int(state.get(key) or 0)
                except (TypeError, ValueError):
                    return 0

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
            kind = classify_result(result_text)[2] if result_text else None

            runners = sum(1 for b in ("base1", "base2", "base3") if _i(b) > 0)

            # 투구는 textOptions에 역순으로 담겨 있으므로 번호로 정렬한다.
            for t in sorted(pitches, key=lambda x: x.get("pitchNum") or 0):
                q = pts.get(t.get("ptsPitchId")) or {}
                try:
                    derived = math.sqrt(q["vx0"] ** 2 + q["vy0"] ** 2 + q["vz0"] ** 2) * FT_S_TO_KMH
                except (KeyError, TypeError):
                    derived = np.nan
                try:
                    speed = float(t.get("speed"))
                except (TypeError, ValueError):
                    speed = np.nan

                rx, rz = _project(q, RELEASE_Y)
                _, pz = _project(q, PLATE_Y)
                rows.append({
                    "game_id": raw["game_id"], "season": raw["season"],
                    "inning": int(inn_str), "pitcher_code": pcode,
                    "pitcher": roster.get(pcode),
                    "batting_order": slot, "batter": batter,
                    "투구번호": t.get("pitchNum"),
                    "구종": t.get("stuff") or None,
                    "구속": speed, "구속_계산": derived,
                    "투구결과": (t.get("text") or "").split(maxsplit=1)[-1] or None,
                    # 50피트 평면의 원값(과거 분석이 쓴 것)과 55피트로 되돌린 값을 함께 남긴다.
                    "p50_x": q.get("x0"), "p50_z": q.get("z0"),
                    "release_x": rx, "release_z": rz,
                    "plate_x": q.get("crossPlateX"), "plate_z": pz,
                    "sz_top": q.get("topSz"), "sz_bot": q.get("bottomSz"),
                    "ax": q.get("ax"), "az": q.get("az"),
                    "타석결과": result_text, "타석종류": kind,
                    "주자수": runners, "아웃카운트": _i("out"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["투수_누적투구수"] = df.groupby("pitcher_code").cumcount() + 1
    df["타순회전"] = (df.drop_duplicates(["pitcher_code", "batting_order", "batter", "inning"])
                      .groupby(["pitcher_code", "batting_order"]).cumcount() + 1
                      ).reindex(df.index).ffill()
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
        out.to_parquet(DATA / f"naver_full_{season}.parquet", index=False)

        have_stuff = out["구종"].notna().mean() * 100
        have_speed = out["구속"].notna().mean() * 100
        both = out.dropna(subset=["구속", "구속_계산"])
        corr = both["구속"].corr(both["구속_계산"]) if len(both) else float("nan")
        gap = (both["구속"] - both["구속_계산"]).abs().median() if len(both) else float("nan")
        print(f"[{season}] 투구 {len(out):,}개 · 구종 {have_stuff:.1f}% · 실측구속 {have_speed:.1f}% "
              f"· 계산구속과 상관 {corr:.4f} (중앙 오차 {gap:.2f}km/h)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seasons", nargs="+", type=int)
    a = p.parse_args()
    build(a.seasons)
