"""네이버 스포츠 중계에서 투구 추적 데이터(PTS)를 수집한다.

KBO 공식 기록실에는 구속이 없다. 그래서 지금까지 '구위 저하'를 헛스윙률이라는
대리지표로만 볼 수 있었다(발견 13). 네이버 중계 API의 ptsOptions에는
PITCHf/x 형식의 원시 추적값이 들어 있어 구속을 직접 계산할 수 있다.

    구속(ft/s) = sqrt(vx0^2 + vy0^2 + vz0^2)   (y0=50ft 기준)
    릴리스 포인트 = (x0, z0)

호출은 이닝 단위로만 되고(경기당 9~12회), 상업 포털이므로 요청 간격을 넉넉히 둔다.
개인 분석용이며 원자료를 재배포하지 않는 것을 전제로 한다.
"""
import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
BASE = "https://api-gw.sports.naver.com/schedule/games"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://m.sports.naver.com/",
}
FT_S_TO_KMH = 0.3048 * 3.6


def naver_game_id(kbo_game_id: str, season: int) -> str:
    return f"{kbo_game_id}{season}"


def fetch_game(kbo_game_id: str, season: int, session: requests.Session,
               sleep: float = 0.8, max_inning: int = 13) -> dict | None:
    """이닝별로 받아 하나의 dict로 합친다. 투구가 없는 이닝이 연속되면 멈춘다."""
    gid = naver_game_id(kbo_game_id, season)
    innings, roster = {}, {}
    empty_streak = 0

    for inn in range(1, max_inning + 1):
        try:
            r = session.get(f"{BASE}/{gid}/relay", params={"inning": inn},
                            headers=HEADERS, timeout=20)
            r.raise_for_status()
            d = r.json()["result"]["textRelayData"]
        except Exception:
            time.sleep(sleep)
            empty_streak += 1
            if empty_streak >= 2 and inn > 9:
                break
            continue

        # 엔트리(homeEntry/awayEntry)는 실제 등판 투수와 pcode 체계가 어긋난다.
        # 실제로 던진 투수는 라인업 쪽에 있다.
        for side in ("homeLineup", "awayLineup", "homeEntry", "awayEntry"):
            for p in (d.get(side) or {}).get("pitcher", []) or []:
                code = str(p.get("pcode"))
                if p.get("name"):
                    roster.setdefault(code, p.get("name"))
                else:
                    roster.setdefault(code, None)

        relays = d.get("textRelays") or []
        has_pts = any(r_.get("ptsOptions") for r_ in relays)
        if has_pts:
            innings[inn] = relays
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 2 and inn >= 9:
                break
        time.sleep(sleep)

    if not innings:
        return None
    return {"game_id": kbo_game_id, "season": season, "roster": roster,
            "innings": {str(k): v for k, v in innings.items()}}


def parse_game(raw: dict) -> pd.DataFrame:
    """타석마다 붙은 투수 pcode를 투구에 전가해 투구 단위 표로 편다."""
    rows = []
    roster = raw["roster"]
    for inn_str, relays in sorted(raw["innings"].items(), key=lambda kv: int(kv[0])):
        for item in relays:
            pts = item.get("ptsOptions") or []
            if not pts:
                continue
            pcode = None
            for t in (item.get("textOptions") or []):
                cs = t.get("currentGameState") or {}
                if cs.get("pitcher"):
                    pcode = str(cs["pitcher"])
                    break
            for q in pts:
                try:
                    speed = math.sqrt(q["vx0"] ** 2 + q["vy0"] ** 2 + q["vz0"] ** 2)
                except (KeyError, TypeError):
                    continue
                rows.append({
                    "game_id": raw["game_id"], "season": raw["season"],
                    "inning": int(inn_str), "pitch_id": q.get("pitchId"),
                    "pitcher_code": pcode, "pitcher": roster.get(pcode),
                    "구속": speed * FT_S_TO_KMH,
                    "release_x": q.get("x0"), "release_z": q.get("z0"),
                    "plate_x": q.get("crossPlateX"), "plate_z": q.get("crossPlateY"),
                    "ax": q.get("ax"), "az": q.get("az"),
                    "sz_top": q.get("topSz"), "sz_bot": q.get("bottomSz"),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 같은 투수가 그 경기에서 몇 번째로 던진 공인지 — 피로 축이 된다.
    df["투수_누적투구수"] = df.groupby("pitcher_code").cumcount() + 1
    return df


def collect(season: int, limit: int | None, sleep: float, out_name: str,
            fetch_only: bool = False, parse_only: bool = False) -> None:
    cache_dir = DATA / f"naver_pts_cache_{season}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    game_ids = sorted(box["game_id"].unique())
    if limit:
        # 시즌 전체에 고르게 퍼지도록 일정 간격으로 고른다(앞부분만 쏠리지 않게).
        step = max(1, len(game_ids) // limit)
        game_ids = game_ids[::step][:limit]

    session = requests.Session()
    t0 = time.time()
    done = fetched = 0
    for i, gid in enumerate([] if parse_only else game_ids, start=1):
        path = cache_dir / f"{gid}.json.gz"
        if path.exists():
            done += 1
            continue
        raw = fetch_game(gid, season, session, sleep=sleep)
        if raw is None:
            continue
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        fetched += 1
        if i % 20 == 0:
            print(f"  [{season}] {i}/{len(game_ids)}경기, {time.time()-t0:.0f}초", file=sys.stderr)

    if fetch_only:
        print(f"[{season}] 새로 받음 {fetched} · 캐시 재사용 {done} (파싱 생략)")
        return

    frames = []
    for path in sorted(cache_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            df = parse_game(json.load(f))
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DATA / out_name, index=False)
    print(f"[{season}] 새로 받음 {fetched} · 캐시 재사용 {done} "
          f"-> 투구 {len(out):,}개 / {out['game_id'].nunique()}경기")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.8)
    p.add_argument("--out", default=None)
    # 수집과 파싱을 나눠야 한 번의 실행이 시간 제한 안에 끝난다.
    p.add_argument("--fetch-only", action="store_true")
    p.add_argument("--parse-only", action="store_true")
    a = p.parse_args()
    collect(a.season, a.limit, a.sleep, a.out or f"naver_pts_{a.season}.parquet",
            fetch_only=a.fetch_only, parse_only=a.parse_only)
