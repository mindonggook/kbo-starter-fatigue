"""문자중계 원문을 gzip으로 캐시해둔다.

점수·주자 상태 추적은 파싱 규칙을 여러 번 고쳐가며 검증해야 하는데, 그때마다
720경기를 다시 받을 수는 없다. 한 번 받아 캐시해두면 재파싱이 몇 초로 끝난다.
"""
import argparse
import gzip
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from kbo_livetext import fetch_game_text

DATA = Path(__file__).resolve().parent.parent / "data"


def cache_season(season: int, sleep: float = 0.35) -> None:
    cache_dir = DATA / f"livetext_cache_{season}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    game_ids = sorted(box["game_id"].unique())

    session = requests.Session()
    t0 = time.time()
    fetched = skipped = failed = 0
    for i, gid in enumerate(game_ids, start=1):
        path = cache_dir / f"{gid}.html.gz"
        if path.exists():
            skipped += 1
            continue
        try:
            html = fetch_game_text(gid, season, session)
        except Exception as e:
            print(f"  [경고] {gid} 실패: {e}", file=sys.stderr)
            failed += 1
            time.sleep(sleep)
            continue
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(html)
        fetched += 1
        if i % 100 == 0:
            print(f"  [{season}] {i}/{len(game_ids)}경기, {time.time()-t0:.0f}초", file=sys.stderr)
        time.sleep(sleep)

    size = sum(p.stat().st_size for p in cache_dir.glob("*.html.gz")) / 1024 / 1024
    print(f"[{season}] 새로 받음 {fetched} · 이미 있음 {skipped} · 실패 {failed} "
          f"-> {cache_dir.name} ({size:.0f} MB)")


def load_cached(game_id: str, season: int) -> str | None:
    path = DATA / f"livetext_cache_{season}" / f"{game_id}.html.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    a = p.parse_args()
    cache_season(a.season)
