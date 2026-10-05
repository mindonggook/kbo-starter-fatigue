"""시즌 전체 선발투수 등판 데이터셋 구축.

일정을 먼저 받고, 경기마다 박스스코어를 순회 호출해서 투수 등판 기록을
쌓는다. 경기당 요청 1번(약 400ms 간격)이라 정규시즌(720경기 안팎) 기준
5~10분 걸린다. 중간에 끊겨도 이어서 할 수 있게 진행 상황을 주기적으로
체크포인트 parquet에 저장한다.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from kbo_boxscore import fetch_pitcher_appearances
from kbo_client import polite_sleep
from kbo_schedule import fetch_season_schedule

DATA = Path(__file__).resolve().parent.parent / "data"


def build_season_dataset(season: int, checkpoint_every: int = 50) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    schedule_path = DATA / f"kbo_schedule_{season}.parquet"
    if schedule_path.exists():
        schedule = pd.read_parquet(schedule_path)
    else:
        schedule = fetch_season_schedule(season)
        schedule.to_parquet(schedule_path, index=False)
    print(f"[{season}] 일정 {len(schedule)}경기", file=sys.stderr)

    out_path = DATA / f"kbo_pitcher_appearances_{season}.parquet"
    done_ids: set[str] = set()
    frames: list[pd.DataFrame] = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        frames.append(prev)
        done_ids = set(prev["game_id"].unique())
        print(f"[{season}] 기존 체크포인트에서 {len(done_ids)}경기 재사용", file=sys.stderr)

    remaining = schedule[~schedule["game_id"].isin(done_ids)]
    t0 = time.time()
    for i, game in enumerate(remaining.itertuples(), start=1):
        try:
            box = fetch_pitcher_appearances(game.game_id, season)
        except Exception as e:
            print(f"  [경고] {game.game_id} 실패: {e}", file=sys.stderr)
            continue

        if not box.empty:
            box["date"] = game.date
            box["away_team"] = game.away_team
            box["home_team"] = game.home_team
            box["team"] = box["team_side"].map({"away": game.away_team, "home": game.home_team})
            box["opponent"] = box["team_side"].map({"away": game.home_team, "home": game.away_team})
            frames.append(box)

        if i % checkpoint_every == 0 or i == len(remaining):
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            elapsed = time.time() - t0
            print(f"  [{season}] {i}/{len(remaining)}경기 처리, {elapsed:.0f}초 경과 -> 체크포인트 저장", file=sys.stderr)

        polite_sleep(0.4)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    a = p.parse_args()

    df = build_season_dataset(a.season)
    print(f"[{a.season}] 최종 {len(df)}행 (선발 {(df['is_starter']).sum() if not df.empty else 0}건)")
