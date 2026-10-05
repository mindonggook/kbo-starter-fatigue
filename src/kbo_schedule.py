"""KBO 시즌 일정(경기 목록) 수집.

/ws/Schedule.asmx/GetScheduleList 를 월 단위로 호출해 그 달의 모든 경기(day별
최대 5경기, RowSpan으로 날짜 셀 병합)를 가져온다. gameId는 링크(리뷰 버튼)
href에서 뽑는데, 우천취소 등으로 경기가 안 열리면 그 셀이 비어 있어 자연히
걸러진다.
"""
import re
from pathlib import Path

import pandas as pd

from kbo_client import BASE, TEAM_CODE, get_session, strip_tags

DATA = Path(__file__).resolve().parent.parent / "data"

SCHEDULE_URL = f"{BASE}/ws/Schedule.asmx/GetScheduleList"
REGULAR_SEASON_SR_IDS = "0,9,6"  # 정규시즌(더블헤더 포함)

GAMEID_RE = re.compile(r"gameId=(\w+)")
SCORE_RE = re.compile(
    r"<span>([^<]+)</span><em><span[^>]*>(\d+)</span><span>vs</span>"
    r"<span[^>]*>(\d+)</span></em><span>([^<]+)</span>"
)


def fetch_month_schedule(season: int, month: int) -> list[dict]:
    session = get_session()
    payload = {
        "leId": 1,
        "srIdList": REGULAR_SEASON_SR_IDS,
        "seasonId": str(season),
        "gameMonth": f"{month:02d}",
        "teamId": "",
    }
    r = session.post(SCHEDULE_URL, data=payload, timeout=20)
    r.raise_for_status()
    data = r.json()

    games = []
    current_day = None
    for row in data.get("rows", []):
        cells = [c["Text"] for c in row["row"]]
        if re.match(r"^\d{2}\.\d{2}", cells[0]):
            current_day = cells[0]
            cells = cells[1:]

        play_html = cells[1]
        relay_html = cells[2]
        m_id = GAMEID_RE.search(relay_html)
        if not m_id:
            continue  # 취소/경기전 등 리뷰 링크가 없는 셀
        game_id = m_id.group(1)
        game_date = game_id[:8]

        m_score = SCORE_RE.search(play_html)
        if m_score:
            t1, s1, s2, t2 = m_score.groups()
            score1, score2 = int(s1), int(s2)
        else:
            score1 = score2 = None

        away_code, home_code = game_id[8:10], game_id[10:12]
        games.append({
            "game_id": game_id,
            "date": game_date,
            "away_team": TEAM_CODE.get(away_code, away_code),
            "home_team": TEAM_CODE.get(home_code, home_code),
            "away_score": score1,
            "home_score": score2,
            "stadium": strip_tags(cells[6]) if len(cells) > 6 else None,
        })
    return games


def fetch_season_schedule(season: int, months: range = range(3, 12)) -> pd.DataFrame:
    all_games = []
    for month in months:
        all_games.extend(fetch_month_schedule(season, month))
    df = pd.DataFrame(all_games).drop_duplicates(subset="game_id").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    a = p.parse_args()

    df = fetch_season_schedule(a.season)
    DATA.mkdir(exist_ok=True)
    out = DATA / f"kbo_schedule_{a.season}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)}경기 -> {out}")
