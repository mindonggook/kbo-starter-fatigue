"""경기 하나의 투수 등판 기록(박스스코어) 수집.

/ws/Schedule.asmx/GetBoxScoreScroll 응답의 arrPitcher[0]/[1]이 각각
원정/홈 팀의 투수 기록표(JSON 문자열로 이중 인코딩됨)다. 행 하나 = 그 경기에서
투수 한 명의 등판(스타트~교체 시점까지) 스탯 한 줄이라, 선발투수의 경우
'그 스타트에서 몇 구 던지고 몇 이닝/몇 타자를 상대하다 내려갔는지'가 그대로 담겨 있다.

'등판' 컬럼은 선발이면 '선발', 구원이면 '등판회차.등판시타순'(예: '7.8' = 7회
8번타순 타자부터 상대) 형식이다. 이 의미는 KBO가 문서화한 게 아니라 문자중계
(koreabaseball.com/Game/LiveText.aspx)의 실제 교체 시점과 대조해서 확인한
것 — 예컨대 '7.8'로 기록된 선수는 실제로 7회 8번타순 타자를 상대하며
등판했음을 확인함. entry_inning/entry_batting_order 컬럼으로 미리 분리해둔다.
"""
import json
from pathlib import Path

import pandas as pd

from kbo_client import BASE, get_session, parse_innings, polite_sleep

DATA = Path(__file__).resolve().parent.parent / "data"

BOXSCORE_URL = f"{BASE}/ws/Schedule.asmx/GetBoxScoreScroll"

PITCHER_COLS = [
    "선수명", "등판", "결과", "승", "패", "세", "이닝", "타자", "투구수",
    "타수", "피안타", "홈런", "4사구", "삼진", "실점", "자책", "평균자책점",
]


def fetch_pitcher_appearances(game_id: str, season: int, le_id: int = 1, sr_id: int = 0) -> pd.DataFrame:
    session = get_session()
    payload = {"leId": le_id, "srId": sr_id, "seasonId": season, "gameId": game_id}
    r = session.post(BOXSCORE_URL, data=payload, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("code") not in ("100", 100, None):
        return pd.DataFrame(columns=["game_id", "team_side", *PITCHER_COLS])

    rows = []
    for team_side, table_wrap in enumerate(data.get("arrPitcher", [])):
        table = json.loads(table_wrap["table"])
        for row in table.get("rows", []):
            values = [c["Text"] for c in row["row"]]
            record = dict(zip(PITCHER_COLS, values))
            record["game_id"] = game_id
            record["team_side"] = "away" if team_side == 0 else "home"
            rows.append(record)

    df = pd.DataFrame(rows, columns=["game_id", "team_side", *PITCHER_COLS])
    if df.empty:
        return df

    df["결과"] = df["결과"].replace("&nbsp;", pd.NA).str.strip()
    df["이닝_소수"] = df["이닝"].map(parse_innings)
    for col in ["승", "패", "세", "타자", "투구수", "타수", "피안타", "홈런", "4사구", "삼진", "실점", "자책"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["평균자책점"] = pd.to_numeric(df["평균자책점"], errors="coerce")
    df["is_starter"] = df["등판"] == "선발"

    entry = df["등판"].where(~df["is_starter"]).str.extract(r"^(\d+)\.(\d+)$")
    df["entry_inning"] = pd.to_numeric(entry[0], errors="coerce").astype("Int64")
    df["entry_batting_order"] = pd.to_numeric(entry[1], errors="coerce").astype("Int64")
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("game_id")
    p.add_argument("season", type=int)
    a = p.parse_args()

    df = fetch_pitcher_appearances(a.game_id, a.season)
    DATA.mkdir(exist_ok=True)
    out = DATA / f"kbo_boxscore_{a.game_id}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)}행 -> {out}")
    print(df)
    polite_sleep(0)
