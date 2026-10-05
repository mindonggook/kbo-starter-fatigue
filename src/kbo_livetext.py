"""KBO 문자중계(타석별·투구별 기록) 수집과 파싱.

POST /Game/LiveTextView2.aspx 가 경기 하나의 중계 텍스트를 통째로 준다.
이닝별로 <div id="numContN">이 있고 그 안에 <span>이 이벤트 하나씩인데,
**각 div 안의 이벤트는 역순(최신이 위)으로 쌓여 있다** — 실시간 중계가 새 이벤트를
맨 위에 붙이는 방식이라 그렇다. 그래서 div 하나를 통째로 뒤집으면 시간순이 된다.

뒤집은 뒤의 한 타석은 이런 모양이다:
    8번타자 박승욱          <- 타석 시작
    1구 볼 / 2구 헛스윙 ... <- 투구 하나씩
    박승욱 : 투수 땅볼 아웃  <- 타석 결과
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA = Path(__file__).resolve().parent.parent / "data"
LIVETEXT_URL = "https://www.koreabaseball.com/Game/LiveTextView2.aspx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

RE_HALF = re.compile(r"^(\d+)회\s*(초|말)\s+(\S+)\s*공격")
RE_BATTER = re.compile(r"^(\d+)번타자\s+(\S+)$")
RE_BATTER_SUB = re.compile(r"^(\d+)번타자\s+(\S+)\s*:\s*(?:대타|대주자)?\s*(\S+)\s*\(으\)로\s*교체")
RE_PITCH = re.compile(r"^(\d+)구\s+(\S+)")
RE_PITCHER_CHANGE = re.compile(r"^투수\s+(\S+)\s*:\s*투수\s+(\S+)\s*\(으\)로\s*교체")
RE_RESULT = re.compile(r"^(\S+)\s*:\s*(.+)$")

# 결과 문자열 -> (타수로 치는가, 안타인가)
# KBO 중계는 내야안타·번트안타를 '1루타'로 쓰지 않고 그대로 적으므로 따로 받아야 한다.
HIT_TOKENS = ("1루타", "2루타", "3루타", "홈런", "내야안타", "번트안타")
WALK_TOKENS = ("볼넷", "고의4구", "자동 고의4구", "몸에 맞는 볼")
SAC_TOKENS = ("희생번트", "희생플라이", "희생 번트", "희생 플라이")


def fetch_game_text(game_id: str, season: int, session: requests.Session | None = None) -> str:
    session = session or requests.Session()
    payload = {"leagueId": "1", "seriesId": "0", "gameId": game_id, "gyear": str(season)}
    r = session.post(LIVETEXT_URL, data=payload, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.text


def _events_in_order(html: str) -> list[str]:
    """이닝 div를 순서대로 돌면서, 각 div 안은 뒤집어 시간순 이벤트 목록을 만든다."""
    soup = BeautifulSoup(html, "lxml")
    events: list[str] = []
    for idx in range(1, 31):  # 연장 포함해도 충분한 범위
        div = soup.find("div", id=f"numCont{idx}")
        if div is None:
            continue
        spans = div.find_all("span")
        for span in reversed(spans):
            # 투구 라인은 앞에 "- " 구분자와 줄바꿈·들여쓰기가 붙어 오므로 공백을 먼저 접는다.
            text = span.get_text(" ", strip=True).replace("\xa0", " ")
            text = re.sub(r"\s+", " ", text).strip(" -–")
            if text:
                events.append(text)
    return events


def classify_result(result: str) -> tuple[bool, bool, str]:
    """타석 결과 문자열 -> (타수 인정, 안타, 분류명)."""
    if any(t in result for t in HIT_TOKENS):
        return True, True, "안타"
    if any(t in result for t in WALK_TOKENS):
        return False, False, "볼넷·사구"
    if any(t in result for t in SAC_TOKENS):
        return False, False, "희생타"
    # 낫아웃은 삼진이지만 포수가 놓치면 타자가 살아나간다 — 주자 계산이 달라지므로 분리한다.
    if "낫아웃" in result and any(t in result for t in ("출루", "폭투", "포일")):
        return True, False, "낫아웃 출루"
    if "삼진" in result:
        return True, False, "삼진"
    if "실책" in result:
        return True, False, "실책 출루"
    # '유격수 앞 땅볼로 출루' = 다른 주자가 죽고 타자가 살아나간 야수선택. 타수로는 잡힌다.
    if "야수선택" in result or "필드선택" in result or "땅볼로 출루" in result:
        return True, False, "야수선택"
    if "아웃" in result or "병살" in result or "직선" in result:
        return True, False, "인플레이 아웃"
    return False, False, "기타"


def parse_game(html: str, game_id: str, away_starter: str, home_starter: str) -> pd.DataFrame:
    """타석 하나 = 행 하나인 DataFrame으로 만든다."""
    events = _events_in_order(html)

    rows: list[dict] = []
    inning = half = batting_team = None
    # 초에는 홈팀 투수가, 말에는 원정팀 투수가 던진다.
    current_pitcher = {"초": home_starter, "말": away_starter}
    pitch_total: dict[str, int] = {}
    tto_seen: dict[tuple[str, int], int] = {}

    slot = None
    batter = None
    pitches: list[str] = []

    def close_pa(result_text: str) -> None:
        nonlocal slot, batter, pitches
        if batter is None or half is None:
            return
        pitcher = current_pitcher[half]
        n_pitches = len(pitches)
        key = (pitcher, slot)
        tto_seen[key] = tto_seen.get(key, 0) + 1
        is_ab, is_hit, kind = classify_result(result_text)
        rows.append({
            "game_id": game_id,
            "inning": inning,
            "half": half,
            "batting_team": batting_team,
            "pitcher": pitcher,
            "batting_order": slot,
            "batter": batter,
            "result": result_text,
            "종류": kind,
            "타수": is_ab,
            "안타": is_hit,
            "타석_투구수": n_pitches,
            "투수_누적투구수": pitch_total.get(pitcher, 0),
            "타순회전": tto_seen[key],
        })
        slot = batter = None
        pitches = []

    def flush_pending() -> None:
        """타석이 결과 없이 끝나도(도루자·견제사로 이닝 종료 등) 던진 공은 이미 집계돼 있다."""
        nonlocal slot, batter, pitches
        slot = batter = None
        pitches = []

    for text in events:
        m = RE_HALF.match(text)
        if m:
            flush_pending()
            inning, half, batting_team = int(m.group(1)), m.group(2), m.group(3)
            continue

        m = RE_PITCHER_CHANGE.match(text)
        if m and half is not None:
            current_pitcher[half] = m.group(2)
            continue

        m = RE_BATTER_SUB.match(text)
        if m:
            if slot == int(m.group(1)):
                batter = m.group(3)
            continue

        m = RE_BATTER.match(text)
        if m:
            slot, batter = int(m.group(1)), m.group(2)
            pitches = []
            continue

        m = RE_PITCH.match(text)
        if m:
            # 타석 도중 교체가 일어나면 공 하나하나를 그때 마운드에 있던 투수에게 붙여야 한다.
            if half is not None:
                mound = current_pitcher[half]
                pitch_total[mound] = pitch_total.get(mound, 0) + 1
            pitches.append(m.group(2))
            continue

        m = RE_RESULT.match(text)
        if m and batter is not None and m.group(1) == batter:
            close_pa(m.group(2))

    flush_pending()
    df = pd.DataFrame(rows)
    # 타석 도중 교체 때문에 '타석_투구수' 합계는 투수별 실제 투구수와 어긋날 수 있어 따로 싣는다.
    df.attrs["pitch_total"] = pitch_total
    return df


def fetch_and_parse(game_id: str, season: int, away_starter: str, home_starter: str,
                    session: requests.Session | None = None, sleep: float = 0.4) -> pd.DataFrame:
    html = fetch_game_text(game_id, season, session)
    time.sleep(sleep)
    return parse_game(html, game_id, away_starter, home_starter)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("game_id")
    p.add_argument("season", type=int)
    a = p.parse_args()

    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{a.season}.parquet")
    g = box[(box["game_id"] == a.game_id) & box["is_starter"]]
    away = g[g["team_side"] == "away"]["선수명"].iloc[0]
    home = g[g["team_side"] == "home"]["선수명"].iloc[0]

    df = fetch_and_parse(a.game_id, a.season, away, home)
    out = DATA / f"livetext_{a.game_id}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(df)} 타석 -> {out}")
