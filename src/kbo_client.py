"""KBO 공식 기록실(koreabaseball.com) 공용 HTTP 클라이언트/파싱 유틸.

이 사이트의 GameCenter는 서버 렌더링 HTML이 아니라 ajax(.asmx) 엔드포인트가
JSON으로 표(rows/headers)를 내려주고 JS가 그걸 테이블에 꽂아 넣는 구조다.
그래서 requests로 곧장 두 엔드포인트만 호출하면 된다.

- 일정: POST /ws/Schedule.asmx/GetScheduleList   (월 단위, 하루에 여러 경기)
- 박스스코어: POST /ws/Schedule.asmx/GetBoxScoreScroll  (경기 하나, 투수/타자 기록표)
"""
import re
import time
from dataclasses import dataclass

import requests

BASE = "https://www.koreabaseball.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE}/Schedule/Schedule.aspx",
}

# gameId 안의 2글자 팀 코드 -> 팀명. (예: 20240402LTHH0 -> 롯데 vs 한화)
TEAM_CODE = {
    "LT": "롯데", "HH": "한화", "NC": "NC", "LG": "LG", "WO": "키움",
    "SS": "삼성", "OB": "두산", "SK": "SSG", "HT": "KIA", "KT": "KT",
}


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "").replace("&nbsp;", " ").strip()


def parse_innings(text: str) -> float:
    """'6' / '5 2/3' / '1 1/3' / '2/3' / '1/3' -> 소수 이닝(아웃수/3 반영)."""
    text = (text or "").strip()
    if not text:
        return 0.0
    m = re.match(r"^(?:(\d+)\s+)?(\d+)/(\d+)$", text)
    if m:
        whole, num, den = m.groups()
        return (int(whole) if whole else 0) + int(num) / int(den)
    try:
        return float(text)
    except ValueError:
        return 0.0


def polite_sleep(seconds: float = 0.4) -> None:
    time.sleep(seconds)


@dataclass
class GameRef:
    game_id: str
    date: str  # YYYYMMDD
    away_team: str
    home_team: str
    away_score: int | None
    home_score: int | None
