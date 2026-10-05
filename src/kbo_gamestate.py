"""문자중계에서 타석마다의 경기 상태(점수·아웃·주자 수)를 복원한다.

주자를 '누가 몇 루에 있는지'까지 추적하면 규칙이 금방 복잡해지고 검증도 어렵다.
대신 한 이닝 안에서 세 가지만 세면 주자 '수'는 정확히 나온다:

    현재 주자 수 = 출루한 타자 수 - 홈인한 주자 수 - 주자가 아웃된 수

홈런은 출루(+1)와 홈인(+1)이 동시에 일어나므로 이 식에서 자연히 상쇄된다.
검증은 세 가지로 한다 — 최종 점수가 공식 기록과 맞는지, 완료된 하프이닝의
아웃이 정확히 3인지, 주자 수가 0~3을 벗어나지 않는지.
"""
import re

import pandas as pd

from kbo_livetext import (RE_BATTER, RE_BATTER_SUB, RE_HALF, RE_PITCH,
                          RE_PITCHER_CHANGE, RE_RESULT, _events_in_order,
                          classify_result)

# 'N루주자 X : ...' / '타자주자 X : ...' 형태의 주자 관련 이벤트
RE_RUNNER = re.compile(r"^(?:\d루주자|타자주자)\s+\S+\s*:\s*(.+)$")

RUNNER_OUT_TOKENS = ("아웃", "견제사")
# '이중도루 실패시 2루까지 진루'처럼 실패·아웃이라는 말이 들어가도 실제로는 살아서
# 진루한 경우가 있다. 진루나 홈인이 적혀 있으면 그 주자는 죽지 않은 것이다.
RUNNER_SAFE_TOKENS = ("진루", "홈인")
OUT_RESULT_KINDS = ("인플레이 아웃", "삼진", "희생타")
REACH_RESULT_KINDS = ("안타", "볼넷·사구", "실책 출루", "야수선택", "낫아웃 출루")

PA_COLS = [
    "game_id", "inning", "half", "batting_team", "pitcher", "is_starter",
    "batting_order", "batter", "result", "종류", "타수", "안타",
    "타석_투구수", "투수_누적투구수", "타순회전",
    "아웃카운트", "주자수", "점수_투수팀", "점수_타자팀", "점수차_투수팀기준",
]


def parse_game_with_state(html: str, game_id: str, away_starter: str,
                          home_starter: str) -> pd.DataFrame:
    events = _events_in_order(html)

    rows: list[dict] = []
    inning = half = batting_team = None
    current_pitcher = {"초": home_starter, "말": away_starter}
    pitch_total: dict[str, int] = {}
    tto_seen: dict[tuple[str, int], int] = {}
    runs = {"초": 0, "말": 0}          # 그 반이닝에 공격하는 팀의 누적 득점
    other = {"초": "말", "말": "초"}

    outs = 0
    reached = scored = runner_outs = 0
    slot = batter = None
    pitches: list[str] = []
    problems: list[str] = []
    half_out_log: list[int] = []

    def reset_half() -> None:
        nonlocal outs, reached, scored, runner_outs, slot, batter, pitches
        outs = reached = scored = runner_outs = 0
        slot = batter = None
        pitches = []

    def close_pa(result_text: str) -> None:
        nonlocal slot, batter, pitches, outs, reached, runner_outs
        if batter is None or half is None:
            return
        pitcher = current_pitcher[half]
        key = (pitcher, slot)
        tto_seen[key] = tto_seen.get(key, 0) + 1
        is_ab, is_hit, kind = classify_result(result_text)

        runners_before = reached - scored - runner_outs
        rows.append({
            "game_id": game_id, "inning": inning, "half": half,
            "batting_team": batting_team, "pitcher": pitcher,
            "batting_order": slot, "batter": batter, "result": result_text,
            "종류": kind, "타수": is_ab, "안타": is_hit,
            "타석_투구수": len(pitches), "투수_누적투구수": pitch_total.get(pitcher, 0),
            "타순회전": tto_seen[key],
            "아웃카운트": outs, "주자수": runners_before,
            "점수_투수팀": runs[other[half]], "점수_타자팀": runs[half],
            "점수차_투수팀기준": runs[other[half]] - runs[half],
        })

        # 병살·삼중살로 함께 죽는 주자는 별도의 '주자 : 포스아웃' 이벤트로 다시 들어오므로
        # 여기서 더하면 이중 계산이 된다(실측으로 확인).
        if kind in REACH_RESULT_KINDS:
            reached += 1
        elif kind in OUT_RESULT_KINDS:
            outs += 1
        slot = batter = None
        pitches = []

    for text in events:
        m = RE_HALF.match(text)
        if m:
            if half is not None:
                half_out_log.append(outs)
            inning, half, batting_team = int(m.group(1)), m.group(2), m.group(3)
            reset_half()
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
            if half is not None:
                mound = current_pitcher[half]
                pitch_total[mound] = pitch_total.get(mound, 0) + 1
            pitches.append(m.group(2))
            continue

        m = RE_RUNNER.match(text)
        if m and half is not None:
            what = m.group(1)
            if "교체" in what:
                continue  # 대주자 투입은 상태를 바꾸지 않는다
            if "홈인" in what:
                runs[half] += 1
                scored += 1
            elif (any(tok in what for tok in RUNNER_OUT_TOKENS)
                  and not any(tok in what for tok in RUNNER_SAFE_TOKENS)):
                outs += 1
                runner_outs += 1
            continue

        m = RE_RESULT.match(text)
        if m and batter is not None and m.group(1) == batter:
            result_text = m.group(2)
            close_pa(result_text)
            if "홈런" in result_text and half is not None:
                # 타자 본인의 득점은 주자 이벤트로 나오지 않으므로 여기서 더한다.
                runs[half] += 1
                reached += 1
                scored += 1

    if half is not None:
        half_out_log.append(outs)

    df = pd.DataFrame(rows)
    df.attrs["pitch_total"] = pitch_total
    df.attrs["final_runs"] = {"away": runs["초"], "home": runs["말"]}
    df.attrs["half_outs"] = half_out_log
    df.attrs["problems"] = problems
    return df


def validate(df: pd.DataFrame, away_score: int, home_score: int) -> dict:
    """세 가지 검증 — 최종 점수 일치, 하프이닝 아웃 3개, 주자 수 0~3."""
    final = df.attrs.get("final_runs", {})
    half_outs = df.attrs.get("half_outs", [])
    # 마지막 하프이닝은 끝내기·경기종료로 3아웃이 아닐 수 있어 제외한다.
    completed = half_outs[:-1] if half_outs else []
    bad_runners = df[(df["주자수"] < 0) | (df["주자수"] > 3)] if not df.empty else df

    return {
        "score_ok": final.get("away") == away_score and final.get("home") == home_score,
        "tracked_away": final.get("away"), "tracked_home": final.get("home"),
        "outs_ok": all(o == 3 for o in completed) if completed else False,
        "bad_out_halves": sum(1 for o in completed if o != 3),
        "runner_ok": len(bad_runners) == 0,
        "bad_runner_pa": len(bad_runners),
    }
