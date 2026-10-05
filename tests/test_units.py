"""데이터 없이도 도는 단위 테스트.

test_validation.py는 parquet이 있어야 돌고, 없으면 전부 건너뛴다.
저장소를 내려받은 사람에게는 아무것도 검증하지 않는다는 뜻이다.

여기 있는 것은 <순수 함수>만 본다. 데이터가 필요 없으므로 CI에서 항상 돈다.
고른 기준은 하나다 -- <실제로 한 번 틀렸던 곳>.
파싱 분류, 이닝 표기, 릴리스 역외삽, 주자 판정, 보직 등급.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── 타석 결과 분류 ──────────────────────────────────────────
# 이 분류에서 세 번 틀렸다: 내야안타 1,065건 누락, '땅볼로 출루' 미분류,
# 낫아웃 출루를 삼진으로 묶음.
@pytest.mark.parametrize("text,is_ab,is_hit,kind", [
    ("우익수 앞 1루타", True, True, "안타"),
    ("좌중간 2루타", True, True, "안타"),
    ("중견수 뒤 홈런", True, True, "안타"),
    ("3루수 내야안타", True, True, "안타"),          # 한 번 놓쳤던 것
    ("투수 앞 번트안타", True, True, "안타"),          # 한 번 놓쳤던 것
    ("볼넷", False, False, "볼넷·사구"),
    ("몸에 맞는 볼", False, False, "볼넷·사구"),   # KBO 표기는 '공'이 아니라 '볼'이다
    ("자동 고의4구", False, False, "볼넷·사구"),
    ("삼진 아웃", True, False, "삼진"),
    ("낫아웃 폭투 출루", True, False, "낫아웃 출루"),   # 삼진과 분리해야 한다
    ("유격수 앞 땅볼로 출루", True, False, "야수선택"),  # 한 번 놓쳤던 것
    ("2루수 실책 출루", True, False, "실책 출루"),
    ("중견수 플라이 아웃", True, False, "인플레이 아웃"),
    ("3루수 병살타", True, False, "인플레이 아웃"),
    ("투수 희생번트 아웃", False, False, "희생타"),
])
def test_classify_result(text, is_ab, is_hit, kind):
    from kbo_livetext import classify_result
    assert classify_result(text) == (is_ab, is_hit, kind)


def test_classify_result_unknown_text_falls_through():
    """모르는 표현은 '기타'로 떨어지고 타수로 세지 않는다.

    조용히 아웃이나 안타로 분류되면 리그 타율이 틀어진다 -- 실제로 그렇게 1,065건을
    놓쳤다. 분류표에 없는 새 표현이 들어오면 기타로 모여 눈에 띄어야 한다.
    """
    from kbo_livetext import classify_result
    assert classify_result("알 수 없는 새로운 표현") == (False, False, "기타")


def test_classify_result_hits_are_always_at_bats():
    """안타인데 타수가 아닌 경우는 없다."""
    from kbo_livetext import classify_result
    for t in ("1루타", "2루타", "3루타", "홈런", "내야안타", "번트안타"):
        ab, hit, _ = classify_result(f"좌익수 앞 {t}")
        assert ab and hit, t


# ── 이닝 표기 ───────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("5", 5.0), ("5 1/3", 5 + 1 / 3), ("5 2/3", 5 + 2 / 3),
    ("1/3", 1 / 3), ("2/3", 2 / 3), ("0", 0.0),
])
def test_parse_innings(text, expected):
    from kbo_client import parse_innings
    assert parse_innings(text) == pytest.approx(expected, abs=1e-9)


# ── 릴리스 역외삽 물리 ──────────────────────────────────────
# x0·z0는 홈플레이트 50피트 지점의 좌표다. 그걸 릴리스로 쓴 탓에
# 발견 22의 효과가 네 배로 부풀어 있었다.
def test_release_projection_returns_input_plane():
    """목표 평면을 y0으로 주면 입력 좌표가 그대로 나와야 한다."""
    from naver_full import _project
    q = {"x0": 1.5, "y0": 50.0, "z0": 6.0,
         "vx0": -8.0, "vy0": -130.0, "vz0": -5.0,
         "ax": 14.0, "ay": 28.0, "az": -17.0}
    x, z = _project(q, 50.0)
    assert x == pytest.approx(1.5, abs=1e-6)
    assert z == pytest.approx(6.0, abs=1e-6)


def test_release_projection_goes_backwards_and_up():
    """55피트는 공이 더 일찍 있던 자리다 -- 아직 덜 떨어졌으므로 더 높다."""
    from naver_full import _project
    q = {"x0": 1.5, "y0": 50.0, "z0": 6.0,
         "vx0": -8.0, "vy0": -130.0, "vz0": -5.0,
         "ax": 14.0, "ay": 28.0, "az": -17.0}
    _, z55 = _project(q, 55.0)
    _, z00 = _project(q, 50.0)
    assert z55 > z00


def test_release_projection_to_plate_lands_in_strike_zone():
    """같은 식으로 플레이트까지 보내면 스트라이크존 높이가 나와야 한다."""
    from naver_full import _project, PLATE_Y
    q = {"x0": 1.5, "y0": 50.0, "z0": 6.0,
         "vx0": -8.0, "vy0": -130.0, "vz0": -5.0,
         "ax": 14.0, "ay": 28.0, "az": -17.0}
    _, zp = _project(q, PLATE_Y)
    assert 0.5 < zp < 5.0, f"플레이트 높이 {zp:.2f}ft — 역외삽이 틀렸다"


def test_release_projection_handles_missing_fields():
    from naver_full import _project
    x, z = _project({"x0": 1.0}, 55.0)
    assert np.isnan(x) and np.isnan(z)


# ── 주자 판정 토큰 ──────────────────────────────────────────
# '이중도루 실패시 2루까지 진루'가 '실패'라는 글자 때문에 아웃으로 잡혔다.
def test_runner_tokens_do_not_overlap():
    from kbo_gamestate import RUNNER_OUT_TOKENS, RUNNER_SAFE_TOKENS
    for safe in RUNNER_SAFE_TOKENS:
        for out in RUNNER_OUT_TOKENS:
            assert out not in safe, f"'{safe}' 안에 '{out}'이 들어 있다"


# ── 보직 등급 ───────────────────────────────────────────────
def test_tier_grade_thresholds():
    from tiers import _grade
    got = list(_grade(np.array([0, 4, 5, 14, 15, 40])))
    assert got == ["추격조", "추격조", "중간", "중간", "필승조", "필승조"]


# ── 득점가치 항등식 (합성 데이터) ───────────────────────────
def test_run_value_telescopes_on_synthetic_inning():
    """한 반이닝의 득점가치 합 = 그 이닝 득점 - 시작 상태의 기대값.

    실제 데이터에서 이 성질이 깨져 투수 교체 시 정렬 버그를 찾았다.
    여기서는 합성 반이닝으로 산수 자체를 고정한다.
    """
    re = {(0, 0): 0.50, (1, 0): 0.27, (2, 0): 0.10, (0, 1): 0.90, (1, 1): 0.55}
    states = [(0, 0), (0, 1), (1, 1), (2, 0)]   # 안타 뒤 아웃 둘
    runs = [0, 0, 0, 0]
    total = 0.0
    for i, st in enumerate(states):
        nxt = re[states[i + 1]] if i + 1 < len(states) else 0.0
        total += runs[i] + nxt - re[st]
    assert total == pytest.approx(sum(runs) - re[states[0]], abs=1e-9)


def test_linear_weight_ordering_is_baseball_sane():
    """가중치 순서가 야구 상식과 맞는지 -- 실제 파일이 있으면 그것으로."""
    import pandas as pd
    p = ROOT / "data" / "linear_weights.parquet"
    if not p.exists():
        pytest.skip("linear_weights.parquet 없음 (실데이터 테스트에서 다룬다)")
    w = pd.read_parquet(p)["가중치"]
    assert w["안타"] > w["볼넷·사구"] > 0 > w["희생타"] > w["삼진"]
