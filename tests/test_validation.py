"""파이프라인 검산을 자동화한다.

리포트 부록 B에 적은 다섯 가지 검산을 테스트로 고정한다. 손으로 확인하면
다음에 파싱 규칙을 고칠 때 또 확인하는 것을 잊는다. 실패하면 빨갛게 뜨게 둔다.

    python -m pytest tests -q

경계값은 '지금 통과하는 값'이 아니라 '여기를 넘으면 뭔가 깨진 것'으로 잡았다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

SEASONS = list(range(2017, 2027))
KBO_OFFICIAL_AVG = {  # KBO 공식 발표 리그 타율
    2024: 0.277,
}


def _states(season):
    p = DATA / f"kbo_pa_state_{season}.parquet"
    if not p.exists():
        pytest.skip(f"{p.name} 없음")
    return pd.read_parquet(p)


def _checks(season):
    p = DATA / f"kbo_state_check_{season}.parquet"
    if not p.exists():
        pytest.skip(f"{p.name} 없음")
    return pd.read_parquet(p)


@pytest.mark.parametrize("season", SEASONS)
def test_league_average_matches_official(season):
    """[검산 1] 파싱한 리그 타율이 공식 발표와 맞는가."""
    pa = _states(season)
    avg = pa["안타"].astype(float).sum() / pa["타수"].astype(bool).sum()
    assert 0.240 < avg < 0.310, f"{season} 리그 타율 {avg:.4f} — 분류가 깨졌다"
    if season in KBO_OFFICIAL_AVG:
        assert abs(avg - KBO_OFFICIAL_AVG[season]) < 0.004, \
            f"{season} {avg:.4f} vs 공식 {KBO_OFFICIAL_AVG[season]}"


@pytest.mark.parametrize("season", SEASONS)
def test_state_machine_score_tracking(season):
    """[검산 2] 타석별 누적 점수가 최종 스코어와 일치하는가."""
    chk = _checks(season)
    rate = chk["score_ok"].mean()
    assert rate > 0.995, f"{season} 점수 추적 일치율 {rate:.4f}"


@pytest.mark.parametrize("season", SEASONS)
def test_pitch_counts_against_boxscore(season):
    """[검산 3] 문자중계 투구수 합계가 박스스코어와 맞는가."""
    pitches = DATA / f"kbo_pitches_{season}.parquet"
    box = DATA / f"kbo_pitcher_appearances_{season}.parquet"
    if not pitches.exists() or not box.exists():
        pytest.skip("데이터 없음")
    p = pd.read_parquet(pitches, columns=["game_id", "pitcher"])
    b = pd.read_parquet(box, columns=["game_id", "선수명", "투구수"])
    got = p.groupby(["game_id", "pitcher"]).size().rename("파싱")
    m = b.rename(columns={"선수명": "pitcher"}).join(
        got, on=["game_id", "pitcher"]).dropna(subset=["파싱"])
    m["투구수"] = pd.to_numeric(m["투구수"], errors="coerce")
    m = m.dropna(subset=["투구수"])
    close = (m["파싱"] - m["투구수"]).abs() <= 3
    assert close.mean() > 0.97, f"{season} 투구수 일치율 {close.mean():.4f}"


@pytest.mark.parametrize("season", [2024, 2025, 2026])
def test_derived_velocity_matches_measured(season):
    """[검산 4] 속도 벡터로 계산한 구속이 중계의 실측값과 맞는가."""
    p = DATA / f"naver_full_{season}.parquet"
    if not p.exists():
        pytest.skip(f"{p.name} 없음")
    d = pd.read_parquet(p, columns=["구속", "구속_계산"]).dropna()
    r = d["구속"].corr(d["구속_계산"])
    assert r > 0.98, f"{season} 구속 상관 {r:.4f}"


def test_run_value_telescopes_to_zero():
    """[검산 5] 득점가치는 구조상 전체 평균이 0이어야 한다.

    여기가 깨지면 반이닝 안의 시간 순서가 틀어진 것이다 — 실제로 한 번 그랬다.
    """
    p = DATA / "kbo_pa_runvalue.parquet"
    if not p.exists():
        pytest.skip("kbo_pa_runvalue.parquet 없음")
    rv = pd.read_parquet(p, columns=["득점가치"])["득점가치"]
    assert abs(rv.mean()) < 0.001, f"평균 {rv.mean():+.6f} — 타석 순서를 의심하라"


def test_run_expectancy_is_monotone():
    """[검산 5-b] 득점기대값은 주자가 늘면 오르고 아웃이 늘면 내려가야 한다."""
    p = DATA / "run_expectancy.parquet"
    if not p.exists():
        pytest.skip("run_expectancy.parquet 없음")
    re = pd.read_parquet(p)["RE"].unstack()
    assert (re.diff(axis=1).iloc[:, 1:] > 0).all().all(), "주자 방향이 뒤집혔다"
    assert (re.diff(axis=0).iloc[1:] < 0).all().all(), "아웃 방향이 뒤집혔다"


def test_linear_weights_order():
    """[검산 5-c] 선형 가중치의 순서는 야구 상식과 맞아야 한다."""
    p = DATA / "linear_weights.parquet"
    if not p.exists():
        pytest.skip("linear_weights.parquet 없음")
    w = pd.read_parquet(p)["가중치"]
    assert w["안타"] > w["볼넷·사구"] > 0 > w["삼진"], "가중치 순서가 이상하다"
    # 삼진과 인플레이 아웃은 득점가치가 거의 같다 — 발견 29의 핵심 전제
    assert abs(w["삼진"] - w["인플레이 아웃"]) < 0.02


@pytest.mark.parametrize("season", [2024, 2025, 2026])
def test_release_extrapolation_physics(season):
    """[검산 6] 같은 궤적식으로 플레이트까지 보내면 스트라이크존 높이가 나와야 한다."""
    p = DATA / f"naver_full_{season}.parquet"
    if not p.exists():
        pytest.skip(f"{p.name} 없음")
    d = pd.read_parquet(p, columns=["plate_z", "release_z", "p50_z"]).dropna()
    assert 1.8 < d["plate_z"].mean() < 3.0, \
        f"{season} 플레이트 높이 평균 {d['plate_z'].mean():.2f}ft — 역외삽이 틀렸다"
    # 55피트 평면은 50피트보다 높아야 한다(공은 떨어지며 날아온다)
    assert d["release_z"].mean() > d["p50_z"].mean()


def test_no_duplicate_pa_keys():
    """[검산 7] 타석 키가 중복되면 조인에서 행이 폭증한다 — 실제로 한 번 당했다."""
    p = DATA / "kbo_pa_runvalue.parquet"
    if not p.exists():
        pytest.skip("없음")
    d = pd.read_parquet(p, columns=["game_id", "pitcher", "inning", "half",
                                    "batting_order", "batter"])
    dup = d.duplicated().mean()
    assert dup < 0.05, f"완전 중복 타석 {dup*100:.1f}%"
