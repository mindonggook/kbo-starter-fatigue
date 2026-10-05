"""홀드아웃 검정을 위한 사전 등록 — 2026을 빼고 예측값을 고정한다.

외부 평가의 여섯 번째 지적이다. 발견 13~36은 열 시즌 전체를 보면서 설계했으므로,
'2024에서 만든 방법론이 아홉 시즌에 통과했다'(발견 12)는 더 이상 깨끗한 검정이 아니다.
백테스트를 돌리면서 전략을 고친 것과 같다.

유일한 해법은 <아직 보지 않은 데이터>에 미리 고정한 예측을 맞히는 것이다.
2026 시즌은 10월 9일에 끝나고 현재 684경기까지만 반영돼 있다.

  예측값은 2017~2025 아홉 시즌<만>으로 계산한다. 2026은 한 숫자도 쓰지 않는다.
  성공 기준은 지금 적는다 -- 나중에 고치면 사전 등록이 아니다.
  구간은 랜덤효과 예측구간 m ± 1.96·sqrt(se² + τ²)로 잡는다.
    '새로운 한 시즌'이 떨어질 범위이므로 평균의 신뢰구간보다 넓어야 맞다.

정직하게 적을 것: 나는 2026 데이터를 이미 봤다(684경기). 그러므로 이것은
완전한 홀드아웃이 아니라 <예측값이 2026에 오염되지 않은> 준홀드아웃이다.
완전한 검정은 2027 시즌이다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import eda_runvalue_sweep as SW

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN = list(range(2017, 2026))      # 2026은 쓰지 않는다


def predictive_interval(per):
    """새 시즌 하나가 떨어질 구간 — 랜덤효과 예측구간."""
    p = SW.pool(per)
    if not np.isfinite(p["est"]):
        return None
    sd = np.sqrt(p["re_se"] ** 2 + p["tau2"])
    return dict(est=p["re_est"], lo=p["re_est"] - 1.96 * sd,
                hi=p["re_est"] + 1.96 * sd, tau2=p["tau2"], k=p["k"])


def hook_runner_rate(seasons):
    """강판 시점에 주자가 있던 비율 — 시즌마다 하나씩."""
    out = []
    for s in seasons:
        pa_path = DATA / f"kbo_pa_state_{s}.parquet"
        chk_path = DATA / f"kbo_state_check_{s}.parquet"
        if not pa_path.exists():
            continue
        pa = pd.read_parquet(pa_path)
        chk = pd.read_parquet(chk_path)
        ok = set(chk.loc[chk["score_ok"] & chk["starter_ok"], "game_id"])
        d = pa[pa["game_id"].isin(ok) & pa["is_starter"]].copy()
        d["start"] = d["game_id"].astype(str) + "_" + d["pitcher"].astype(str)
        d["seq"] = np.arange(len(d))
        last = d.loc[d.groupby("start")["seq"].idxmax()]
        p = float((last["주자수"] > 0).mean())
        out.append((p, float(np.sqrt(p * (1 - p) / len(last)))))
    return out


def main():
    pa = SW.load()
    pa = pa[pa["season"].isin(TRAIN)]

    rows = []
    for code, label, fn, scale, unit in SW.METRICS:
        for y, ylab in (("안타", "피안타율"), ("득점가치", "득점가치")):
            per = [fn(pa[pa["season"] == s], y) for s in TRAIN]
            pi = predictive_interval(per)
            if pi:
                rows.append((f"{label} · {ylab}", pi, scale, unit))

    runner = hook_runner_rate(TRAIN)
    pi_r = predictive_interval(runner)

    L = ["=== 사전 등록 — 2026 완주 시즌 홀드아웃 검정 ===",
         f"작성 시각 기준 데이터: 2017~2025 아홉 시즌 (2026은 한 숫자도 쓰지 않았다)",
         "구간은 랜덤효과 예측구간 — '새 시즌 하나'가 떨어질 범위다.",
         "성공 기준: 2026 완주 시즌의 추정치가 아래 구간 안에 들어오면 통과.\n"]

    L.append("[예측값]")
    for name, pi, scale, unit in rows:
        L.append(f"  {name:36s} {pi['est']*scale:+8.2f}{unit} "
                 f"[{pi['lo']*scale:+7.2f}, {pi['hi']*scale:+7.2f}] "
                 f"(τ²={pi['tau2']*scale**2:.2f}, k={pi['k']})")
    if pi_r:
        L.append(f"  {'강판 시 주자 있던 비율':36s} {pi_r['est']*100:+8.2f}% "
                 f"[{pi_r['lo']*100:+7.2f}, {pi_r['hi']*100:+7.2f}] "
                 f"(τ²={pi_r['tau2']*1e4:.2f}, k={pi_r['k']})")
    L.append("")

    L.append("[같이 고정하는 방향 예측]")
    L.append("  1. 타순 1→3바퀴 효과는 2026에서도 유의하게 양수다.")
    L.append("  2. 5일 이하 휴식 효과는 2026에서도 유의하지 않다(0을 포함한다).")
    L.append("  3. 강판 시 주자 있던 비율은 70~78% 안에 있다.")
    L.append("  4. 강판 확률 모델(2017~2025 학습)의 2026 AUC는 0.93~0.97이다.")
    L.append("  5. 투구수 효과는 타순 회전 효과보다 작다.")
    L.append("")
    L.append("[검정 방법]")
    L.append("  10월 9일 시즌 종료 후 2026 전 경기를 다시 수집하고,")
    L.append("  eda_runvalue_sweep.py --seasons 2026 으로 같은 함수를 그대로 돌린다.")
    L.append("  스크립트를 고치지 않는다 — 고치면 사전 등록이 아니다.")
    L.append("")
    L.append("[유보]")
    L.append("  나는 2026 데이터를 이미 봤다(684경기). 예측값 계산에서 제외했을 뿐이다.")
    L.append("  그러므로 이것은 완전한 홀드아웃이 아니라 '예측값이 오염되지 않은' 준홀드아웃이다.")
    L.append("  완전히 깨끗한 검정은 2027 시즌이며, 위 구간을 그대로 2027에도 적용한다.")

    out = DATA.parent / "prereg.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
