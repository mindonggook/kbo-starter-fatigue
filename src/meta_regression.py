"""한계 4 — 효과 크기가 왜 시즌마다 다른가.

Cochran Q는 '다르다'까지만 말해준다. 왜 다른지 물으려면 시즌 수준의 설명변수에
효과 크기를 회귀시켜야 한다(메타회귀). 시즌이 셋일 때는 점이 모자라 불가능했고,
여섯이 되면서 비로소 가능해진다.

    effectₛ = β₀ + β₁ · 환경ₛ + 오차,   가중치 wₛ = 1/seₛ²

설명변수 후보는 그 시즌 자체에서 뽑는다 — 리그 타율(타고투저 정도),
피치클락 도입 여부(2024년부터), 선발 평균 투구수(운영 강도).

표본이 여섯 점뿐이라 결론을 단정하기보다 '방향과 크기가 설명되는가'를 본다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from compare_seasons import metrics

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

PITCH_CLOCK_FROM = 2024  # KBO 피치클락 도입

TARGETS = [
    ("시즌 누적 +100구 → 피안타율", "시즌 누적 피로 (+100구당 피안타율)"),
    ("1바퀴 대비 3바퀴", "타순 1→3바퀴 피안타율"),
    ("76~90구 주자 유무 강판율 차", "주자 유무 강판율 차"),
    ("5일 대 6일 휴식 (이닝)", "5일 휴식의 이닝 효과"),
]

COVARIATES = [
    ("리그타율", "리그 타율"),
    ("피치클락", "피치클락 도입 이후"),
    ("선발평균투구수", "선발 평균 투구수"),
]


def se_from_ci(d: dict) -> float:
    return (d["hi"] - d["lo"]) / (2 * 1.96)


def season_table(seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    ms = {s: metrics(s) for s in seasons}
    rows = []
    for s in seasons:
        m = ms[s]
        rows.append({
            "season": s,
            "리그타율": m["리그 타율(파싱 검산)"]["est"] if "리그 타율(파싱 검산)" in m else np.nan,
            "선발평균투구수": m["선발 평균 투구수"]["est"],
            "피치클락": 1.0 if s >= PITCH_CLOCK_FROM else 0.0,
        })
    return pd.DataFrame(rows).set_index("season"), ms


def meta_regress(effects: pd.Series, ses: pd.Series, x: pd.Series) -> dict:
    """역분산 가중 최소제곱 — 정밀한 시즌에 더 큰 무게를 준다."""
    ok = effects.notna() & ses.notna() & x.notna()
    e, s, xv = effects[ok], ses[ok], x[ok]
    if len(e) < 4 or xv.nunique() < 2:
        return {"가능": False, "n": int(len(e))}

    w = 1.0 / s ** 2
    X = sm.add_constant(xv.astype(float).values)
    fit = sm.WLS(e.astype(float).values, X, weights=w.values).fit()
    ci = fit.conf_int()[1]
    # 잔차 이질성 — 설명변수를 넣고도 남는 시즌 차이
    resid_q = float((w.values * fit.resid ** 2).sum())
    df = len(e) - 2
    return {"가능": True, "n": int(len(e)), "slope": float(fit.params[1]),
            "lo": float(ci[0]), "hi": float(ci[1]),
            "유의": bool((ci[0] > 0) == (ci[1] > 0)),
            "R2": float(fit.rsquared), "잔차Q": resid_q, "df": df}


def fig_meta(target_label: str, df: pd.DataFrame, cov: str, cov_label: str,
             res: dict, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150, facecolor=SURFACE)
    x = df[cov].astype(float)
    y = df["effect"].astype(float)
    err = df["se"].astype(float) * 1.96

    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=GRID, elinewidth=1.8, capsize=5, zorder=1)
    ax.scatter(x, y, s=[min(260, 30 + 3 / (v ** 2) * 1e-5) for v in df["se"]],
               color=BLUE, alpha=0.85, zorder=2, edgecolors="none")
    for xi, yi, s in zip(x, y, df.index):
        ax.annotate(str(s), (xi, yi), textcoords="offset points", xytext=(9, 7),
                    color=SECONDARY_INK, fontsize=9)

    if res.get("가능"):
        xs = np.linspace(x.min(), x.max(), 50)
        w = 1.0 / df["se"].astype(float) ** 2
        fit = sm.WLS(y.values, sm.add_constant(x.values), weights=w.values).fit()
        ax.plot(xs, fit.params[0] + fit.params[1] * xs, color=ORANGE, linewidth=2.2)
        note = (f"기울기 {res['slope']:+.4f} [{res['lo']:+.4f}, {res['hi']:+.4f}] · "
                f"R²={res['R2']:.2f}")
        ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top",
                color=SECONDARY_INK, fontsize=9.5)

    ax.axhline(0, color=MUTED, linewidth=1.2, linestyle="--")
    ax.set_title(f"{target_label} vs {cov_label}", color=INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel(cov_label, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("효과 크기 (95% 신뢰구간)", color=SECONDARY_INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int,
                   default=[2021, 2022, 2023, 2024, 2025, 2026])
    a = p.parse_args()
    seasons = [s for s in a.seasons
               if (DATA / f"kbo_pa_state_{s}.parquet").exists()]

    env, ms = season_table(seasons)
    lines = [f"=== 메타회귀 — 효과 크기가 시즌 환경으로 설명되는가 ({len(seasons)}시즌) ===\n"]
    lines.append("[시즌 환경]")
    for s, r in env.iterrows():
        clock = "피치클락" if r["피치클락"] else "도입 전"
        lines.append(f"  {s}: 리그타율 {r['리그타율']:.4f} · 선발 평균 {r['선발평균투구수']:.1f}구 · {clock}")
    lines.append("")

    for key, label in TARGETS:
        rows = []
        for s in seasons:
            m = ms[s].get(key)
            if m and "lo" in m:
                rows.append({"season": s, "effect": m["est"], "se": se_from_ci(m)})
        if len(rows) < 4:
            lines.append(f"[{label}] 시즌 {len(rows)}개 — 메타회귀에 부족\n")
            continue
        df = pd.DataFrame(rows).set_index("season").join(env)

        lines.append(f"[{label}]")
        for s, r in df.iterrows():
            lines.append(f"  {s}: {r['effect']:+.4f} (se {r['se']:.4f})")
        for cov, cov_label in COVARIATES:
            res = meta_regress(df["effect"], df["se"], df[cov])
            if not res.get("가능"):
                continue
            mark = "유의함" if res["유의"] else "유의하지 않음"
            lines.append(f"  ~ {cov_label}: 기울기 {res['slope']:+.5f} "
                         f"[{res['lo']:+.5f}, {res['hi']:+.5f}] {mark} · R²={res['R2']:.2f}")
            safe = key.replace(" ", "_").replace("→", "to").replace("+", "p").replace("~", "-")
            fig_meta(label, df, cov, cov_label, res, f"meta_{safe}_{cov}.png")
        lines.append("")

    out = "\n".join(lines)
    path = RESULTS / "meta_regression_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
