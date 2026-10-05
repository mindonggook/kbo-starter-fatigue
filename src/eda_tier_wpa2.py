"""3차 평가 ① — WPA를 직접 재는 대신 '실점 × 그 상황의 환율'로 환산한다.

평가의 지적이 정확하다. 평균 WPA를 역할끼리 맞대면 레버리지가 섞인다.
WPA의 크기는 등판 상황의 무게에 비례하는데 필승조는 1.42배, 추격조는 0.68배다.
그래서 같은 실력이어도 레버리지가 낮은 쪽은 손해가 작게 찍힌다 --
두 점추정의 부호가 실점 결과와 정반대로 나온 것이 그 편향의 지문이다.

대안: 타석마다 실점 기여를 그 상황의 <환율>로 곱해 승률로 바꾼다.

    y_i = (-득점가치_i) x (그 점수차 구간의 1점당 승률)

실현된 WPA의 잡음을 '결정론적 환율'로 바꾸는 것이라 분산이 훨씬 작다.
대신 가정이 하나 붙는다 -- 구간 안에서 환율이 일정하다고 본다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import tiers as T
from eda_tier_wpa import load, roles, BASE

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#7b5bd6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SIG = {True: "유의함", False: "유의하지 않음"}
BANDS = [(-1, 0, "동점"), (1, 2, "1~2점"), (3, 4, "3~4점"), (5, 99, "5점 이상")]


def band_of(diff):
    out = pd.Series(index=diff.index, dtype=object)
    a = diff.abs()
    for lo, hi, lab in BANDS:
        out[(a > lo) & (a <= hi)] = lab
    return out


def exchange_rates(m):
    """점수차 구간마다 '1점당 승률' -- 5~8회에서 추정한다."""
    mid = m[m["inning"].between(5, 8)]
    rates = {}
    for _, _, lab in BANDS:
        g = mid[mid["구간"] == lab]
        if len(g) > 500:
            rates[lab] = float(np.polyfit(-g["득점가치"], g["WPA"], 1)[0])
    return rates


def fit_roles(d, y):
    X = pd.DataFrame(index=d.index)
    cols = []
    for t in T.TIERS:
        key = f"구원 {t}"
        if (d["역할"] == key).any():
            X[key] = (d["역할"] == key).astype(float)
            cols.append(key)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = sm.add_constant(X)
    f = sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["game_id"]})
    return f, cols


def gain(f, key, scale=900, pitcher_view=True):
    """교체 이득.

    부호 주의 -- 두 지표의 관점이 반대다.
      득점가치 : 타자팀 관점(클수록 투수에게 나쁘다) -> 이득 = -계수
      WPA·환산 : 투수팀 관점(클수록 투수에게 좋다)   -> 이득 = +계수
    """
    ci = f.conf_int().loc[key]
    if pitcher_view:
        return f.params[key] * scale, ci[0] * scale, ci[1] * scale, (ci[0] > 0) == (ci[1] > 0)
    return -f.params[key] * scale, -ci[1] * scale, -ci[0] * scale, (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2018, 2027)))
    a = p.parse_args()

    pa = load(a.seasons)
    rv = pd.read_parquet(DATA / "kbo_pa_runvalue.parquet",
                         columns=["game_id", "inning", "half", "batting_order",
                                  "batter", "득점가치"])
    m = pa.merge(rv, on=["game_id", "inning", "half", "batting_order", "batter"],
                 how="inner")
    m["구간"] = band_of(m["점수차_투수팀기준"])
    rates = exchange_rates(m)
    # 환율을 곱해 '예상 승률 기여'로 바꾼다.
    m["환율"] = m["구간"].map(rates)
    m = m.dropna(subset=["환율", "득점가치"])
    m["예상승률기여"] = -m["득점가치"] * m["환율"]

    tier_df = T.build_pit(a.seasons).dropna(subset=["등급"])

    L = ["=== 교체 이득을 '실점 × 환율'로 승률 환산 (9시즌) ===",
         "평균 WPA를 직접 맞대면 레버리지가 섞인다. 그 대신 타석마다",
         "실점 기여에 그 점수차 구간의 1점당 승률을 곱해 승률로 바꿨다.",
         "단위: 9타자당 승리확률 %p · 경기 단위 군집보정\n"]

    L.append("[환율] 5~8회 · 점수차 구간별 1점당 승률")
    for lab, r in rates.items():
        n = (m["구간"] == lab).sum()
        L.append(f"  {lab:8s} {r*100:5.2f}%p (n={n:,})")
    L.append("")

    for cut_lab, sub in (("5~8회 전체", m),
                         ("접전 (2점 이내)", m[m["점수차_투수팀기준"].abs() <= 2])):
        both = roles(sub, tier_df)
        f_rv, cols = fit_roles(both, "득점가치")
        f_wp, _ = fit_roles(both, "예상승률기여")
        f_raw, _ = fit_roles(both, "WPA")
        n = both.groupby("역할").size()
        L.append(f"[{cut_lab}]")
        L.append("  " + "교체 대상".ljust(10)
                 + f"{'실점(점)':>18s}{'환산 승률(%p)':>24s}{'직접 WPA(%p)':>20s}")
        for c in cols:
            g_rv = gain(f_rv, c, 9, pitcher_view=False)
            g_wp = gain(f_wp, c, 900)
            g_raw = gain(f_raw, c, 900)
            L.append("  " + c.replace("구원 ", "").ljust(10)
                     + f"{g_rv[0]:>+9.3f}{'*' if g_rv[3] else ' '}"
                     + f" [{g_rv[1]:+.3f},{g_rv[2]:+.3f}]"
                     + f"{g_wp[0]:>+9.3f}{'*' if g_wp[3] else ' '}"
                     + f" [{g_wp[1]:+.3f},{g_wp[2]:+.3f}]"
                     + f"{g_raw[0]:>+9.3f}{'*' if g_raw[3] else ' '}")
        L.append(f"  (* 유의 · n: " + " / ".join(
            f"{c.replace('구원 ','')} {n.get(c,0):,}" for c in cols) + ")")
        # 구간 폭 비교 — 환산이 얼마나 덜 흔들리나
        for c in cols[:1]:
            w_wp = gain(f_wp, c, 900)[2] - gain(f_wp, c, 900)[1]
            w_raw = gain(f_raw, c, 900)[2] - gain(f_raw, c, 900)[1]
            L.append(f"  신뢰구간 폭: 환산 {w_wp:.3f}%p vs 직접 WPA {w_raw:.3f}%p "
                     f"({w_raw/w_wp:.1f}배 좁다)")
        L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150, facecolor=SURFACE)
    for ax, (cut_lab, sub) in zip(axes, (("5~8회 전체", m),
                                         ("접전 (2점 이내)",
                                          m[m["점수차_투수팀기준"].abs() <= 2]))):
        both = roles(sub, tier_df)
        f_wp, cols = fit_roles(both, "예상승률기여")
        f_raw, _ = fit_roles(both, "WPA")
        x = np.arange(len(cols))
        for off, (f_, color, lab) in ((-0.19, (f_wp, VIOLET, "실점 × 환율")),
                                      (0.19, (f_raw, MUTED, "직접 WPA"))):
            vals = [gain(f_, c, 900)[0] for c in cols]
            errs = [gain(f_, c, 900)[0] - gain(f_, c, 900)[1] for c in cols]
            ax.bar(x + off, vals, color=color, width=0.36, label=lab)
            ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK,
                        elinewidth=1.2, capsize=4)
        ax.axhline(0, color=INK, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("구원 ", "") + "로\n교체" for c in cols], fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
        ax.set_title(cut_lab, color=INK, fontsize=12, loc="left", pad=10)
        ax.set_ylabel("9타자당 승리확률 (%p)", color=SECONDARY_INK, fontsize=9.5)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_facecolor(SURFACE)
    fig.suptitle("레버리지를 걷어내고 승률로 환산하면", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "tier_wpa2.png")
    plt.close(fig)

    path = RESULTS / "eda_tier_wpa2_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
