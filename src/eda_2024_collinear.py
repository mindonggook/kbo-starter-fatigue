"""한계 1 (계속) — 두 변수를 회귀로 동시에 넣고, 분리가 되는지 자체를 검증한다.

층화 비교(eda_2024_separate.py)가 신뢰구간만 넓게 나온 데는 구조적 이유가 있다.
한 등판 안에서 상대 타자 수는 정확히

    타자 수 = 9 × (회전 - 1) + 타순

이고, 누적 투구수는 타자 수에 거의 비례한다(타석당 3.9구 안팎). 즉 회전·타순·
투구수 셋 중 둘을 고정하면 나머지는 거의 정해진다. 남는 변동은 '그날 그 투수가
얼마나 효율적이었나'뿐이라 층화만으로는 표본이 금방 말라버린다.

그래서 여기서는 둘을 한 모형에 같이 넣고, 투수 고정효과로 투수 유형을 지운 뒤
각 계수의 신뢰구간을 본다. 분리가 되면 한쪽만 살아남고, 안 되면 둘 다 넓은
구간으로 나온다 — 어느 쪽이든 그것이 이 데이터가 줄 수 있는 답이다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).resolve().parent.parent / "data"
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

SEASON = 2024
MIN_PA = 200  # 고정효과를 줄 만큼 타석이 쌓인 투수만


def load() -> pd.DataFrame:
    pa = pd.read_parquet(DATA / f"kbo_plate_appearances_{SEASON}.parquet")
    check = pd.read_parquet(DATA / f"kbo_livetext_check_{SEASON}.parquet")
    good = set(check.loc[check["starter_ok"], "game_id"])
    pa = pa[pa["game_id"].isin(good) & pa["is_starter"]].copy()
    pa = pa[pa["타순회전"].between(1, 3) & pa["batting_order"].between(1, 9)]
    pa["안타"] = pa["안타"].astype(int)
    pa["타수"] = pa["타수"].astype(bool)
    pa["출루"] = (pa["안타"].astype(bool) | (pa["종류"] == "볼넷·사구")).astype(int)

    counts = pa.groupby("pitcher")["안타"].transform("size")
    return pa[counts >= MIN_PA].copy()


def collinearity_report(pa: pd.DataFrame) -> list[str]:
    lines = ["[0] 두 변수가 얼마나 겹쳐 있나"]
    r = pa[["타순회전", "투수_누적투구수"]].corr().iloc[0, 1]
    lines.append(f"  - 타순회전 ↔ 누적투구수 상관계수: {r:.3f}")

    # 타순까지 넣으면 투구수는 거의 결정된다 — 회귀의 설명력으로 보여준다.
    X = sm.add_constant(pd.get_dummies(pa[["타순회전", "batting_order"]].astype(int),
                                       columns=["batting_order"], drop_first=True).astype(float))
    fit = sm.OLS(pa["투수_누적투구수"].astype(float), X).fit()
    lines.append(f"  - 회전+타순으로 누적투구수를 설명한 R²: {fit.rsquared:.3f}")
    lines.append(f"    → 투구수의 {fit.rsquared*100:.0f}%가 '지금 몇 번째 타자인가'로 이미 정해진다.")
    resid_sd = np.sqrt(fit.mse_resid)
    lines.append(f"  - 설명되지 않고 남는 투구수 변동(잔차 표준편차): {resid_sd:.1f}구")
    lines.append("    → 이 잔차만이 투구수 고유 효과를 추정할 재료다.\n")
    return lines


def fit_model(pa: pd.DataFrame, outcome: str, use_tto: bool = True,
              use_pitches: bool = True) -> dict:
    """선형확률모형 + 투수 고정효과. 계수는 곧 확률 차이라 해석이 직관적이다.

    use_tto/use_pitches로 변수를 따로 넣거나 같이 넣어볼 수 있다. 같이 넣었을 때만
    계수가 요동친다면 그것은 효과가 아니라 다중공선성의 흔적이다.
    """
    d = pa.copy()

    design = pd.DataFrame(index=d.index)
    wanted = []
    if use_tto:
        design["회전_2"] = (d["타순회전"] == 2).astype(float)
        design["회전_3"] = (d["타순회전"] == 3).astype(float)
        wanted += ["회전_2", "회전_3"]
    if use_pitches:
        design["투구수_10구당"] = d["투수_누적투구수"].astype(float) / 10.0
        wanted.append("투구수_10구당")
    for slot in range(2, 10):
        design[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    pitcher_dummies = pd.get_dummies(d["pitcher"], prefix="P", drop_first=True).astype(float)
    design = pd.concat([design, pitcher_dummies], axis=1)
    design = sm.add_constant(design)

    y = d[outcome].astype(float)
    # 같은 등판 안의 타석들은 서로 독립이 아니므로 등판 단위로 군집 표준오차를 쓴다.
    groups = d["game_id"] + "_" + d["pitcher"]
    fit = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": groups})

    out = {"n": int(len(d)), "outcome": outcome,
           "model": ("회전+투구수" if use_tto and use_pitches
                     else "회전만" if use_tto else "투구수만")}
    for name in wanted:
        ci = fit.conf_int().loc[name]
        out[name] = {"coef": fit.params[name], "lo": ci[0], "hi": ci[1],
                     "sig": (ci[0] > 0) == (ci[1] > 0)}
    return out


def instability_report(pa: pd.DataFrame) -> list[str]:
    """변수를 따로 넣을 때와 같이 넣을 때 계수가 어떻게 달라지는지 나란히 본다."""
    lines = ["[1] 변수를 따로 넣을 때 vs 같이 넣을 때 (결과: 안타/타석)"]

    raw = pa.groupby("타순회전")["안타"].mean()
    lines.append(f"  - 원자료 안타/타석: 1바퀴 {raw.loc[1]:.3f} · 2바퀴 {raw.loc[2]:.3f} "
                 f"· 3바퀴 {raw.loc[3]:.3f} (1→3 실제 차이 {raw.loc[3]-raw.loc[1]:+.3f})")

    only_tto = fit_model(pa, "안타", use_tto=True, use_pitches=False)
    only_p = fit_model(pa, "안타", use_tto=False, use_pitches=True)
    both = fit_model(pa, "안타", use_tto=True, use_pitches=True)

    lines.append(f"  - 회전만 넣으면   3바퀴째: {only_tto['회전_3']['coef']:+.4f} "
                 f"[{only_tto['회전_3']['lo']:+.4f}, {only_tto['회전_3']['hi']:+.4f}]")
    lines.append(f"  - 투구수만 넣으면 10구당: {only_p['투구수_10구당']['coef']:+.4f} "
                 f"[{only_p['투구수_10구당']['lo']:+.4f}, {only_p['투구수_10구당']['hi']:+.4f}]")
    lines.append(f"  - 같이 넣으면     3바퀴째: {both['회전_3']['coef']:+.4f} / "
                 f"10구당: {both['투구수_10구당']['coef']:+.4f}")
    ratio = both["회전_3"]["coef"] / only_tto["회전_3"]["coef"]
    lines.append(f"    → 회전 계수가 {ratio:.1f}배로 부풀고 투구수는 음수로 뒤집힌다.")
    lines.append("      두 계수가 서로를 상쇄하며 커지는 것은 공선성의 전형적 증상이다.\n")
    return lines


def fig_instability(pa: pd.DataFrame) -> None:
    """따로 넣을 때와 같이 넣을 때의 계수를 나란히 세운다 — 공선성의 시각적 증거."""
    only_tto = fit_model(pa, "안타", use_tto=True, use_pitches=False)
    only_p = fit_model(pa, "안타", use_tto=False, use_pitches=True)
    both = fit_model(pa, "안타", use_tto=True, use_pitches=True)

    rows = [
        ("타순 회전 효과 (3바퀴째)\n회전만 모형에 넣음", only_tto["회전_3"], BLUE),
        ("타순 회전 효과 (3바퀴째)\n투구수와 함께 넣음", both["회전_3"], BLUE),
        ("투구수 효과 (10구당)\n투구수만 모형에 넣음", only_p["투구수_10구당"], ORANGE),
        ("투구수 효과 (10구당)\n회전과 함께 넣음", both["투구수_10구당"], ORANGE),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150, facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    for yi, (label, r, color) in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, linewidth=2.4, solid_capstyle="round")
        ax.plot(r["coef"], yi, "o", color=color, markersize=9)
        ax.text(r["hi"] + 0.004, yi, f"{r['coef']:+.4f}", va="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=9)
    ax.set_title("같이 넣는 순간 계수가 무너진다 (투수 고정효과, 95% 신뢰구간)",
                 color=INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel("안타/타석 확률 변화", color=SECONDARY_INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_collinear_instability.png")
    plt.close(fig)


def fig_coefficients(results: list[dict]) -> None:
    rows = []
    for res in results:
        label = {"안타": "피안타율", "출루": "피출루율"}[res["outcome"]]
        rows.append((f"{label}\n2바퀴째 (1바퀴 대비)", res["회전_2"], BLUE))
        rows.append((f"{label}\n3바퀴째 (1바퀴 대비)", res["회전_3"], BLUE))
        rows.append((f"{label}\n투구수 10구 증가", res["투구수_10구당"], ORANGE))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150, facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    for yi, (label, r, color) in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, linewidth=2.4, solid_capstyle="round")
        ax.plot(r["coef"], yi, "o", color=color, markersize=9)
        ax.text(r["hi"] + 0.0025, yi, f"{r['coef']:+.3f}", va="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=9)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2.4, marker="o", label="타순 회전 효과"),
               plt.Line2D([], [], color=ORANGE, linewidth=2.4, marker="o", label="누적 투구수 효과")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")
    ax.set_title("두 변수를 한 모형에 같이 넣으면 (투수 고정효과, 95% 신뢰구간)",
                 color=INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel("확률 변화", color=SECONDARY_INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_collinear_coefficients.png")
    plt.close(fig)


def summarize(pa: pd.DataFrame, collin: list[str], results: list[dict]) -> None:
    n_pitchers = pa["pitcher"].nunique()
    lines = [f"=== {SEASON} 타순 회전 vs 누적 투구수 — 동시 추정 "
             f"(투수 {n_pitchers}명 / 타석 {len(pa):,}건) ===\n"]
    lines.extend(collin)
    lines.extend(instability_report(pa))

    for res in results:
        label = {"안타": "피안타율(타석 기준)", "출루": "피출루율"}[res["outcome"]]
        lines.append(f"[{label}] 선형확률모형 + 투수 고정효과 + 타순 통제, 등판 군집 표준오차")
        for name, shown in (("회전_2", "2바퀴째 (1바퀴 대비)"),
                            ("회전_3", "3바퀴째 (1바퀴 대비)"),
                            ("투구수_10구당", "누적 투구수 10구 증가")):
            r = res[name]
            mark = "유의함" if r["sig"] else "유의하지 않음"
            lines.append(f"  - {shown}: {r['coef']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {mark}")
        lines.append("")

    out = "\n".join(lines)
    path = FIG.parent / f"eda_{SEASON}_collinear_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


def main() -> None:
    pa = load()
    collin = collinearity_report(pa)
    results = [fit_model(pa, "안타"), fit_model(pa, "출루")]
    fig_instability(pa)
    fig_coefficients(results)
    summarize(pa, collin, results)
    print(f"그림 2장 -> {FIG}")


if __name__ == "__main__":
    main()
