"""구원투수의 우위는 구속에서 오는가, 낯섦에서 오는가 — 발견 15·16 재검정.

발견 16은 필승조로 바꾸면 피안타율이 31포인트 좋아진다고 했고,
발견 15·17은 그 이점을 '낯섦'으로 설명했다. 그런데 통념은 다르다 —
"불펜은 한 이닝만 던지니 전력투구하고, 그래서 구속이 높다".

이제 둘을 가를 수 있다. 발견 18에서 구속은 결과를 거의 예측하지 못했으므로
(매개 몫 ≈ 0%), 예측이 따라 나온다. <구원투수가 훨씬 빠르더라도 그 구속 우위가
성과 우위를 설명하지 못해야 한다>. 설명한다면 발견 18이 흔들리고,
설명하지 못한다면 '낯섦' 해석이 한 번 더 지지된다.
"""
import argparse
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
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def bullpen_tiers(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{s}.parquet")
        rel = box[~box["is_starter"]].copy()
        rel["결과"] = rel["결과"].fillna("")
        rel["성공"] = rel["결과"].str.contains("세|홀드", regex=True).astype(int)
        g = rel.groupby("선수명")["성공"].sum().reset_index()
        g["season"] = s
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    t["등급"] = np.select([t["성공"] >= 15, t["성공"] >= 5],
                        ["필승조", "중간"], default="추격조")
    return t.rename(columns={"선수명": "pitcher"})[["season", "pitcher", "등급"]]


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"naver_pa_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    pa = pd.concat(frames, ignore_index=True)
    pa = pa.dropna(subset=["최고구속", "pitcher", "타수"])
    pa = pa[pa["최고구속"].between(100, 170)].copy()
    pa["타수"] = pa["타수"].astype(bool)
    pa["안타"] = pa["안타"].astype(bool)

    first_inn = pa[pa["inning"] == 1]
    starters = first_inn.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    pa["is_starter"] = [c in starters.get(g, set())
                        for g, c in zip(pa["game_id"], pa["pitcher_code"])]
    pa["start"] = pa["game_id"].astype(str) + "_" + pa["pitcher_code"].astype(str)

    tiers = bullpen_tiers(seasons)
    pa = pa.merge(tiers, on=["season", "pitcher"], how="left")

    # 발견 15와 같은 틀 — 5~8회, 선발은 3바퀴째, 구원은 1바퀴째
    d = pa[pa["inning"].between(5, 8) & pa["타수"]].copy()
    role = np.where(d["is_starter"] & (d["타순회전"] >= 3), "선발 3바퀴째",
           np.where(d["is_starter"] & (d["타순회전"] == 1), "선발 1바퀴째",
           np.where(~d["is_starter"] & (d["타순회전"] == 1),
                    "구원 " + d["등급"].fillna("미분류"), None)))
    d["역할"] = role
    return d[d["역할"].notna() & ~d["역할"].str.contains("미분류")].copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    d = load(a.seasons)
    d["안타_i"] = d["안타"].astype(float)

    tab = d.groupby("역할").agg(
        구속=("최고구속", "mean"), 피안타율=("안타_i", "mean"), n=("안타_i", "size"))
    tab["구속_se"] = d.groupby("역할")["최고구속"].sem()
    tab["타율_se"] = np.sqrt(tab["피안타율"] * (1 - tab["피안타율"]) / tab["n"])

    # 구속을 넣으면 등급 계수가 줄어드는가 — 줄면 구속이 설명하는 것이다.
    base = pd.get_dummies(d["역할"], drop_first=False).astype(float)
    ref = "선발 3바퀴째"
    cols = [c for c in base.columns if c != ref]
    X1 = sm.add_constant(base[cols])
    fit1 = sm.OLS(d["안타_i"], X1).fit(cov_type="cluster", cov_kwds={"groups": d["start"]})

    X2 = X1.copy()
    X2["구속"] = d["최고구속"].astype(float)
    X2["타석투구수"] = d["투구수"].astype(float)
    fit2 = sm.OLS(d["안타_i"], X2).fit(cov_type="cluster", cov_kwds={"groups": d["start"]})

    order = ["선발 1바퀴째", "구원 추격조", "구원 중간", "구원 필승조"]
    order = [o for o in order if o in cols]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    show = [ref] + order
    show = [s_ for s_ in show if s_ in tab.index]
    x = np.arange(len(show))
    vals = [tab.loc[s_, "구속"] for s_ in show]
    errs = [tab.loc[s_, "구속_se"] * 1.96 for s_ in show]
    colors = [ORANGE] + [BLUE, MUTED, AQUA, VIOLET][:len(show) - 1]
    ax.bar(x, vals, color=colors, width=0.58)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s_.replace(" ", "\n") for s_ in show], fontsize=8.5)
    ax.set_ylim(min(vals) - 4, max(vals) + 2.5)
    ax.set_title("역할별 평균 최고구속", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("km/h", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    y = np.arange(len(order))[::-1]
    for yi, o in zip(y, order):
        b1, b2 = fit1.params[o], fit2.params[o]
        ci1 = fit1.conf_int().loc[o]
        ax.plot([ci1[0] * 1000, ci1[1] * 1000], [yi + 0.12, yi + 0.12],
                color=MUTED, linewidth=2.2, solid_capstyle="round")
        ax.plot(b1 * 1000, yi + 0.12, "o", color=MUTED, markersize=8)
        ci2 = fit2.conf_int().loc[o]
        ax.plot([ci2[0] * 1000, ci2[1] * 1000], [yi - 0.12, yi - 0.12],
                color=VIOLET, linewidth=2.2, solid_capstyle="round")
        ax.plot(b2 * 1000, yi - 0.12, "D", color=VIOLET, markersize=8)
        ax.text(max(ci1[1], ci2[1]) * 1000 + 1.2, yi,
                f"{b1*1000:+.1f} → {b2*1000:+.1f}", va="center",
                color=SECONDARY_INK, fontsize=9, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    handles = [plt.Line2D([], [], color=MUTED, marker="o", linestyle="", label="구속 미통제"),
               plt.Line2D([], [], color=VIOLET, marker="D", linestyle="", label="구속 통제 후")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")
    ax.set_title("구속을 통제하면 역할 효과가 사라지나", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_xlabel("선발 3바퀴째 대비 피안타율 차 (1/1000)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("구원투수의 우위는 구속에서 오는가", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "relief_velocity.png")
    plt.close(fig)

    lines = [f"=== 구원투수의 우위는 구속에서 오는가 ({len(a.seasons)}시즌 · 5~8회) ===",
             f"타수 {len(d):,}건 / 등판 {d['start'].nunique():,}개\n"]

    lines.append("[1] 역할별 평균 최고구속과 피안타율")
    for idx in show:
        r = tab.loc[idx]
        lines.append(f"  {idx}: 구속 {r['구속']:.2f} km/h (±{r['구속_se']*1.96:.2f}) · "
                     f"피안타율 {r['피안타율']:.4f} (n={int(r['n']):,})")
    if "구원 필승조" in tab.index:
        gap = tab.loc["구원 필승조", "구속"] - tab.loc[ref, "구속"]
        lines.append(f"  → 필승조는 선발 3바퀴째보다 {gap:+.2f} km/h 빠르다")
    lines.append("")

    lines.append("[2] 구속을 통제하면 역할 효과가 줄어드는가 (선발 3바퀴째 기준, 1/1000)")
    for o in order:
        b1, b2 = fit1.params[o] * 1000, fit2.params[o] * 1000
        ci2 = fit2.conf_int().loc[o] * 1000
        shrink = (1 - b2 / b1) * 100 if b1 else float("nan")
        lines.append(f"  {o}: {b1:+.2f} → {b2:+.2f} "
                     f"[{ci2[0]:+.2f}, {ci2[1]:+.2f}] (변화 {shrink:+.1f}%)")
    vb = fit2.params["구속"]
    vci = fit2.conf_int().loc["구속"]
    lines.append(f"\n  모형 안의 구속 계수: {vb:+.5f} [{vci[0]:+.5f}, {vci[1]:+.5f}] "
                 f"{'유의함' if (vci[0]>0)==(vci[1]>0) else '유의하지 않음'}")

    out = "\n".join(lines)
    path = DATA.parent / "eda_relief_velocity_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
