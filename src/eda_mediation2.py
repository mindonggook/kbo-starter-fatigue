"""매개를 다시 고른다 — 구속이 아니라 릴리스 흔들림, 피안타율이 아니라 피출루율.

발견 18은 구속을 매개로 놓고 기여분 ≈ 0%을 얻었다. 그런데 매개와 결과를 모두
잘못 고른 가능성이 있다.

  - 매개: 구속은 결과를 거의 예측하지 못했다(발견 18). 반면 릴리스 흔들림은
          볼 비율을 크게 움직인다(발견 22, 1cm당 +0.24%p).
  - 결과: 볼이 늘면 피안타율이 아니라 <피출루율>이 오른다. 발견 18은 피안타율만 봐서
          제구 경로를 구조적으로 놓쳤을 수 있다.

그래서 같은 분해를 (릴리스 흔들림 → 피출루율) 축에서 다시 한다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"naver_pa_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    pa = pd.concat(frames, ignore_index=True)
    pa = pa.dropna(subset=["종류", "release_x", "release_z", "최고구속"])
    pa = pa[pa["최고구속"].between(100, 170)].copy()

    first = pa[pa["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    pa = pa[[c in starters.get(g, set())
             for g, c in zip(pa["game_id"], pa["pitcher_code"])]].copy()
    pa["start"] = pa["game_id"].astype(str) + "_" + pa["pitcher_code"].astype(str)

    reached = pa[pa["투수_누적투구수"] >= 90]["start"].unique()
    pa = pa[pa["start"].isin(reached) & pa["이전_누적투구수"].between(0, 100)].copy()

    for c in ("release_x", "release_z"):
        pa[f"{c}_dev"] = pa[c] - pa.groupby("start")[c].transform("mean")
    pa["릴리스_흔들림"] = np.sqrt(pa["release_x_dev"] ** 2 + pa["release_z_dev"] ** 2) * 30.48
    pa["출루"] = pa["종류"].isin(["안타", "볼넷·사구"]).astype(float)
    pa["볼넷"] = (pa["종류"] == "볼넷·사구").astype(float)
    pa["투구수_10구당"] = pa["이전_누적투구수"] / 10.0
    return pa


def fit_ols(d: pd.DataFrame, y: str, cols: list[str]):
    X = pd.DataFrame(index=d.index)
    for c in cols:
        X[c] = d[c].astype(float)
    X["타석투구수"] = d["투구수"].astype(float)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = sm.add_constant(X)
    return sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    d = load(a.seasons)

    results = {}
    for y, label in (("출루", "피출루율"), ("볼넷", "볼넷률")):
        # 경로 a: 투구수 → 릴리스 흔들림
        Xa = pd.DataFrame({"투구수_10구당": d["투구수_10구당"].astype(float),
                           "타석투구수": d["투구수"].astype(float)}, index=d.index)
        fa = sm.OLS(d["릴리스_흔들림"].astype(float), sm.add_constant(Xa)).fit(
            cov_type="cluster", cov_kwds={"groups": d["start"]})
        a_coef = fa.params["투구수_10구당"]

        total_fit = fit_ols(d, y, ["투구수_10구당"])
        both_fit = fit_ols(d, y, ["투구수_10구당", "릴리스_흔들림"])
        total = total_fit.params["투구수_10구당"]
        t_ci = total_fit.conf_int().loc["투구수_10구당"]
        b_coef = both_fit.params["릴리스_흔들림"]
        b_ci = both_fit.conf_int().loc["릴리스_흔들림"]
        direct = both_fit.params["투구수_10구당"]
        mediated = a_coef * b_coef
        share = mediated / total * 100 if total else float("nan")
        results[label] = dict(a=a_coef, b=b_coef, b_ci=b_ci, total=total, t_ci=t_ci,
                              direct=direct, mediated=mediated, share=share)

    # 그림
    d["흔들림_구간"] = pd.qcut(d["릴리스_흔들림"], 5,
                            labels=["매우 안정", "안정", "보통", "흔들림", "매우 흔들림"])
    tab = d.groupby("흔들림_구간", observed=True).agg(
        피출루율=("출루", "mean"), 볼넷률=("볼넷", "mean"), n=("출루", "size"))
    tab["se"] = np.sqrt(tab["피출루율"] * (1 - tab["피출루율"]) / tab["n"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    x = np.arange(len(tab))
    ax.bar(x, tab["피출루율"], color=BLUE, width=0.58, label="피출루율")
    ax.plot(x, tab["볼넷률"], color=ORANGE, linewidth=2.2, marker="o", markersize=7, label="볼넷률")
    ax.errorbar(x, tab["피출루율"], yerr=tab["se"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, r in zip(x, tab.itertuples()):
        ax.text(xi, r.피출루율 + r.se * 1.96 + 0.008, f"{r.피출루율:.3f}", ha="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index.astype(str), fontsize=8.5)
    ax.set_ylim(0, tab["피출루율"].max() * 1.35)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")
    ax.set_title("릴리스가 흔들린 타석의 결과", color=INK, fontsize=12.5, loc="left", pad=10)

    ax = axes[1]
    r = results["피출루율"]
    parts = ["총효과\n(투구수 10구당)", "릴리스가\n설명하는 몫", "나머지\n(직접효과)"]
    vals = [r["total"] * 1000, r["mediated"] * 1000, r["direct"] * 1000]
    ax.bar(np.arange(3), vals, color=[MUTED, ORANGE, AQUA], width=0.55)
    ax.axhline(0, color=MUTED, linewidth=1.2)
    for xi, v in zip(range(3), vals):
        ax.text(xi, v + (0.1 if v >= 0 else -0.1), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(parts, fontsize=9)
    ax.set_title(f"릴리스 흔들림이 설명하는 몫: {r['share']:.1f}%",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("피출루율 변화 (1/1000)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("제구 경로로 다시 분해하다", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "mediation_release.png")
    plt.close(fig)

    lines = [f"=== 릴리스 흔들림을 매개로 한 재분해 ({len(a.seasons)}시즌) ===",
             f"선발 타석 {len(d):,}건 / 등판 {d['start'].nunique():,}개\n"]

    for label, r in results.items():
        lines.append(f"[{label}]")
        lines.append(f"  경로 a (투구수 10구 → 릴리스 흔들림): {r['a']:+.4f} cm")
        lines.append(f"  경로 b (흔들림 1cm → {label}): {r['b']:+.5f} "
                     f"[{r['b_ci'][0]:+.5f}, {r['b_ci'][1]:+.5f}]"
                     f" {'유의함' if (r['b_ci'][0]>0)==(r['b_ci'][1]>0) else '유의하지 않음'}")
        lines.append(f"  총효과 (투구수 10구 → {label}): {r['total']:+.5f} "
                     f"[{r['t_ci'][0]:+.5f}, {r['t_ci'][1]:+.5f}]"
                     f" {'유의함' if (r['t_ci'][0]>0)==(r['t_ci'][1]>0) else '유의하지 않음'}")
        lines.append(f"  매개분 = {r['mediated']:+.5f} · 직접효과 = {r['direct']:+.5f}")
        lines.append(f"  → 릴리스가 설명하는 몫: {r['share']:.1f}%\n")

    lines.append("[참고] 흔들림 5분위별")
    for idx, row in tab.iterrows():
        lines.append(f"  {idx}: 피출루율 {row['피출루율']:.4f} · 볼넷률 {row['볼넷률']:.4f} "
                     f"(n={int(row['n']):,})")

    out = "\n".join(lines)
    path = RESULTS / "eda_mediation2_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
