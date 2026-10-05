"""발견 13의 모순을 닫는다 — 폼이 무너지는데 제구가 좋아지는가.

발견 13: 투구수가 쌓이면 볼 비율이 오히려 줄었다(-0.90%p).
발견 17: 같은 구간에서 릴리스 포인트는 두 배로 흔들린다(+6.2cm).

둘이 동시에 참일 수는 있지만 설명이 필요하다. 가능한 그림은 셋이다.
  (1) 릴리스 흔들림이 제구와 무관하다 — 투수가 다른 식으로 보정한다
  (2) 흔들리면 제구가 나빠지는데, 볼 비율 감소는 생존 편향의 잔재다
  (3) 흔들림이 '의도된 변화'다 — 구종을 섞으면 릴리스가 달라 보인다

같은 투구에 릴리스 좌표와 볼/스트라이크가 함께 있으므로 (1)과 (2)를 가를 수 있다.
릴리스가 평소보다 많이 벗어난 투구가 실제로 볼이 되는지만 보면 된다.
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
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"naver_pitch_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["결과", "release_x", "release_z", "구속"])
    d = d[d["구속"].between(100, 170)].copy()

    first = d[d["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    d["is_starter"] = [c in starters.get(g, set())
                       for g, c in zip(d["game_id"], d["pitcher_code"])]
    d["start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)

    d = d[d["is_starter"]].copy()
    # 90구 이상 간 등판만 — 발견 13·17과 같은 생존 편향 통제
    reached = d[d["투수_누적투구수"] >= 90]["start"].unique()
    d = d[d["start"].isin(reached) & d["투수_누적투구수"].between(1, 100)].copy()

    for c in ("release_x", "release_z"):
        d[f"{c}_dev"] = d[c] - d.groupby("start")[c].transform("mean")
    d["릴리스_흔들림"] = np.sqrt(d["release_x_dev"] ** 2 + d["release_z_dev"] ** 2) * 30.48
    d["볼"] = (d["결과"] == "볼").astype(float)
    d["헛스윙"] = d["결과"].isin(["헛스윙", "번트헛스윙"]).astype(float)
    d["구속_편차"] = d["구속"] - d.groupby("start")["구속"].transform("mean")
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    d = load(a.seasons)
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], [0, 25, 50, 75, 100],
                           labels=["1~25구", "26~50구", "51~75구", "76~100구"])

    # 흔들림 5분위별 볼 비율 — 관계가 있는지부터 눈으로 본다.
    d["흔들림_구간"] = pd.qcut(d["릴리스_흔들림"], 5,
                            labels=["매우 안정", "안정", "보통", "흔들림", "매우 흔들림"])
    tab = d.groupby("흔들림_구간", observed=True).agg(
        볼비율=("볼", "mean"), 흔들림=("릴리스_흔들림", "median"), n=("볼", "size"))
    tab["se"] = np.sqrt(tab["볼비율"] * (1 - tab["볼비율"]) / tab["n"])

    # 회귀 — 투구수를 통제하고 흔들림만의 효과를 본다.
    X = pd.DataFrame(index=d.index)
    X["흔들림_cm"] = d["릴리스_흔들림"].astype(float)
    X["투구수_10구당"] = d["투수_누적투구수"].astype(float) / 10.0
    X["구속_편차"] = d["구속_편차"].astype(float)
    X = sm.add_constant(X)
    fit = sm.OLS(d["볼"], X).fit(cov_type="cluster", cov_kwds={"groups": d["start"]})

    def grab(name):
        ci = fit.conf_int().loc[name]
        return fit.params[name], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)

    wob = grab("흔들림_cm")
    pit = grab("투구수_10구당")

    # 투구수 구간 × 흔들림 — 둘이 같이 움직이는지
    grid = (d.groupby(["투구수_구간", "흔들림_구간"], observed=True)["볼"]
              .agg(["mean", "size"]).rename(columns={"mean": "볼비율", "size": "n"}))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    x = np.arange(len(tab))
    ax.bar(x, tab["볼비율"] * 100, color=BLUE, width=0.58)
    ax.errorbar(x, tab["볼비율"] * 100, yerr=tab["se"] * 196, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, r in zip(x, tab.itertuples()):
        ax.text(xi, r.볼비율 * 100 + r.se * 196 + 0.25, f"{r.볼비율*100:.1f}", ha="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
        ax.text(xi, 1.0, f"{r.흔들림:.0f}cm", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index.astype(str), fontsize=8.5)
    ax.set_ylim(0, tab["볼비율"].max() * 130)
    ax.set_title("릴리스가 흔들린 투구는 볼이 되는가", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("볼 비율 (%)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    bands = ["1~25구", "26~50구", "51~75구", "76~100구"]
    for lab, color in (("매우 안정", BLUE), ("매우 흔들림", ORANGE)):
        vals, pos = [], []
        for i, b in enumerate(bands):
            if (b, lab) in grid.index:
                vals.append(grid.loc[(b, lab), "볼비율"] * 100)
                pos.append(i)
        ax.plot(pos, vals, color=color, linewidth=2.2, marker="o", markersize=7, label=lab)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    ax.set_title("투구수가 쌓여도 그 관계가 유지되나", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("볼 비율 (%)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("폼이 흔들리면 제구가 나빠지는가", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "release_command.png")
    plt.close(fig)

    lines = [f"=== 릴리스 흔들림과 제구 ({len(a.seasons)}시즌) ===",
             f"선발 투구 {len(d):,}개 / 등판 {d['start'].nunique():,}개 "
             f"(90구 이상 간 등판, 등판 단위 군집 보정)\n"]

    lines.append("[1] 릴리스 흔들림 5분위별 볼 비율")
    for idx, r in tab.iterrows():
        lines.append(f"  {idx} (중앙 {r['흔들림']:.1f}cm): {r['볼비율']*100:.2f}% "
                     f"(±{r['se']*196:.2f}, n={int(r['n']):,})")
    first, last = tab.index[0], tab.index[-1]
    diff = tab.loc[last, "볼비율"] - tab.loc[first, "볼비율"]
    se = np.sqrt(tab.loc[first, "se"] ** 2 + tab.loc[last, "se"] ** 2)
    lines.append(f"  → 가장 안정 대비 가장 흔들림: {diff*100:+.2f}%p "
                 f"[{(diff-1.96*se)*100:+.2f}, {(diff+1.96*se)*100:+.2f}]\n")

    lines.append("[2] 회귀 (투구수·구속 통제, 등판 군집 보정)")
    lines.append(f"  릴리스 흔들림 1cm 증가: {wob[0]*100:+.4f}%p "
                 f"[{wob[1]*100:+.4f}, {wob[2]*100:+.4f}] "
                 f"{'유의함' if wob[3] else '유의하지 않음'}")
    lines.append(f"  투구수 10구 증가      : {pit[0]*100:+.4f}%p "
                 f"[{pit[1]*100:+.4f}, {pit[2]*100:+.4f}] "
                 f"{'유의함' if pit[3] else '유의하지 않음'}\n")

    lines.append("[3] 투구수 구간 × 흔들림 구간별 볼 비율")
    for (band, wob_label), r in grid.iterrows():
        if wob_label in ("매우 안정", "매우 흔들림"):
            lines.append(f"  {band} / {wob_label}: {r['볼비율']*100:.2f}% (n={int(r['n']):,})")

    out = "\n".join(lines)
    path = RESULTS / "eda_release_command_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
