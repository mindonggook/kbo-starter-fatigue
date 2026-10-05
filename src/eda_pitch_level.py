"""한계 1 재도전 — 투구 단위 지표로 '익숙함'과 '피로'를 가른다.

발견 08에서 타순 회전과 누적 투구수를 분리하지 못한 것은 결과 지표(피안타율)가
두 기제에 모두 반응하기 때문이었다. 투구 단위에는 반응 방식이 다른 지표가 있다.

  볼 비율   — 투수의 제구. 타자가 세 번째 본다고 투수가 볼을 더 던지지는 않는다.
             즉 익숙함으로는 설명되지 않는, 피로에 가장 가까운 신호다.
  헛스윙률  — 투수의 구위. 익숙함도 깎지만 결과 지표보다는 덜하다.
  파울 비율 — 타자가 공을 맞혀내기 시작했다는 신호. 익숙함 쪽에 가깝다.

세 지표를 타순 회전별·누적 투구수별로 각각 재고, 타순을 표준화해 비교한다.
회전에서만 움직이면 익숙함, 투구수에서만 움직이면 피로다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

METRICS = [
    ("볼", "볼 비율", "제구 — 익숙함과 무관", BLUE),
    ("헛스윙", "헛스윙률", "구위", AQUA),
    ("파울", "파울 비율", "타자가 맞혀내기 시작", ORANGE),
]


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=SECONDARY_INK, fontsize=9.5)
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"kbo_pitches_{s}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        d["season"] = s
        frames.append(d)
    pa = pd.concat(frames, ignore_index=True)
    pa = pa[pa["is_starter"] & pa["타순회전"].between(1, 3)
            & pa["batting_order"].between(1, 9)].copy()
    pa["헛스윙"] = pa["결과"].isin(["헛스윙", "번트헛스윙"])
    pa["볼"] = pa["결과"] == "볼"
    pa["파울"] = pa["결과"].isin(["파울", "번트파울"])
    return pa


def standardize(cells: pd.DataFrame, level_col: str, weights: pd.Series,
                num: str) -> pd.DataFrame:
    """타순 구성을 공통 가중치로 맞춘 비율 — 상위타순 비중 변화를 제거한다."""
    out = []
    for level, grp in cells.groupby(level_col, observed=True):
        g = grp.set_index("batting_order")
        common = weights.index.intersection(g.index)
        w = weights.loc[common] / weights.loc[common].sum()
        p = (g.loc[common, num] / g.loc[common, "n"]).astype(float)
        var = (w ** 2 * p * (1 - p) / g.loc[common, "n"]).sum()
        out.append({level_col: level, "rate": float((w * p).sum()),
                    "se": float(np.sqrt(var)), "n": int(g.loc[common, "n"].sum())})
    return pd.DataFrame(out).set_index(level_col)


def survivors(pa: pd.DataFrame, axis: str) -> pd.DataFrame:
    """끝까지 간 등판만 남긴다 — 이게 없으면 생존 편향이 피로 신호를 뒤집는다.

    제구가 흔들린 투수는 애초에 76~100구 구간에 도달하지 못한다. 그래서 통제 없이
    구간별로 비교하면 '투구수가 쌓일수록 좋아진다'는 거꾸로 된 그림이 나온다.
    """
    if axis == "타순회전":
        reached = pa[pa["타순회전"] >= 3]
    else:
        reached = pa[pa["투수_누적투구수"] >= 90]
    keys = reached[["game_id", "pitcher"]].drop_duplicates()
    return pa.merge(keys, on=["game_id", "pitcher"], how="inner")


def by_axis(pa: pd.DataFrame, axis: str, num: str, control: bool = True) -> pd.DataFrame:
    d = survivors(pa, axis) if control else pa
    cells = (d.groupby([axis, "batting_order"], observed=True)
               .agg(n=(num, "size"), **{num: (num, "sum")}).reset_index())
    last = cells[axis].max() if axis == "타순회전" else cells[axis].cat.categories[-1]
    w = cells[cells[axis] == last].set_index("batting_order")["n"]
    return standardize(cells, axis, w, num)


def diff_ci(t: pd.DataFrame, a, b) -> tuple[float, float, float, bool]:
    d = t.loc[b, "rate"] - t.loc[a, "rate"]
    se = np.sqrt(t.loc[a, "se"] ** 2 + t.loc[b, "se"] ** 2)
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return d, lo, hi, (lo > 0) == (hi > 0)


def fig_two_axes(pa: pd.DataFrame, tables: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), dpi=150, facecolor=SURFACE)
    for col, (num, label, note, color) in enumerate(METRICS):
        for row, (axis, xlab) in enumerate([("타순회전", "타순 회전"),
                                            ("투구수_구간", "누적 투구수")]):
            ax = axes[row][col]
            t = tables[(num, axis)]
            x = np.arange(len(t))
            ax.bar(x, t["rate"] * 100, color=color, width=0.55)
            ax.errorbar(x, t["rate"] * 100, yerr=t["se"] * 196, fmt="none",
                        ecolor=SECONDARY_INK, elinewidth=1.3, capsize=4)
            for xi, r in zip(x, t.itertuples()):
                ax.text(xi, (r.rate + r.se * 1.96) * 100 + 0.25, f"{r.rate*100:.1f}",
                        ha="center", color=SECONDARY_INK, fontsize=8.5, fontweight="bold")
            ax.set_xticks(x)
            labels = ([f"{i}바퀴" for i in t.index] if axis == "타순회전"
                      else [str(c) for c in t.index])
            ax.set_xticklabels(labels, fontsize=8.5)
            lo = min(t["rate"]) * 100
            ax.set_ylim(max(0, lo - 4), (max(t["rate"]) + max(t["se"]) * 1.96) * 100 + 1.6)
            title = f"{label} — {xlab}별" + (f"\n({note})" if row == 0 else "")
            _style_ax(ax, title, "", "%" if col == 0 else "")
    fig.suptitle("투구 단위 지표: 익숙함(회전)과 피로(투구수) 중 어느 축이 움직이나",
                 color=INK, fontsize=14, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "pitch_level_two_axes.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int,
                   default=[2021, 2022, 2023, 2024, 2025, 2026])
    a = p.parse_args()

    pa = load(a.seasons)
    pa["투구수_구간"] = pd.cut(pa["투수_누적투구수"], [0, 25, 50, 75, 100],
                           labels=["1~25구", "26~50구", "51~75구", "76~100구"])
    pa_pc = pa.dropna(subset=["투구수_구간"])

    tables, raw = {}, {}
    for num, *_ in METRICS:
        tables[(num, "타순회전")] = by_axis(pa, "타순회전", num, control=True)
        tables[(num, "투구수_구간")] = by_axis(pa_pc, "투구수_구간", num, control=True)
        raw[(num, "타순회전")] = by_axis(pa, "타순회전", num, control=False)
        raw[(num, "투구수_구간")] = by_axis(pa_pc, "투구수_구간", num, control=False)

    fig_two_axes(pa, tables)

    lines = [f"=== 투구 단위 분석 ({len(a.seasons)}시즌 · 선발 투구 {len(pa):,}개) ===\n",
             "타순 구성을 표준화한 비율이다. 회전에서만 움직이면 익숙함,",
             "투구수에서만 움직이면 피로로 읽는다.\n"]

    for num, label, note, _ in METRICS:
        lines.append(f"[{label}]  ({note})")
        t1 = tables[(num, "타순회전")]
        for idx, r in t1.iterrows():
            lines.append(f"  {idx}바퀴째: {r['rate']*100:.2f}% (±{r['se']*196:.2f}, n={int(r['n']):,})")
        d, lo, hi, sig = diff_ci(t1, 1, 3)
        r1 = raw[(num, "타순회전")]
        rd, *_ = diff_ci(r1, 1, 3)
        lines.append(f"  → 1→3바퀴: {d*100:+.2f}%p [{lo*100:+.2f}, {hi*100:+.2f}] "
                     f"{'유의함' if sig else '유의하지 않음'}  (통제 전 {rd*100:+.2f}%p)")

        t2 = tables[(num, "투구수_구간")]
        for idx, r in t2.iterrows():
            lines.append(f"  {idx}: {r['rate']*100:.2f}% (±{r['se']*196:.2f}, n={int(r['n']):,})")
        first, last = t2.index[0], t2.index[-1]
        d, lo, hi, sig = diff_ci(t2, first, last)
        r2 = raw[(num, "투구수_구간")]
        rd, *_ = diff_ci(r2, first, last)
        lines.append(f"  → {first}→{last}: {d*100:+.2f}%p [{lo*100:+.2f}, {hi*100:+.2f}] "
                     f"{'유의함' if sig else '유의하지 않음'}  (통제 전 {rd*100:+.2f}%p)")
        lines.append("")

    out = "\n".join(lines)
    path = RESULTS / "eda_pitch_level_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
