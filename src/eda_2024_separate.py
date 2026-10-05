"""한계 1 — 타순 회전과 누적 투구수를 분리한다.

한 등판 안에서 두 변수는 거의 같이 올라가지만 완전히 겹치지는 않는다. 효율적인
투수는 60구에 3바퀴째로 들어가고, 볼카운트가 깊은 투수는 같은 60구에 아직
2바퀴째다. 이 '겹치는 구간'을 쓰면 한쪽을 묶어놓고 다른 쪽만 움직여볼 수 있다.

  A. 투구수를 묶고 회전만 바꾼다  -> 회전 고유의 효과
  B. 회전을 묶고 투구수만 바꾼다  -> 투구수 고유의 효과

B에는 투수 고정이 반드시 들어가야 한다. 같은 회전인데 투구수가 많다는 건 곧
'삼진이 많은 투수'라는 뜻이라(삼진율 20.7% vs 14.5%), 투수를 가로질러 비교하면
피로가 아니라 투수 유형을 재게 된다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)


def load() -> pd.DataFrame:
    pa = pd.read_parquet(DATA / f"kbo_plate_appearances_{SEASON}.parquet")
    check = pd.read_parquet(DATA / f"kbo_livetext_check_{SEASON}.parquet")
    good = set(check.loc[check["starter_ok"], "game_id"])
    pa = pa[pa["game_id"].isin(good) & pa["is_starter"]].copy()
    pa["타수"] = pa["타수"].astype(bool)
    pa["안타"] = pa["안타"].astype(bool)
    # 피출루율은 삼진 비중에 덜 휘둘린다 — 피안타율 결과의 교차 확인용.
    pa["출루"] = pa["안타"] | (pa["종류"] == "볼넷·사구")
    return pa


def stratified_diff(df: pd.DataFrame, group_col: str, strata: list[str],
                    num: str, den: str) -> dict:
    """층별로 비율 차이를 구해 가중 평균한다 (Mantel-Haenszel 방식 위험차).

    층(타순·투수) 안에서만 두 그룹을 비교하므로, 층을 이루는 변수의 영향이 상쇄된다.
    """
    levels = sorted(df[group_col].dropna().unique())
    if len(levels) != 2:
        raise ValueError(f"두 그룹이어야 합니다: {levels}")
    a, b = levels

    num_sum = 0.0
    den_sum = 0.0
    var_sum = 0.0
    used = 0
    for _, g in df.groupby(strata, observed=True):
        ga, gb = g[g[group_col] == a], g[g[group_col] == b]
        n1, n2 = ga[den].sum(), gb[den].sum()
        if n1 < 1 or n2 < 1:
            continue
        p1, p2 = ga[num].sum() / n1, gb[num].sum() / n2
        w = (n1 * n2) / (n1 + n2)
        num_sum += w * (p2 - p1)
        den_sum += w
        var_sum += w ** 2 * (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        used += 1

    diff = num_sum / den_sum
    se = np.sqrt(var_sum) / den_sum
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return {"기준": a, "비교": b, "차이": diff, "lo": lo, "hi": hi,
            "유의": (lo > 0) == (hi > 0), "층수": used,
            f"{a}_n": int(df.loc[df[group_col] == a, den].sum()),
            f"{b}_n": int(df.loc[df[group_col] == b, den].sum())}


def test_a_fix_pitches(pa: pd.DataFrame) -> list[dict]:
    """투구수를 묶고 타순 회전만 바꾼다. 층 = 타순."""
    results = []
    for lo_p, hi_p, t1, t2 in [(30, 45, 1, 2), (60, 75, 2, 3)]:
        band = pa[(pa["투수_누적투구수"] > lo_p) & (pa["투수_누적투구수"] <= hi_p)
                  & pa["타순회전"].isin([t1, t2])].copy()
        band["그룹"] = band["타순회전"]
        for metric_num, metric_den, name in [("안타", "타수", "피안타율"), ("출루", "타석", "피출루율")]:
            d = band.copy()
            d["타석"] = 1
            r = stratified_diff(d, "그룹", ["batting_order"], metric_num, metric_den)
            r.update({"구간": f"{lo_p+1}~{hi_p}구", "지표": name,
                      "비교내용": f"{t1}바퀴 → {t2}바퀴"})
            results.append(r)
    return results


def test_b_fix_tto(pa: pd.DataFrame) -> list[dict]:
    """타순 회전을 묶고 투구수만 바꾼다. 층 = 타순 × 투수(투수 유형 제거)."""
    results = []
    specs = [(2, (30, 45), (60, 75)), (3, (60, 75), (90, 999))]
    for tto, low_band, high_band in specs:
        d = pa[pa["타순회전"] == tto].copy()
        in_low = d["투수_누적투구수"].between(low_band[0] + 1, low_band[1])
        in_high = d["투수_누적투구수"].between(high_band[0] + 1, high_band[1])
        d = d[in_low | in_high].copy()
        d["그룹"] = np.where(in_low[in_low | in_high], "적음", "많음")
        d["타석"] = 1
        for metric_num, metric_den, name in [("안타", "타수", "피안타율"), ("출루", "타석", "피출루율")]:
            r = stratified_diff(d, "그룹", ["batting_order", "pitcher"], metric_num, metric_den)
            r.update({"구간": f"{tto}바퀴째", "지표": name,
                      "비교내용": f"{low_band[0]+1}~{low_band[1]}구 → {high_band[0]+1}구 이상"
                      if high_band[1] > 900 else
                      f"{low_band[0]+1}~{low_band[1]}구 → {high_band[0]+1}~{high_band[1]}구"})
            results.append(r)
    return results


def fig_grid(pa: pd.DataFrame) -> pd.DataFrame:
    """회전 × 투구수 격자로 피안타율을 놓고 어느 축을 따라 올라가는지 본다."""
    d = pa[pa["타순회전"].between(1, 3)].copy()
    bins = [0, 30, 45, 60, 75, 90, 999]
    labels = ["~30구", "31~45구", "46~60구", "61~75구", "76~90구", "91구+"]
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], bins=bins, labels=labels)

    g = d.groupby(["타순회전", "투구수_구간"], observed=True).agg(
        안타=("안타", "sum"), 타수=("타수", "sum"))
    g["피안타율"] = g["안타"] / g["타수"]
    grid = g["피안타율"].unstack()
    counts = g["타수"].unstack()
    # 표본이 너무 적은 칸은 신뢰할 수 없어 비운다.
    grid = grid.where(counts >= 200)

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150, facecolor=SURFACE)
    im = ax.imshow(grid.values, cmap="Blues", aspect="auto", origin="lower",
                   vmin=np.nanmin(grid.values), vmax=np.nanmax(grid.values))
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns.astype(str))
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{i}바퀴째" for i in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                n = counts.values[i, j]
                hi = v > np.nanmean(grid.values)
                ax.text(j, i, f"{v:.3f}\n{int(n):,}타수", ha="center", va="center",
                        color="white" if hi else INK, fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.set_label("피안타율", color=SECONDARY_INK, fontsize=9)
    _style_ax(ax, f"타순 회전 × 누적 투구수 격자 ({SEASON})", "그 타석 시점의 누적 투구수", "타순 회전")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_separate_grid.png")
    plt.close(fig)
    return grid


def fig_summary(res_a: list[dict], res_b: list[dict]) -> None:
    """두 검정의 효과 크기와 신뢰구간을 한 그림에 세워 비교한다."""
    rows = [r for r in res_a + res_b if r["지표"] == "피안타율"]
    labels = [f"{r['구간']}\n{r['비교내용']}" for r in rows]
    kinds = ["회전 효과"] * len(res_a[::2]) + ["투구수 효과"] * len(res_b[::2])

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150, facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    for yi, r, kind in zip(y, rows, kinds):
        color = BLUE if kind == "회전 효과" else ORANGE
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, linewidth=2.4,
                solid_capstyle="round")
        ax.plot(r["차이"], yi, "o", color=color, markersize=9)
        ax.text(r["hi"] + 0.004, yi, f"{r['차이']:+.3f}", va="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-0.09, 0.09)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2.4, marker="o", label="투구수 고정 → 회전 변화"),
               plt.Line2D([], [], color=ORANGE, linewidth=2.4, marker="o", label="회전 고정 → 투구수 변화")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=SECONDARY_INK,
              loc="lower right")
    _style_ax(ax, f"어느 축이 피안타율을 올리는가 ({SEASON})",
              "피안타율 차이 (95% 신뢰구간)", "")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_separate_effects.png")
    plt.close(fig)


def summarize(res_a, res_b, grid) -> None:
    lines = [f"=== {SEASON} 타순 회전 vs 누적 투구수 분리 ===\n"]

    lines.append("[A] 투구수를 고정하고 타순 회전만 바꿨을 때 (층: 타순)")
    for r in res_a:
        mark = "유의함" if r["유의"] else "유의하지 않음"
        lines.append(f"  - {r['구간']} / {r['비교내용']} / {r['지표']}: "
                     f"{r['차이']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {mark}")
    lines.append("")

    lines.append("[B] 타순 회전을 고정하고 투구수만 바꿨을 때 (층: 타순 × 투수)")
    for r in res_b:
        mark = "유의함" if r["유의"] else "유의하지 않음"
        lines.append(f"  - {r['구간']} / {r['비교내용']} / {r['지표']}: "
                     f"{r['차이']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {mark}")
    lines.append("")

    lines.append("[C] 회전 × 투구수 격자 피안타율 (타수 200 미만 칸은 제외)")
    lines.append(grid.round(3).to_string())

    out = "\n".join(lines)
    path = FIG.parent / f"eda_{SEASON}_separate_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


def main() -> None:
    pa = load()
    res_a = test_a_fix_pitches(pa)
    res_b = test_b_fix_tto(pa)
    grid = fig_grid(pa)
    fig_summary(res_a, res_b)
    summarize(res_a, res_b, grid)
    print(f"그림 2장 -> {FIG}")


if __name__ == "__main__":
    main()
