"""마지막 질문 — 그래서 언제 내려야 하나.

지금까지는 전부 서술이었다. 교체 결정은 결국 한 가지 비교로 환원된다.

    3바퀴째를 맞는 선발  vs  타순을 처음 보는 구원투수

선발이 3바퀴째에 +0.025만큼 불리해진다는 것은 확인됐다(발견 06).
구원투수는 거의 항상 1바퀴째로 들어오므로 그 페널티를 지지 않는다.
그 차이가 교체로 얻는 것이고, 여기서 손익분기점이 나온다.

주의할 함정이 둘 있다.
  1. 구원투수는 아무 때나 나오지 않는다 — 접전에는 필승조, 대패에는 추격조.
     그래서 이닝과 점수 상황을 맞춰놓고 비교해야 한다.
  2. 불펜은 무한하지 않다. 일찍 내리면 그만큼 뒤가 비는데, 이 비용은
     이 비교에 들어 있지 않다. 결론을 읽을 때 반드시 함께 봐야 한다.
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
        pa_path = DATA / f"kbo_pa_state_{s}.parquet"
        chk_path = DATA / f"kbo_state_check_{s}.parquet"
        if not pa_path.exists():
            continue
        pa = pd.read_parquet(pa_path)
        chk = pd.read_parquet(chk_path)
        ok = set(chk.loc[chk["score_ok"] & chk["starter_ok"], "game_id"])
        d = pa[pa["game_id"].isin(ok)].copy()
        d["season"] = s
        frames.append(d)
    pa = pd.concat(frames, ignore_index=True)
    pa["안타"] = pa["안타"].astype(int)
    pa["타수"] = pa["타수"].astype(bool)
    pa["출루"] = (pa["안타"].astype(bool) | (pa["종류"] == "볼넷·사구")).astype(int)
    return pa


def standardize(cells: pd.DataFrame, level: str, weights: pd.Series,
                num: str, den: str) -> pd.DataFrame:
    """타순 구성을 맞춘 비율 — 선발과 구원이 상대하는 타순대가 다르므로 꼭 필요하다."""
    out = []
    for lv, grp in cells.groupby(level, observed=True):
        g = grp.set_index("batting_order")
        common = weights.index.intersection(g.index)
        w = weights.loc[common] / weights.loc[common].sum()
        p = (g.loc[common, num] / g.loc[common, den]).astype(float)
        var = (w ** 2 * p * (1 - p) / g.loc[common, den]).sum()
        out.append({level: lv, "rate": float((w * p).sum()),
                    "se": float(np.sqrt(var)), "n": int(g.loc[common, den].sum())})
    return pd.DataFrame(out).set_index(level)


def compare_by_inning(pa: pd.DataFrame, num: str, den: str) -> pd.DataFrame:
    """같은 이닝 안에서 '3바퀴째 선발'과 '1바퀴째 구원'을 맞붙인다."""
    d = pa[pa["inning"].between(5, 8)].copy()
    starter3 = d[d["is_starter"] & (d["타순회전"] >= 3)].assign(역할="선발 3바퀴째")
    relief1 = d[~d["is_starter"] & (d["타순회전"] == 1)].assign(역할="구원 1바퀴째")
    both = pd.concat([starter3, relief1], ignore_index=True)

    rows = []
    for inn, g in both.groupby("inning"):
        cells = (g.groupby(["역할", "batting_order"], observed=True)
                   .agg(**{num: (num, "sum"), den: (den, "sum") if den != "타석" else (num, "size")})
                   .reset_index())
        if cells["역할"].nunique() < 2:
            continue
        w = cells[cells["역할"] == "구원 1바퀴째"].set_index("batting_order")[den]
        t = standardize(cells, "역할", w, num, den)
        if len(t) < 2:
            continue
        diff = t.loc["선발 3바퀴째", "rate"] - t.loc["구원 1바퀴째", "rate"]
        se = np.sqrt(t.loc["선발 3바퀴째", "se"] ** 2 + t.loc["구원 1바퀴째", "se"] ** 2)
        rows.append({"이닝": int(inn),
                     "선발3바퀴": t.loc["선발 3바퀴째", "rate"],
                     "구원1바퀴": t.loc["구원 1바퀴째", "rate"],
                     "차이": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
                     "유의": (diff - 1.96 * se > 0) == (diff + 1.96 * se > 0),
                     "선발n": int(t.loc["선발 3바퀴째", "n"]),
                     "구원n": int(t.loc["구원 1바퀴째", "n"])})
    return pd.DataFrame(rows).set_index("이닝")


def compare_by_tto(pa: pd.DataFrame, num: str, den: str) -> pd.DataFrame:
    """선발을 회전별로 나눠 구원 1바퀴째와 비교 — 어느 지점에서 뒤집히는지."""
    d = pa[pa["inning"].between(5, 8)].copy()
    starters = d[d["is_starter"] & d["타순회전"].between(1, 3)].copy()
    starters["역할"] = "선발 " + starters["타순회전"].astype(int).astype(str) + "바퀴째"
    relief = d[~d["is_starter"] & (d["타순회전"] == 1)].assign(역할="구원 1바퀴째")
    both = pd.concat([starters, relief], ignore_index=True)

    cells = (both.groupby(["역할", "batting_order"], observed=True)
               .agg(**{num: (num, "sum"), den: (den, "sum") if den != "타석" else (num, "size")})
               .reset_index())
    w = cells[cells["역할"] == "구원 1바퀴째"].set_index("batting_order")[den]
    return standardize(cells, "역할", w, num, den)


def fig_hook_value(by_tto: pd.DataFrame, by_inn: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    order = ["선발 1바퀴째", "선발 2바퀴째", "선발 3바퀴째", "구원 1바퀴째"]
    order = [o for o in order if o in by_tto.index]
    colors = [BLUE, BLUE, ORANGE, AQUA][: len(order)]
    x = np.arange(len(order))
    vals = [by_tto.loc[o, "rate"] for o in order]
    errs = [by_tto.loc[o, "se"] * 1.96 for o in order]
    ax.bar(x, vals, color=colors, width=0.58)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK, elinewidth=1.4, capsize=5)
    for xi, v, e, o in zip(x, vals, errs, order):
        ax.text(xi, v + e + 0.004, f"{v:.3f}", ha="center", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
        ax.text(xi, 0.008, f"{int(by_tto.loc[o,'n']):,}", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(" ", "\n") for o in order], fontsize=9)
    ax.set_ylim(0, max(vals) * 1.3)
    _style_ax(ax, "5~8회에서 누가 더 잘 막나", "", "피안타율 (타순 구성 표준화)")

    ax = axes[1]
    x = np.arange(len(by_inn))
    ax.bar(x, by_inn["차이"] * 1000, color=[ORANGE if s else MUTED for s in by_inn["유의"]], width=0.55)
    ax.errorbar(x, by_inn["차이"] * 1000,
                yerr=(by_inn["hi"] - by_inn["차이"]) * 1000, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.4, capsize=5)
    ax.axhline(0, color=MUTED, linewidth=1.3)
    for xi, v in zip(x, by_inn["차이"] * 1000):
        ax.text(xi, v + (3 if v >= 0 else -3), f"{v:+.0f}", ha="center",
                va="bottom" if v >= 0 else "top", color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}회" for i in by_inn.index])
    _style_ax(ax, "교체로 얻는 것 (이닝별)", "", "피안타율 차이 (1/1000 단위)")

    fig.suptitle("3바퀴째 선발을 내리고 구원을 올리면 얼마나 이득인가",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "hook_value.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2017, 2027)))
    a = p.parse_args()

    pa = load(a.seasons)
    by_tto = compare_by_tto(pa, "안타", "타수")
    by_inn = compare_by_inning(pa, "안타", "타수")
    by_tto_ob = compare_by_tto(pa, "출루", "타석")
    fig_hook_value(by_tto, by_inn)

    lines = [f"=== 교체의 가치 ({len(a.seasons)}시즌 · 5~8회 타석) ===\n",
             "선발이 3바퀴째를 맞는 것과, 구원이 타순을 처음 보는 것을 같은 이닝에서 비교한다.",
             "타순 구성은 구원투수가 상대한 분포로 표준화했다.\n"]

    lines.append("[1] 역할별 피안타율 (5~8회)")
    for idx, r in by_tto.iterrows():
        lines.append(f"  {idx}: {r['rate']:.4f} (±{r['se']*1.96:.4f}, {int(r['n']):,}타수)")
    if "선발 3바퀴째" in by_tto.index and "구원 1바퀴째" in by_tto.index:
        d = by_tto.loc["선발 3바퀴째", "rate"] - by_tto.loc["구원 1바퀴째", "rate"]
        se = np.sqrt(by_tto.loc["선발 3바퀴째", "se"] ** 2 + by_tto.loc["구원 1바퀴째", "se"] ** 2)
        lines.append(f"  → 선발 3바퀴째 − 구원 1바퀴째: {d:+.4f} "
                     f"[{d-1.96*se:+.4f}, {d+1.96*se:+.4f}]")
        d2 = by_tto.loc["선발 2바퀴째", "rate"] - by_tto.loc["구원 1바퀴째", "rate"]
        se2 = np.sqrt(by_tto.loc["선발 2바퀴째", "se"] ** 2 + by_tto.loc["구원 1바퀴째", "se"] ** 2)
        lines.append(f"  → 선발 2바퀴째 − 구원 1바퀴째: {d2:+.4f} "
                     f"[{d2-1.96*se2:+.4f}, {d2+1.96*se2:+.4f}]")
    lines.append("")

    lines.append("[2] 피출루율로 봐도 같은가")
    for idx, r in by_tto_ob.iterrows():
        lines.append(f"  {idx}: {r['rate']:.4f} (±{r['se']*1.96:.4f}, {int(r['n']):,}타석)")
    lines.append("")

    lines.append("[3] 이닝별 — 교체로 얻는 피안타율 (양수면 교체가 이득)")
    for idx, r in by_inn.iterrows():
        mark = "유의함" if r["유의"] else "유의하지 않음"
        lines.append(f"  {idx}회: 선발3바퀴 {r['선발3바퀴']:.4f} vs 구원1바퀴 {r['구원1바퀴']:.4f} "
                     f"→ {r['차이']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {mark} "
                     f"(선발 {r['선발n']:,}타수 / 구원 {r['구원n']:,}타수)")

    out = "\n".join(lines)
    path = RESULTS / "eda_hook_value_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
