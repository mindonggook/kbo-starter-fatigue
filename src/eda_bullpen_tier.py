"""발견 15의 약점 보강 — '평균 구원투수'가 아니라 '지금 대기 중인 그 투수'.

발견 15는 선발 3바퀴째(.2727)와 구원 1바퀴째(.2707)가 같다고 했지만, 그 구원은
리그 평균이다. 실제 교체 결정은 평균이 아니라 '뒤에 누가 있는가'로 내려진다.

구원투수를 감독이 드러낸 신뢰도로 나눈다 — 그 시즌에 세이브·홀드를 몇 개
받았는지다. 성적으로 나누면 '잘하는 투수가 잘한다'는 동어반복이 되지만,
보직은 결과를 보기 전에 감독이 내린 판단이라 우리가 재는 값과 독립적이다.

  필승조   세이브+홀드 15개 이상
  중간     5~14개
  추격조   5개 미만
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbo_client import TEAM_CODE

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
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

TIERS = ["추격조", "중간", "필승조"]
TIER_COLORS = {"추격조": MUTED, "중간": AQUA, "필승조": VIOLET}


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


def build_tiers(seasons: list[int]) -> pd.DataFrame:
    """시즌·투수별 세이브+홀드 수로 보직 등급을 매긴다."""
    rows = []
    for s in seasons:
        box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{s}.parquet")
        rel = box[~box["is_starter"]].copy()
        rel["결과"] = rel["결과"].fillna("")
        rel["성공"] = rel["결과"].str.contains("세|홀드", regex=True).astype(int)
        g = rel.groupby(["team", "선수명"])["성공"].sum().reset_index()
        g["season"] = s
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    t["등급"] = np.select(
        [t["성공"] >= 15, t["성공"] >= 5],
        ["필승조", "중간"], default="추격조")
    return t.rename(columns={"선수명": "pitcher", "team": "pitcher_team"})


def load_pa(seasons: list[int]) -> pd.DataFrame:
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

    # 투수의 소속팀은 gameId(원정/홈 코드)와 초·말로 결정된다.
    away = pa["game_id"].str[8:10].map(TEAM_CODE)
    home = pa["game_id"].str[10:12].map(TEAM_CODE)
    pa["pitcher_team"] = np.where(pa["half"] == "초", home, away)
    return pa


def standardize(cells: pd.DataFrame, level: str, weights: pd.Series) -> pd.DataFrame:
    out = []
    for lv, grp in cells.groupby(level, observed=True):
        g = grp.set_index("batting_order")
        common = weights.index.intersection(g.index)
        if len(common) == 0:
            continue
        w = weights.loc[common] / weights.loc[common].sum()
        p = (g.loc[common, "안타"] / g.loc[common, "타수"]).astype(float)
        var = (w ** 2 * p * (1 - p) / g.loc[common, "타수"]).sum()
        out.append({level: lv, "rate": float((w * p).sum()),
                    "se": float(np.sqrt(var)), "n": int(g.loc[common, "타수"].sum())})
    return pd.DataFrame(out).set_index(level)


def analyse(pa: pd.DataFrame, tiers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pa[pa["inning"].between(5, 8)].copy()

    starter3 = d[d["is_starter"] & (d["타순회전"] >= 3)].assign(역할="선발 3바퀴째")
    relief = d[~d["is_starter"] & (d["타순회전"] == 1)].merge(
        tiers, on=["season", "pitcher", "pitcher_team"], how="left")
    relief = relief.dropna(subset=["등급"])
    relief["역할"] = "구원 " + relief["등급"]

    both = pd.concat([starter3, relief], ignore_index=True)
    cells = (both.groupby(["역할", "batting_order"], observed=True)
               .agg(안타=("안타", "sum"), 타수=("타수", "sum")).reset_index())
    w = cells[cells["역할"] == "선발 3바퀴째"].set_index("batting_order")["타수"]
    table = standardize(cells, "역할", w)

    usage = (relief.groupby("등급").size() / len(relief) * 100).rename("5~8회 구원 타석 비중")
    return table, usage.to_frame()


def diff_vs_starter(table: pd.DataFrame, tier: str) -> dict:
    base = table.loc["선발 3바퀴째"]
    row = table.loc[f"구원 {tier}"]
    d = base["rate"] - row["rate"]          # 양수면 교체가 이득
    se = np.sqrt(base["se"] ** 2 + row["se"] ** 2)
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return {"diff": d, "lo": lo, "hi": hi, "유의": (lo > 0) == (hi > 0)}


def fig_tier(table: pd.DataFrame, diffs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    order = ["선발 3바퀴째"] + [f"구원 {t}" for t in TIERS]
    order = [o for o in order if o in table.index]
    colors = [ORANGE] + [TIER_COLORS[t] for t in TIERS if f"구원 {t}" in table.index]
    x = np.arange(len(order))
    vals = [table.loc[o, "rate"] for o in order]
    errs = [table.loc[o, "se"] * 1.96 for o in order]
    ax.bar(x, vals, color=colors, width=0.58)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK, elinewidth=1.4, capsize=5)
    ax.axhline(table.loc["선발 3바퀴째", "rate"], color=ORANGE, linewidth=1.4, linestyle="--", alpha=.7)
    for xi, v, e, o in zip(x, vals, errs, order):
        ax.text(xi, v + e + 0.004, f"{v:.3f}", ha="center", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
        ax.text(xi, 0.008, f"{int(table.loc[o,'n']):,}", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(" ", "\n") for o in order], fontsize=9)
    ax.set_ylim(0, max(vals) * 1.3)
    _style_ax(ax, "5~8회, 불펜 등급별 피안타율", "", "피안타율 (타순 구성 표준화)")

    ax = axes[1]
    tiers_present = [t for t in TIERS if t in diffs]
    y = np.arange(len(tiers_present))[::-1]
    for yi, t in zip(y, tiers_present):
        r = diffs[t]
        color = TIER_COLORS[t] if r["유의"] else MUTED
        ax.plot([r["lo"] * 1000, r["hi"] * 1000], [yi, yi], color=color,
                linewidth=2.6, solid_capstyle="round")
        ax.plot(r["diff"] * 1000, yi, "o", color=color, markersize=10)
        ax.text(r["hi"] * 1000 + 1.5, yi, f"{r['diff']*1000:+.0f}", va="center",
                color=SECONDARY_INK, fontsize=10, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}로 교체" for t in tiers_present], fontsize=10)
    _style_ax(ax, "교체로 얻는 것 (양수면 이득)", "피안타율 차이 (1/1000 단위)", "")

    fig.suptitle("뒤에 누가 있느냐가 답을 바꾼다", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "bullpen_tier.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2017, 2027)))
    a = p.parse_args()

    tiers = build_tiers(a.seasons)
    pa = load_pa(a.seasons)
    table, usage = analyse(pa, tiers)
    diffs = {t: diff_vs_starter(table, t) for t in TIERS if f"구원 {t}" in table.index}
    fig_tier(table, diffs)

    lines = [f"=== 불펜 등급별 교체 가치 ({len(a.seasons)}시즌 · 5~8회) ===\n",
             "등급은 그 시즌 세이브+홀드 수로 매겼다(감독의 사전 판단, 결과와 독립).",
             "타순 구성은 선발 3바퀴째가 상대한 분포로 표준화했다.\n"]

    lines.append("[1] 역할별 피안타율")
    for idx, r in table.iterrows():
        lines.append(f"  {idx}: {r['rate']:.4f} (±{r['se']*1.96:.4f}, {int(r['n']):,}타수)")
    lines.append("")

    lines.append("[2] 선발 3바퀴째를 내리고 그 등급으로 바꾸면 (양수면 이득)")
    for t in TIERS:
        if t not in diffs:
            continue
        r = diffs[t]
        mark = "유의함" if r["유의"] else "유의하지 않음"
        lines.append(f"  {t}: {r['diff']*1000:+.1f}/1000 "
                     f"[{r['lo']*1000:+.1f}, {r['hi']*1000:+.1f}] {mark}")
    lines.append("")

    lines.append("[3] 5~8회 구원 타석의 등급 분포")
    for idx, r in usage.iterrows():
        lines.append(f"  {idx}: {r['5~8회 구원 타석 비중']:.1f}%")

    out = "\n".join(lines)
    path = RESULTS / "eda_bullpen_tier_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
