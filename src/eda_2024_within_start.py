"""등판 '안에서' 일어나는 피로 — 타순 회전별·투구수 구간별 선발 성적.

지금까지는 등판 하나를 한 줄로 집계한 데이터만 있어서 '80구를 넘기면서 실제로
나빠지는가'를 볼 수 없었다. 문자중계에서 타석 단위로 펼친 데이터가 생겼으니
같은 등판 안에서 시간이 갈수록 성적이 어떻게 변하는지 직접 계산한다.

핵심 주의: 투구수가 쌓인 구간에는 '잘 던지고 있어서 계속 남은 투수'만 살아남는다
(생존 편향). 그래서 전체 평균 비교와 함께, 끝까지 간 등판만 추려 같은 투수들
안에서 비교한 결과를 같이 낸다.
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
AQUA = "#1baf7a"
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
    return pa


def _rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "타석": g.size(),
        "타수": g["타수"].sum(),
        "안타": g["안타"].sum(),
    })
    out["피안타율"] = out["안타"] / out["타수"]
    out["표준오차"] = np.sqrt(out["피안타율"] * (1 - out["피안타율"]) / out["타수"])
    return out


def fig_tto(pa: pd.DataFrame) -> pd.DataFrame:
    d = pa[pa["타순회전"].between(1, 3)].copy()
    table = _rate_table(d, "타순회전")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150, facecolor=SURFACE)
    x = table.index.astype(int)
    ax.bar(x, table["피안타율"], color=BLUE, width=0.5)
    ax.errorbar(x, table["피안타율"], yerr=table["표준오차"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.4, capsize=6)
    for xi, row in zip(x, table.itertuples()):
        ax.text(xi, row.피안타율 + row.표준오차 * 1.96 + 0.004, f"{row.피안타율:.3f}",
                ha="center", color=SECONDARY_INK, fontsize=10, fontweight="bold")
        ax.text(xi, 0.006, f"{int(row.타수):,}타수", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}바퀴째" for i in x])
    ax.set_ylim(0, max(table["피안타율"]) * 1.28)
    _style_ax(ax, f"타순 회전별 선발투수 피안타율 ({SEASON})", "그 선발이 타순을 도는 횟수", "피안타율")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_tto_avg.png")
    plt.close(fig)
    return table


def fig_pitch_bucket(pa: pd.DataFrame) -> pd.DataFrame:
    d = pa.copy()
    bins = [0, 25, 50, 75, 100, 999]
    labels = ["1~25구", "26~50구", "51~75구", "76~100구", "101구+"]
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], bins=bins, labels=labels)
    table = _rate_table(d.dropna(subset=["투구수_구간"]), "투구수_구간")

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    x = np.arange(len(table))
    ax.bar(x, table["피안타율"], color=BLUE, width=0.55)
    ax.errorbar(x, table["피안타율"], yerr=table["표준오차"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.4, capsize=6)
    for xi, row in zip(x, table.itertuples()):
        ax.text(xi, row.피안타율 + row.표준오차 * 1.96 + 0.004, f"{row.피안타율:.3f}",
                ha="center", color=SECONDARY_INK, fontsize=10, fontweight="bold")
        ax.text(xi, 0.006, f"{int(row.타수):,}타수", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index.astype(str))
    ax.set_ylim(0, max(table["피안타율"]) * 1.28)
    _style_ax(ax, f"누적 투구수 구간별 선발투수 피안타율 ({SEASON})",
              "그 타석 시점의 누적 투구수", "피안타율")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_pitch_bucket_avg.png")
    plt.close(fig)
    return table


def fig_survivor_control(pa: pd.DataFrame) -> pd.DataFrame:
    """생존 편향 통제 — 3바퀴째까지 간 등판만 추려서 같은 등판 안에서 비교한다."""
    reached = (
        pa[pa["타순회전"] >= 3].groupby(["game_id", "pitcher"]).size().reset_index()[["game_id", "pitcher"]]
    )
    d = pa.merge(reached, on=["game_id", "pitcher"], how="inner")
    d = d[d["타순회전"].between(1, 3)]
    table = _rate_table(d, "타순회전")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150, facecolor=SURFACE)
    x = table.index.astype(int)
    ax.bar(x, table["피안타율"], color=AQUA, width=0.5)
    ax.errorbar(x, table["피안타율"], yerr=table["표준오차"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.4, capsize=6)
    for xi, row in zip(x, table.itertuples()):
        ax.text(xi, row.피안타율 + row.표준오차 * 1.96 + 0.004, f"{row.피안타율:.3f}",
                ha="center", color=SECONDARY_INK, fontsize=10, fontweight="bold")
        ax.text(xi, 0.006, f"{int(row.타수):,}타수", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}바퀴째" for i in x])
    ax.set_ylim(0, max(table["피안타율"]) * 1.28)
    _style_ax(ax, f"3바퀴째까지 간 등판만 — 같은 투수들 안에서 비교 ({SEASON})",
              "그 선발이 타순을 도는 횟수", "피안타율")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_tto_survivor.png")
    plt.close(fig)
    return table


def fig_pitch_bucket_survivor(pa: pd.DataFrame) -> pd.DataFrame:
    """투구수 쪽에도 타순 회전과 같은 통제를 건다 — 90구 이상 간 등판만 비교."""
    reached = (
        pa[pa["투수_누적투구수"] >= 90].groupby(["game_id", "pitcher"]).size()
        .reset_index()[["game_id", "pitcher"]]
    )
    d = pa.merge(reached, on=["game_id", "pitcher"], how="inner").copy()
    bins = [0, 25, 50, 75, 100]
    labels = ["1~25구", "26~50구", "51~75구", "76~100구"]
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], bins=bins, labels=labels)
    table = _rate_table(d.dropna(subset=["투구수_구간"]), "투구수_구간")

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    x = np.arange(len(table))
    ax.bar(x, table["피안타율"], color=AQUA, width=0.55)
    ax.errorbar(x, table["피안타율"], yerr=table["표준오차"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.4, capsize=6)
    for xi, row in zip(x, table.itertuples()):
        ax.text(xi, row.피안타율 + row.표준오차 * 1.96 + 0.004, f"{row.피안타율:.3f}",
                ha="center", color=SECONDARY_INK, fontsize=10, fontweight="bold")
        ax.text(xi, 0.006, f"{int(row.타수):,}타수", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index.astype(str))
    ax.set_ylim(0, max(table["피안타율"]) * 1.28)
    _style_ax(ax, f"90구 이상 던진 등판만 — 투구수 구간별 피안타율 ({SEASON})",
              "그 타석 시점의 누적 투구수", "피안타율")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_pitch_bucket_survivor.png")
    plt.close(fig)
    return table


def fig_disentangle(pa: pd.DataFrame) -> pd.DataFrame:
    """타순 회전과 투구수를 갈라놓는 결정적 비교.

    같은 '몇 바퀴째'인 타석들만 모아 놓고, 그 안에서 누적 투구수가 많은 쪽과 적은 쪽을
    가른다. 회전 수가 고정된 상태에서도 투구수가 영향을 준다면 진짜 피로가 있는 것이고,
    차이가 없다면 문제는 피로가 아니라 '타자가 그 투수를 여러 번 봤다는 사실'이다.
    """
    d = pa[pa["타순회전"].between(1, 3)].copy()
    rows = []
    for tto, grp in d.groupby("타순회전"):
        median = grp["투수_누적투구수"].median()
        for side, sub in (("투구수 적음", grp[grp["투수_누적투구수"] < median]),
                          ("투구수 많음", grp[grp["투수_누적투구수"] >= median])):
            ab, h = sub["타수"].sum(), sub["안타"].sum()
            rows.append({"타순회전": int(tto), "구분": side, "기준투구수": median,
                         "타수": int(ab), "안타": int(h), "피안타율": h / ab})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    ttos = sorted(table["타순회전"].unique())
    width = 0.34
    x = np.arange(len(ttos))
    for offset, (side, color) in zip((-width / 2, width / 2),
                                     (("투구수 적음", BLUE), ("투구수 많음", ORANGE))):
        vals = [table[(table["타순회전"] == t) & (table["구분"] == side)]["피안타율"].iloc[0] for t in ttos]
        ax.bar(x + offset, vals, width=width, color=color, label=side)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", color=SECONDARY_INK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}바퀴째" for t in ttos])
    ax.set_ylim(0, table["피안타율"].max() * 1.3)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")
    _style_ax(ax, f"같은 바퀴 안에서 투구수만 다르면? ({SEASON})",
              "타순 회전", "피안타율")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_disentangle.png")
    plt.close(fig)
    return table


def _standardize(cells: pd.DataFrame, level_col: str, weights: pd.Series) -> pd.DataFrame:
    """타순(스트라텀)별 비율을 공통 가중치로 표준화한다 — 직접표준화.

    타순 회전이 올라갈수록 상위타순 비중이 커지고(3바퀴째는 1~5번을 주로 상대한다),
    같은 바퀴 안에서도 투구수가 쌓인 타석은 하위타순 쪽에 몰린다. 두 경우 모두
    '타자가 누구였나'가 섞여 들어오므로, 모든 그룹이 같은 타순 구성을 상대한 것처럼
    맞춰놓고 비교해야 한다.
    """
    out = []
    for level, grp in cells.groupby(level_col):
        g = grp.set_index("batting_order")
        common = weights.index.intersection(g.index)
        w = weights.loc[common] / weights.loc[common].sum()
        p = (g.loc[common, "안타"] / g.loc[common, "타수"]).astype(float)
        var = (w ** 2 * p * (1 - p) / g.loc[common, "타수"]).sum()
        out.append({level_col: level, "표준화_피안타율": float((w * p).sum()),
                    "표준오차": float(np.sqrt(var)),
                    "타수": int(g.loc[common, "타수"].sum())})
    return pd.DataFrame(out).set_index(level_col)


def _cells(df: pd.DataFrame, level_col: str) -> pd.DataFrame:
    return (df.groupby([level_col, "batting_order"], observed=True)
              .agg(안타=("안타", "sum"), 타수=("타수", "sum"))
              .reset_index())


def fig_tto_standardized(pa: pd.DataFrame) -> pd.DataFrame:
    """타순 구성을 맞춘 뒤의 타순 회전 효과 — 이 분석의 핵심 수치."""
    reached = (pa[pa["타순회전"] >= 3].groupby(["game_id", "pitcher"]).size()
               .reset_index()[["game_id", "pitcher"]])
    d = pa.merge(reached, on=["game_id", "pitcher"], how="inner")
    d = d[d["타순회전"].between(1, 3)]

    cells = _cells(d, "타순회전")
    # 3바퀴째가 실제로 상대한 타순 구성을 기준으로 1·2바퀴째를 맞춘다.
    weights = cells[cells["타순회전"] == 3].set_index("batting_order")["타수"]
    table = _standardize(cells, "타순회전", weights)

    # 0부터 그리는 막대는 .260과 .283의 차이를 가린다 -- 축을 자른 점 그래프로.
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150, facecolor=SURFACE)
    x = table.index.astype(int)
    lo = (table["표준화_피안타율"] - table["표준오차"] * 1.96).min()
    hi = (table["표준화_피안타율"] + table["표준오차"] * 1.96).max()
    pad = (hi - lo) * 0.45
    ax.plot(x, table["표준화_피안타율"], color=BLUE, linewidth=1.6,
            marker="o", markersize=11, zorder=3)
    ax.errorbar(x, table["표준화_피안타율"], yerr=table["표준오차"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.5, capsize=7, zorder=2)
    for xi, row in zip(x, table.itertuples()):
        ax.text(xi, row.표준화_피안타율 + row.표준오차 * 1.96 + pad * 0.12,
                f"{row.표준화_피안타율:.3f}", ha="center", color=SECONDARY_INK,
                fontsize=11, fontweight="bold")
        ax.text(xi, lo - pad * 0.72, f"{int(row.타수):,}타수", ha="center",
                color=MUTED, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}바퀴째" for i in x])
    ax.set_xlim(x.min() - 0.45, x.max() + 0.45)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    _style_ax(ax, f"타순 구성까지 맞춘 타순 회전 효과 ({SEASON})",
              "그 선발이 타순을 도는 횟수", "표준화 피안타율 (축을 잘랐다)")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_tto_standardized.png")
    plt.close(fig)
    return table


def fig_disentangle_standardized(pa: pd.DataFrame) -> pd.DataFrame:
    """타순 회전과 타순을 동시에 고정한 뒤 투구수만 비교한다."""
    d = pa[pa["타순회전"].between(1, 3)].copy()
    # (바퀴, 타순) 칸마다 그 칸의 중앙값으로 투구수 많고 적음을 가른다.
    d["기준"] = d.groupby(["타순회전", "batting_order"])["투수_누적투구수"].transform("median")
    d["구분"] = np.where(d["투수_누적투구수"] >= d["기준"], "투구수 많음", "투구수 적음")

    cells = _cells(d, "구분")
    weights = cells.groupby("batting_order")["타수"].sum()
    table = _standardize(cells, "구분", weights)

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150, facecolor=SURFACE)
    order = ["투구수 적음", "투구수 많음"]
    x = np.arange(len(order))
    vals = [table.loc[o, "표준화_피안타율"] for o in order]
    errs = [table.loc[o, "표준오차"] * 1.96 for o in order]
    ax.bar(x, vals, color=[BLUE, ORANGE], width=0.45)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK, elinewidth=1.4, capsize=6)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.004, f"{v:.3f}", ha="center", color=SECONDARY_INK,
                fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["누적 투구수 적은 타석", "누적 투구수 많은 타석"])
    ax.set_ylim(0, max(vals) * 1.3)
    _style_ax(ax, f"바퀴와 타순을 고정하고 투구수만 비교 ({SEASON})", "", "표준화 피안타율")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_disentangle_standardized.png")
    plt.close(fig)
    return table


def std_diff_ci(table: pd.DataFrame, a, b) -> tuple[float, float, float, bool]:
    """표준화된 두 비율의 차이와 95% 신뢰구간."""
    diff = table.loc[b, "표준화_피안타율"] - table.loc[a, "표준화_피안타율"]
    se = np.sqrt(table.loc[a, "표준오차"] ** 2 + table.loc[b, "표준오차"] ** 2)
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return diff, lo, hi, (lo > 0) == (hi > 0)


def prop_ci(h1, ab1, h2, ab2) -> tuple[float, float, float, bool]:
    """두 비율 차이의 95% 신뢰구간 (정규근사)."""
    p1, p2 = h1 / ab1, h2 / ab2
    diff = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / ab1 + p2 * (1 - p2) / ab2)
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return diff, lo, hi, (lo > 0) == (hi > 0)


def summarize(pa: pd.DataFrame, tto, bucket, survivor, bucket_surv, disent, tto_std, disent_std) -> None:
    n_games = pa["game_id"].nunique()
    lines = [f"=== {SEASON} 등판 안에서의 피로 — 타석 {len(pa):,}건 / {n_games}경기 ===\n"]

    lines.append("[1] 타순 회전별 피안타율 (전체 선발)")
    for idx, row in tto.iterrows():
        lines.append(f"  - {idx}바퀴째: {row['피안타율']:.3f} "
                     f"({int(row['안타'])}안타/{int(row['타수'])}타수, 타석 {int(row['타석'])})")
    d, lo, hi, sig = prop_ci(tto.loc[1, "안타"], tto.loc[1, "타수"], tto.loc[3, "안타"], tto.loc[3, "타수"])
    lines.append(f"  → 1바퀴 대비 3바퀴: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                 f"{'유의함' if sig else '유의하지 않음'}\n")

    lines.append("[2] 누적 투구수 구간별 피안타율")
    for idx, row in bucket.iterrows():
        lines.append(f"  - {idx}: {row['피안타율']:.3f} "
                     f"({int(row['안타'])}안타/{int(row['타수'])}타수)")
    lines.append("")

    lines.append("[3] 생존 편향 통제 — 3바퀴째까지 간 등판만")
    for idx, row in survivor.iterrows():
        lines.append(f"  - {idx}바퀴째: {row['피안타율']:.3f} "
                     f"({int(row['안타'])}안타/{int(row['타수'])}타수)")
    d, lo, hi, sig = prop_ci(survivor.loc[1, "안타"], survivor.loc[1, "타수"],
                             survivor.loc[3, "안타"], survivor.loc[3, "타수"])
    lines.append(f"  → 1바퀴 대비 3바퀴: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                 f"{'유의함' if sig else '유의하지 않음'}\n")

    lines.append("[4] 생존 편향 통제 — 90구 이상 던진 등판만, 투구수 구간별")
    for idx, row in bucket_surv.iterrows():
        lines.append(f"  - {idx}: {row['피안타율']:.3f} "
                     f"({int(row['안타'])}안타/{int(row['타수'])}타수)")
    first, last = bucket_surv.index[0], bucket_surv.index[-1]
    d, lo, hi, sig = prop_ci(bucket_surv.loc[first, "안타"], bucket_surv.loc[first, "타수"],
                             bucket_surv.loc[last, "안타"], bucket_surv.loc[last, "타수"])
    lines.append(f"  → {first} 대비 {last}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                 f"{'유의함' if sig else '유의하지 않음'}\n")

    lines.append("[5] 같은 바퀴 안에서 투구수만 다르게 (타순 통제 없음 — 교란됨)")
    for tto_n in sorted(disent["타순회전"].unique()):
        sub = disent[disent["타순회전"] == tto_n]
        low = sub[sub["구분"] == "투구수 적음"].iloc[0]
        high = sub[sub["구분"] == "투구수 많음"].iloc[0]
        d, lo, hi, sig = prop_ci(low["안타"], low["타수"], high["안타"], high["타수"])
        lines.append(f"  - {tto_n}바퀴째 (기준 {low['기준투구수']:.0f}구): "
                     f"적음 {low['피안타율']:.3f} vs 많음 {high['피안타율']:.3f} → "
                     f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] {'유의함' if sig else '유의하지 않음'}")
    lines.append("  ※ 같은 바퀴 안에서 투구수가 많다 = 타순 뒤쪽(하위타선)이라는 뜻이라")
    lines.append("     이 비교는 투수가 아니라 타자 수준 차이를 잡아낸다. [6][7]에서 보정한다.\n")

    lines.append("[6] 타순 구성까지 맞춘 타순 회전 효과 (직접표준화)")
    for idx, row in tto_std.iterrows():
        lines.append(f"  - {idx}바퀴째: {row['표준화_피안타율']:.3f} "
                     f"(±{row['표준오차']*1.96:.3f}, {int(row['타수']):,}타수)")
    d, lo, hi, sig = std_diff_ci(tto_std, 1, 3)
    lines.append(f"  → 1바퀴 대비 3바퀴: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                 f"{'유의함' if sig else '유의하지 않음'}\n")

    lines.append("[7] 바퀴와 타순을 고정하고 투구수만 비교 (직접표준화)")
    for idx, row in disent_std.iterrows():
        lines.append(f"  - {idx}: {row['표준화_피안타율']:.3f} "
                     f"(±{row['표준오차']*1.96:.3f}, {int(row['타수']):,}타수)")
    d, lo, hi, sig = std_diff_ci(disent_std, "투구수 적음", "투구수 많음")
    lines.append(f"  → 투구수 적음 대비 많음: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                 f"{'유의함' if sig else '유의하지 않음'}")

    out = "\n".join(lines)
    path = FIG.parent / f"eda_{SEASON}_within_start_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


def main() -> None:
    pa = load()
    tto = fig_tto(pa)
    bucket = fig_pitch_bucket(pa)
    survivor = fig_survivor_control(pa)
    bucket_surv = fig_pitch_bucket_survivor(pa)
    disent = fig_disentangle(pa)
    tto_std = fig_tto_standardized(pa)
    disent_std = fig_disentangle_standardized(pa)
    summarize(pa, tto, bucket, survivor, bucket_surv, disent, tto_std, disent_std)
    print(f"그림 7장 -> {FIG}")


if __name__ == "__main__":
    main()
