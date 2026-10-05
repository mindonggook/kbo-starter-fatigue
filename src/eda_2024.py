"""2024 시즌 선발투수 피로도/교체 타이밍 1차 탐색적 분석(EDA).

data/kbo_pitcher_appearances_2024.parquet(등판 기록)과
data/kbo_schedule_2024.parquet(경기 결과)을 합쳐 그림 5장 + 요약 텍스트를 낸다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
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


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    appearances = pd.read_parquet(DATA / "kbo_pitcher_appearances_2024.parquet")
    schedule = pd.read_parquet(DATA / "kbo_schedule_2024.parquet")
    return appearances, schedule


def fig_pitch_count_hist(starters: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
    ax.hist(starters["투구수"], bins=24, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    median = starters["투구수"].median()
    ax.axvline(median, color=ORANGE, linewidth=2)
    ax.text(median + 1.5, ax.get_ylim()[1] * 0.92, f"중앙값 {median:.0f}구",
            color=ORANGE, fontsize=10, fontweight="bold")
    _style_ax(ax, "선발투수 강판 시점 — 투구수 분포 (2024)", "투구수", "선발 등판 수")
    fig.tight_layout()
    fig.savefig(FIG / "2024_starter_pitch_count_hist.png")
    plt.close(fig)


def fig_innings_bar(starters: pd.DataFrame) -> None:
    counts = starters["이닝_소수"].apply(lambda x: int(x) if x == int(x) else int(x)).value_counts()
    # 완투 등 드문 구간은 자연스럽게 작은 막대로 보이도록 0~9 전 구간을 채운다.
    counts = counts.reindex(range(0, 10), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
    ax.bar(counts.index, counts.values, color=BLUE, width=0.6)
    _style_ax(ax, "선발투수 강판 시점 — 소화 이닝 분포 (2024)", "완료 이닝 수(그 이상 소화 후 강판)", "선발 등판 수")
    ax.set_xticks(range(0, 10))
    fig.tight_layout()
    fig.savefig(FIG / "2024_starter_innings_bar.png")
    plt.close(fig)


def fig_team_avg_pitch_count(starters: pd.DataFrame) -> None:
    team_avg = starters.groupby("team")["투구수"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor=SURFACE)
    ax.barh(team_avg.index, team_avg.values, color=BLUE, height=0.6)
    league_avg = starters["투구수"].mean()
    ax.axvline(league_avg, color=MUTED, linewidth=1.5, linestyle="--")
    ax.text(league_avg + 0.3, -0.7, f"리그 평균 {league_avg:.1f}구", color=MUTED, fontsize=9)
    _style_ax(ax, "팀별 선발투수 평균 투구수 (2024)", "평균 투구수", "")
    fig.tight_layout()
    fig.savefig(FIG / "2024_team_avg_pitch_count.png")
    plt.close(fig)


def fig_entry_heatmap(relievers: pd.DataFrame) -> None:
    valid = relievers.dropna(subset=["entry_inning", "entry_batting_order"])
    valid = valid[valid["entry_inning"].between(1, 9) & valid["entry_batting_order"].between(1, 9)]
    pivot = (
        valid.groupby(["entry_batting_order", "entry_inning"]).size()
        .unstack(fill_value=0)
        .reindex(index=range(1, 10), columns=range(1, 10), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150, facecolor=SURFACE)
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(9))
    ax.set_xticklabels(range(1, 10))
    ax.set_yticks(range(9))
    ax.set_yticklabels(range(1, 10))
    for i in range(9):
        for j in range(9):
            v = pivot.values[i, j]
            if v > 0:
                color = "white" if v > pivot.values.max() * 0.6 else INK
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.set_label("구원 등판 건수", color=SECONDARY_INK, fontsize=9)
    _style_ax(ax, "구원투수 최초 등판 시점 — 회 × 타순 (2024)", "등판한 회", "등판 시 타순")
    fig.tight_layout()
    fig.savefig(FIG / "2024_reliever_entry_heatmap.png")
    plt.close(fig)


def fig_pitch_count_vs_er(starters: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor=SURFACE)
    ax.scatter(starters["투구수"], starters["자책"], s=22, color=BLUE, alpha=0.35, edgecolors="none")
    _style_ax(ax, "선발 등판 1회당 투구수 vs 자책점 (2024)", "투구수", "자책점")
    fig.tight_layout()
    fig.savefig(FIG / "2024_pitch_count_vs_er.png")
    plt.close(fig)


def print_summary(appearances: pd.DataFrame, schedule: pd.DataFrame, starters: pd.DataFrame) -> None:
    relievers = appearances[~appearances["is_starter"]]
    lines = []
    lines.append("=== 2024 시즌 선발투수 피로도/교체 타이밍 EDA 요약 ===\n")

    lines.append(f"[강판 시점] 중앙값 {starters['투구수'].median():.0f}구 / "
                 f"{starters['이닝_소수'].median():.1f}이닝, "
                 f"IQR {starters['투구수'].quantile(.25):.0f}~{starters['투구수'].quantile(.75):.0f}구")

    team_avg = starters.groupby("team")["투구수"].mean().sort_values(ascending=False)
    lines.append(f"[팀별 성향] 선발을 가장 길게 쓰는 팀: {team_avg.index[0]} ({team_avg.iloc[0]:.1f}구) / "
                 f"가장 짧게 쓰는 팀: {team_avg.index[-1]} ({team_avg.iloc[-1]:.1f}구)")

    valid_entry = relievers.dropna(subset=["entry_inning"])
    top_inning = valid_entry["entry_inning"].value_counts().idxmax()
    lines.append(f"[교체 타이밍] 구원 등판이 가장 몰리는 회차: {top_inning}회 "
                 f"({(valid_entry['entry_inning'] == top_inning).sum()}건, "
                 f"전체 구원 등판의 {(valid_entry['entry_inning'] == top_inning).mean()*100:.1f}%)")

    top_order = valid_entry["entry_batting_order"].value_counts().idxmax()
    lines.append(f"[교체 타이밍] 구원 등판이 가장 몰리는 타순: {top_order}번 "
                 f"({(valid_entry['entry_batting_order'] == top_order).sum()}건)")

    # 완투(9이닝) 비중
    complete = (starters["이닝_소수"] >= 9).sum()
    lines.append(f"[완투] 2024시즌 선발 완투: {complete}건 / 전체 선발 등판 {len(starters)}건 "
                 f"({complete/len(starters)*100:.2f}%)")

    # 승부와 관계 없이 5이닝 못 채운(퀄리티스타트 미달 가능성 큰) 조기강판 비중
    short = (starters["이닝_소수"] < 5).sum()
    lines.append(f"[조기강판] 5이닝 미만 강판: {short}건 ({short/len(starters)*100:.1f}%)")

    out = "\n".join(lines)
    print(out)
    (FIG.parent / "eda_2024_summary.txt").write_text(out, encoding="utf-8")


def main() -> None:
    appearances, schedule = load()
    starters = appearances[appearances["is_starter"]].copy()
    relievers = appearances[~appearances["is_starter"]].copy()

    fig_pitch_count_hist(starters)
    fig_innings_bar(starters)
    fig_team_avg_pitch_count(starters)
    fig_entry_heatmap(relievers)
    fig_pitch_count_vs_er(starters)
    print_summary(appearances, schedule, starters)
    print(f"\n그림 5장 -> {FIG}")


if __name__ == "__main__":
    main()
