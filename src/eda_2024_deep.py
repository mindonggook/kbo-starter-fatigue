"""2024 시즌 선발투수 심층 분석 — 조기강판 분해 / 누적 피로 / 교체 임계값.

1차 EDA(eda_2024.py)에서 나온 "선발의 31%가 5이닝을 못 채운다"를 출발점으로,
(1) 조기강판이 부진 때문인지 투구수 소진 때문인지 분해하고
(2) 휴식일·직전 등판 투구수·시즌 등판 순번 같은 누적 피로 축을 본 뒤
(3) 감독이 타순 회전/투구수 임계값을 의식하는지 확인한다.
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


def load_starters() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "kbo_pitcher_appearances_2024.parquet")
    s = df[df["is_starter"]].copy()
    s["date"] = pd.to_datetime(s["date"], format="%Y%m%d")
    s["투구수"] = s["투구수"].astype(float)
    s["타자"] = s["타자"].astype(float)
    s["자책"] = s["자책"].astype(float)
    s["실점"] = s["실점"].astype(float)

    # 이닝 0인 등판(아웃카운트 0)은 P/IP가 무한대가 되므로 분모를 아웃 기준으로 잡는다.
    s["아웃"] = (s["이닝_소수"] * 3).round()
    s["P_per_IP"] = np.where(s["아웃"] > 0, s["투구수"] / (s["아웃"] / 3), np.nan)
    s["P_per_BF"] = np.where(s["타자"] > 0, s["투구수"] / s["타자"], np.nan)
    s["조기강판"] = s["이닝_소수"] < 5
    return s.sort_values(["선수명", "team", "date"]).reset_index(drop=True)


def classify_early_hook(s: pd.DataFrame) -> pd.DataFrame:
    """5이닝 미만 강판을 원인별로 나눈다.

    극단형: 투구수 40구 미만 — 정상적인 선발 운용으로 보기 어려운 케이스(부상/오프너).
    소진형: 실점은 3점 이하인데 투구수 80구 이상 — 맞아서가 아니라 투구수를 소모해서 내려간 쪽.
    부진형: 자책 4점 이상 — 두들겨 맞아서 내려간 쪽.
    """
    early = s[s["조기강판"]].copy()
    conditions = [
        early["투구수"] < 40,
        (early["자책"] <= 3) & (early["투구수"] >= 80),
        early["자책"] >= 4,
    ]
    labels = ["극단형(부상·오프너 의심)", "소진형(투구수 소모)", "부진형(피실점)"]
    early["유형"] = np.select(conditions, labels, default="혼합형")
    return early


def fig_early_hook_types(early: pd.DataFrame, n_starts: int) -> None:
    counts = early["유형"].value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=SURFACE)
    ax.barh(counts.index[::-1], counts.values[::-1], color=BLUE, height=0.6)
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + 4, i, f"{v}건 ({v/n_starts*100:.1f}%)", va="center",
                color=SECONDARY_INK, fontsize=9)
    ax.set_xlim(0, counts.max() * 1.25)
    _style_ax(ax, f"5이닝 미만 조기강판 {len(early)}건의 원인 분해 (2024)", "등판 수", "")
    fig.tight_layout()
    fig.savefig(FIG / "2024_early_hook_types.png")
    plt.close(fig)


def fig_early_hook_scatter(s: pd.DataFrame, early: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150, facecolor=SURFACE)
    normal = s[~s["조기강판"]]
    ax.scatter(normal["투구수"], normal["자책"], s=20, color=MUTED, alpha=0.18,
               edgecolors="none", label="5이닝 이상")
    palette = {
        "부진형(피실점)": BLUE,
        "소진형(투구수 소모)": ORANGE,
        "극단형(부상·오프너 의심)": AQUA,
    }
    for label, color in palette.items():
        grp = early[early["유형"] == label]
        ax.scatter(grp["투구수"], grp["자책"], s=26, color=color, alpha=0.75,
                   edgecolors="none", label=label)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")
    _style_ax(ax, "조기강판은 어디에 몰려 있나 — 투구수 × 자책점 (2024)", "투구수", "자책점")
    fig.tight_layout()
    fig.savefig(FIG / "2024_early_hook_scatter.png")
    plt.close(fig)


def add_fatigue_context(s: pd.DataFrame) -> pd.DataFrame:
    """같은 투수의 직전 등판 정보(휴식일·직전 투구수·시즌 등판 순번)를 붙인다."""
    key = ["team", "선수명"]
    s = s.sort_values(key + ["date"]).copy()
    s["휴식일"] = s.groupby(key)["date"].diff().dt.days
    s["직전_투구수"] = s.groupby(key)["투구수"].shift(1)
    s["시즌_등판순번"] = s.groupby(key).cumcount() + 1
    return s


def fig_rest_days(s: pd.DataFrame) -> None:
    d = s.dropna(subset=["휴식일"]).copy()
    # 4일 미만(불펜성 등판/변칙)과 10일 초과(부상 복귀·2군)는 성격이 달라 양끝을 묶는다.
    bins = [0, 4, 5, 6, 7, 10, 999]
    labels = ["~4일", "5일", "6일", "7일", "8~10일", "11일+"]
    d["휴식_구간"] = pd.cut(d["휴식일"], bins=bins, labels=labels, right=True)
    grouped = d.groupby("휴식_구간", observed=True).agg(
        평균이닝=("이닝_소수", "mean"), 평균투구수=("투구수", "mean"), 건수=("이닝_소수", "size")
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=SURFACE)
    ax.bar(grouped.index.astype(str), grouped["평균이닝"], color=BLUE, width=0.55)
    for i, (v, n) in enumerate(zip(grouped["평균이닝"], grouped["건수"])):
        ax.text(i, v + 0.06, f"{v:.2f}", ha="center", color=SECONDARY_INK, fontsize=9)
        ax.text(i, 0.12, f"n={n}", ha="center", color=SURFACE, fontsize=8)
    _style_ax(ax, "휴식일별 선발 소화 이닝 (2024)", "직전 등판과의 간격", "평균 소화 이닝")
    fig.tight_layout()
    fig.savefig(FIG / "2024_rest_days.png")
    plt.close(fig)
    return grouped


def fig_prev_pitch_carryover(s: pd.DataFrame):
    d = s.dropna(subset=["직전_투구수"]).copy()
    bins = [0, 80, 95, 105, 999]
    labels = ["~80구", "81~95구", "96~105구", "106구+"]
    d["직전_구간"] = pd.cut(d["직전_투구수"], bins=bins, labels=labels, right=True)
    grouped = d.groupby("직전_구간", observed=True).agg(
        평균이닝=("이닝_소수", "mean"),
        평균투구수=("투구수", "mean"),
        평균자책=("자책", "mean"),
        조기강판율=("조기강판", "mean"),
        건수=("이닝_소수", "size"),
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=SURFACE)
    ax.bar(grouped.index.astype(str), grouped["조기강판율"] * 100, color=ORANGE, width=0.55)
    for i, (v, n) in enumerate(zip(grouped["조기강판율"] * 100, grouped["건수"])):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", color=SECONDARY_INK, fontsize=9)
        ax.text(i, 1.2, f"n={n}", ha="center", color=SURFACE, fontsize=8)
    _style_ax(ax, "직전 등판 투구수가 다음 등판에 남기는 영향 (2024)",
              "직전 등판 투구수", "다음 등판 조기강판율 (%)")
    fig.tight_layout()
    fig.savefig(FIG / "2024_prev_pitch_carryover.png")
    plt.close(fig)
    return grouped


def fig_season_progression(s: pd.DataFrame):
    d = s[s["시즌_등판순번"] <= 28].copy()
    d["구간"] = pd.cut(d["시즌_등판순번"], bins=[0, 7, 14, 21, 28],
                      labels=["1~7번째", "8~14번째", "15~21번째", "22~28번째"])
    grouped = d.groupby("구간", observed=True).agg(
        평균이닝=("이닝_소수", "mean"), 평균투구수=("투구수", "mean"),
        평균자책=("자책", "mean"), 건수=("이닝_소수", "size")
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=SURFACE)
    ax.plot(grouped.index.astype(str), grouped["평균이닝"], color=BLUE,
            linewidth=2, marker="o", markersize=8)
    for i, v in enumerate(grouped["평균이닝"]):
        ax.text(i, v + 0.035, f"{v:.2f}", ha="center", color=SECONDARY_INK, fontsize=9)
    _style_ax(ax, "시즌이 진행될수록 선발이 짧아지는가 (2024)", "그 투수의 시즌 내 등판 순번", "평균 소화 이닝")
    fig.tight_layout()
    fig.savefig(FIG / "2024_season_progression.png")
    plt.close(fig)
    return grouped


def fig_bf_threshold(s: pd.DataFrame) -> None:
    """상대 타자 수(BF) 분포 — 9의 배수(타순 한 바퀴) 근처에 절벽이 있는지."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    counts = s["타자"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
    for x in (9, 18, 27):
        ax.axvline(x + 0.5, color=ORANGE, linewidth=1.6, linestyle="--", alpha=0.9)
        ax.text(x + 0.8, counts.max() * 0.93, f"{x//9}바퀴 완료",
                color=ORANGE, fontsize=9, fontweight="bold")
    _style_ax(ax, "선발이 상대한 타자 수 분포 — 타순 회전 기준선 (2024)", "상대 타자 수(BF)", "선발 등판 수")
    fig.tight_layout()
    fig.savefig(FIG / "2024_bf_threshold.png")
    plt.close(fig)


def summarize(s: pd.DataFrame, early: pd.DataFrame, rest, carry, season) -> None:
    lines = ["=== 2024 선발투수 심층 분석 ===\n"]

    lines.append("[1] 조기강판 452건의 원인 분해")
    for label, n in early["유형"].value_counts().items():
        grp = early[early["유형"] == label]
        lines.append(f"  - {label}: {n}건 "
                     f"(평균 {grp['이닝_소수'].mean():.2f}이닝 / {grp['투구수'].mean():.0f}구 / 자책 {grp['자책'].mean():.1f})")
    lines.append(f"  → 전체 선발 등판 {len(s)}건 대비 조기강판 {len(early)}건 ({len(early)/len(s)*100:.1f}%)\n")

    normal = s[~s["조기강판"]]
    lines.append("[2] 투구 효율 — 조기강판 vs 정상 소화")
    lines.append(f"  - 이닝당 투구수(P/IP): 조기강판 {early['P_per_IP'].mean():.1f}구 vs 5이닝+ {normal['P_per_IP'].mean():.1f}구")
    lines.append(f"  - 타자당 투구수(P/BF): 조기강판 {early['P_per_BF'].mean():.2f}구 vs 5이닝+ {normal['P_per_BF'].mean():.2f}구\n")

    lines.append("[3] 누적 피로 — 휴식일별 평균 소화 이닝")
    for idx, row in rest.iterrows():
        lines.append(f"  - {idx}: {row['평균이닝']:.2f}이닝 / {row['평균투구수']:.1f}구 (n={int(row['건수'])})")
    lines.append("")

    lines.append("[4] 누적 피로 — 직전 등판 투구수의 이월 효과")
    for idx, row in carry.iterrows():
        lines.append(f"  - 직전 {idx}: 다음 등판 {row['평균이닝']:.2f}이닝 / 조기강판율 {row['조기강판율']*100:.1f}% (n={int(row['건수'])})")
    lines.append("")

    lines.append("[5] 시즌 누적 — 등판 순번별 추이")
    for idx, row in season.iterrows():
        lines.append(f"  - {idx}: {row['평균이닝']:.2f}이닝 / {row['평균투구수']:.1f}구 / 자책 {row['평균자책']:.2f} (n={int(row['건수'])})")
    lines.append("")

    lines.append("[6] 교체 임계값")
    bf = s["타자"]
    lines.append(f"  - 상대 타자 수 중앙값 {bf.median():.0f}명 (타순 {bf.median()/9:.2f}바퀴)")
    for cut in (18, 27):
        lines.append(f"    · {cut}명(={cut//9}바퀴) 직후 이탈 비중: "
                     f"{((bf > cut) & (bf <= cut + 3)).sum()}건 vs 직전 3명 구간 {((bf > cut - 3) & (bf <= cut)).sum()}건")
    over_100 = (s["투구수"] >= 100).mean() * 100
    lines.append(f"  - 100구 이상 던진 비율: {over_100:.1f}% / 최다 투구수 {s['투구수'].max():.0f}구")

    out = "\n".join(lines)
    path = FIG.parent / "eda_2024_deep_summary.txt"
    path.write_text(out, encoding="utf-8")
    # 윈도우 콘솔(cp949)이 못 찍는 문자가 섞여 있어 요약은 파일로만 남긴다.
    print(f"summary -> {path}")


def main() -> None:
    s = load_starters()
    early = classify_early_hook(s)
    fig_early_hook_types(early, len(s))
    fig_early_hook_scatter(s, early)

    s = add_fatigue_context(s)
    rest = fig_rest_days(s)
    carry = fig_prev_pitch_carryover(s)
    season = fig_season_progression(s)
    fig_bf_threshold(s)

    summarize(s, early, rest, carry, season)
    print(f"\n그림 6장 -> {FIG}")


if __name__ == "__main__":
    main()
