"""같은 투수 안에서 비교한 피로 효과 (within-pitcher / 고정효과식 분석).

eda_2024_deep.py의 [4][5]는 "직전에 많이 던진 투수가 다음 등판에 더 오래 간다",
"시즌 후반일수록 더 오래 간다"는 상식과 반대되는 결과를 냈다. 원인은 투수 간
실력 차이다 — 에이스는 늘 많이 던지고 늘 길게 가고, 5선발/대체선발은 늘 적게
던지고 늘 짧게 간다. 그래서 투수를 가로질러 평균을 내면 실력 차이가 피로 효과를
덮어버린다.

여기서는 각 등판을 '그 투수 자신의 시즌 평균 대비 편차'로 바꿔서(디미닝),
투수 간 실력 차이를 제거한 뒤 같은 축들을 다시 본다.
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

MIN_STARTS = 10  # 자기 평균이 의미를 가지려면 등판 표본이 어느 정도 필요


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


def _zero_line(ax):
    ax.axhline(0, color=MUTED, linewidth=1.2)


def _label_deviation_bars(ax, values, counts):
    """0을 기준으로 위아래로 뻗는 막대에 값과 표본 수를 붙인다.

    값 라벨을 막대 바깥에 두면 음수 막대의 라벨이 축 밖으로 나가 눈금과 겹치므로,
    먼저 위아래 여백을 확보한 뒤 라벨을 얹는다.
    """
    span = max(abs(min(values)), abs(max(values)))
    ax.set_ylim(-span * 1.9, span * 1.9)
    pad = span * 0.12
    for i, (v, n) in enumerate(zip(values, counts)):
        ax.text(i, v + (pad if v >= 0 else -pad), f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", color=SECONDARY_INK, fontsize=9)
        ax.text(i, -span * 1.72, f"n={n}", ha="center", color=MUTED, fontsize=8)


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "kbo_pitcher_appearances_2024.parquet")
    s = df[df["is_starter"]].copy()
    s["date"] = pd.to_datetime(s["date"], format="%Y%m%d")
    for col in ("투구수", "타자", "자책", "실점"):
        s[col] = s[col].astype(float)

    key = ["team", "선수명"]
    s = s.sort_values(key + ["date"]).copy()
    s["휴식일"] = s.groupby(key)["date"].diff().dt.days
    s["직전_투구수"] = s.groupby(key)["투구수"].shift(1)
    s["시즌_등판순번"] = s.groupby(key).cumcount() + 1
    s["등판수"] = s.groupby(key)["투구수"].transform("size")

    regulars = s[s["등판수"] >= MIN_STARTS].copy()
    # 각 투수 자신의 시즌 평균을 뺀 편차 — 투수 간 실력 차이가 사라진다.
    for col in ("이닝_소수", "투구수", "자책"):
        regulars[f"{col}_편차"] = regulars[col] - regulars.groupby(key)[col].transform("mean")
    regulars["직전_투구수_편차"] = (
        regulars["직전_투구수"] - regulars.groupby(key)["투구수"].transform("mean")
    )
    return regulars


def fig_prev_pitch_within(s: pd.DataFrame):
    d = s.dropna(subset=["직전_투구수_편차"]).copy()
    bins = [-999, -15, -5, 5, 15, 999]
    labels = ["자기평균\n-15구 이하", "-15~-5구", "평균 수준\n(±5구)", "+5~+15구", "+15구 이상"]
    d["직전_편차_구간"] = pd.cut(d["직전_투구수_편차"], bins=bins, labels=labels)
    grouped = d.groupby("직전_편차_구간", observed=True).agg(
        이닝편차=("이닝_소수_편차", "mean"),
        투구수편차=("투구수_편차", "mean"),
        자책편차=("자책_편차", "mean"),
        건수=("이닝_소수_편차", "size"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=SURFACE)
    colors = [BLUE if v >= 0 else ORANGE for v in grouped["이닝편차"]]
    ax.bar(grouped.index.astype(str), grouped["이닝편차"], color=colors, width=0.55)
    _zero_line(ax)
    _label_deviation_bars(ax, list(grouped["이닝편차"]), list(grouped["건수"]))
    _style_ax(ax, "직전 등판에서 평소보다 많이 던졌다면? (같은 투수 안에서 비교, 2024)",
              "직전 등판 투구수 — 그 투수 자신의 평균 대비", "이번 등판 소화 이닝 편차")
    fig.tight_layout()
    fig.savefig(FIG / "2024_within_prev_pitch.png")
    plt.close(fig)
    return grouped


def fig_rest_within(s: pd.DataFrame):
    d = s.dropna(subset=["휴식일"]).copy()
    d = d[d["휴식일"] <= 10]  # 11일 이상은 부상 복귀·2군행이라 피로와 다른 축
    bins = [0, 4, 5, 6, 7, 10]
    labels = ["~4일", "5일", "6일", "7일", "8~10일"]
    d["휴식_구간"] = pd.cut(d["휴식일"], bins=bins, labels=labels, right=True)
    grouped = d.groupby("휴식_구간", observed=True).agg(
        이닝편차=("이닝_소수_편차", "mean"),
        투구수편차=("투구수_편차", "mean"),
        자책편차=("자책_편차", "mean"),
        건수=("이닝_소수_편차", "size"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=SURFACE)
    colors = [BLUE if v >= 0 else ORANGE for v in grouped["이닝편차"]]
    ax.bar(grouped.index.astype(str), grouped["이닝편차"], color=colors, width=0.55)
    _zero_line(ax)
    _label_deviation_bars(ax, list(grouped["이닝편차"]), list(grouped["건수"]))
    _style_ax(ax, "휴식일이 짧으면 그 투수는 평소보다 짧게 던지나 (2024)",
              "직전 등판과의 간격", "소화 이닝 편차 (자기 평균 대비)")
    fig.tight_layout()
    fig.savefig(FIG / "2024_within_rest_days.png")
    plt.close(fig)
    return grouped


def fig_season_within(s: pd.DataFrame):
    d = s[s["시즌_등판순번"] <= 28].copy()
    d["구간"] = pd.cut(d["시즌_등판순번"], bins=[0, 7, 14, 21, 28],
                      labels=["1~7번째", "8~14번째", "15~21번째", "22~28번째"])
    grouped = d.groupby("구간", observed=True).agg(
        이닝편차=("이닝_소수_편차", "mean"),
        투구수편차=("투구수_편차", "mean"),
        자책편차=("자책_편차", "mean"),
        건수=("이닝_소수_편차", "size"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=SURFACE)
    ax.plot(grouped.index.astype(str), grouped["이닝편차"], color=BLUE,
            linewidth=2, marker="o", markersize=8)
    _zero_line(ax)
    span = max(abs(grouped["이닝편차"].min()), abs(grouped["이닝편차"].max()))
    ax.set_ylim(-span * 1.8, span * 1.8)
    for i, (v, n) in enumerate(zip(grouped["이닝편차"], grouped["건수"])):
        ax.text(i, v + span * 0.14, f"{v:+.3f}", ha="center", color=SECONDARY_INK, fontsize=9)
        ax.text(i, -span * 1.62, f"n={int(n)}", ha="center", color=MUTED, fontsize=8)
    _style_ax(ax, "시즌이 쌓이면 같은 투수가 짧아지는가 (2024)",
              "그 투수의 시즌 내 등판 순번", "소화 이닝 편차 (자기 평균 대비)")
    fig.tight_layout()
    fig.savefig(FIG / "2024_within_season.png")
    plt.close(fig)
    return grouped


def welch_ci(a: pd.Series, b: pd.Series) -> tuple[float, float, float, bool]:
    """두 그룹 평균 차이와 95% 신뢰구간(Welch). 효과 크기가 작아 노이즈와 구별이 필요하다."""
    a, b = np.asarray(a.dropna()), np.asarray(b.dropna())
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return diff, lo, hi, (lo > 0) == (hi > 0)


def significance_lines(s: pd.DataFrame) -> list[str]:
    d = s.dropna(subset=["휴식일"])
    p = s.dropna(subset=["직전_투구수_편차"])
    tests = [
        ("5일 휴식 vs 6일 휴식",
         d[d["휴식일"] == 5]["이닝_소수_편차"], d[d["휴식일"] == 6]["이닝_소수_편차"]),
        ("시즌 22~28번째 등판 vs 1~21번째",
         s[s["시즌_등판순번"].between(22, 28)]["이닝_소수_편차"],
         s[s["시즌_등판순번"].between(1, 21)]["이닝_소수_편차"]),
        ("직전 등판 자기평균 +15구 초과 vs 나머지",
         p[p["직전_투구수_편차"] >= 15]["이닝_소수_편차"],
         p[p["직전_투구수_편차"] < 15]["이닝_소수_편차"]),
    ]
    lines = ["[D] 효과가 노이즈와 구별되는가 (소화 이닝 편차 차이, 95% 신뢰구간)"]
    for name, a, b in tests:
        diff, lo, hi, sig = welch_ci(a, b)
        verdict = "유의함" if sig else "유의하지 않음(0을 포함)"
        lines.append(f"  - {name}: {diff:+.3f}이닝 [{lo:+.3f}, {hi:+.3f}] → {verdict}")
    return lines


def summarize(s: pd.DataFrame, prev, rest, season) -> None:
    n_pitchers = s.groupby(["team", "선수명"]).ngroups
    lines = [f"=== 2024 같은 투수 안에서 본 피로 효과 (등판 {MIN_STARTS}회 이상 투수 {n_pitchers}명 / {len(s)}등판) ===\n"]

    lines.append("[A] 직전 등판 투구수의 이월 효과 (자기 평균 대비 편차)")
    for idx, row in prev.iterrows():
        label = str(idx).replace("\n", " ")
        lines.append(f"  - 직전 {label}: 이번 등판 이닝 {row['이닝편차']:+.3f} / "
                     f"투구수 {row['투구수편차']:+.1f}구 / 자책 {row['자책편차']:+.2f} (n={int(row['건수'])})")
    lines.append("")

    lines.append("[B] 휴식일 효과 (자기 평균 대비 편차)")
    for idx, row in rest.iterrows():
        lines.append(f"  - {idx}: 이닝 {row['이닝편차']:+.3f} / 투구수 {row['투구수편차']:+.1f}구 / "
                     f"자책 {row['자책편차']:+.2f} (n={int(row['건수'])})")
    lines.append("")

    lines.append("[C] 시즌 누적 효과 (자기 평균 대비 편차)")
    for idx, row in season.iterrows():
        lines.append(f"  - {idx}: 이닝 {row['이닝편차']:+.3f} / 투구수 {row['투구수편차']:+.1f}구 / "
                     f"자책 {row['자책편차']:+.2f} (n={int(row['건수'])})")
    lines.append("")
    lines.extend(significance_lines(s))

    out = "\n".join(lines)
    path = FIG.parent / "eda_2024_within_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


def main() -> None:
    s = load()
    prev = fig_prev_pitch_within(s)
    rest = fig_rest_within(s)
    season = fig_season_within(s)
    summarize(s, prev, rest, season)
    print(f"그림 3장 -> {FIG}")


if __name__ == "__main__":
    main()
