"""빠져 있던 변수 — 불펜 사정.

지금까지의 교체 타이밍 분석은 '마운드 위의 투수'만 봤다. 그런데 교체는 내릴
사람이 있어야 가능하다. 전날 불펜을 크게 소모한 팀은 오늘 선발을 더 끌 수밖에
없고, 그렇다면 '선발이 길게 갔다'가 피로가 적었다는 뜻이 아니라 '뒤가 비었다'는
뜻일 수 있다.

박스스코어에 구원 등판 기록이 전부 있으므로 팀·날짜별 불펜 소모를 직접 계산해
직전 1경기·3경기 누적으로 만들어 붙인다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{SEASON}.parquet")
    box["date"] = pd.to_datetime(box["date"], format="%Y%m%d")
    box["투구수"] = box["투구수"].astype(float)

    # 팀·경기별 불펜 소모량
    pen = (box[~box["is_starter"]]
           .groupby(["team", "date"])["투구수"].sum()
           .reset_index().rename(columns={"투구수": "불펜투구수"}))
    pen = pen.sort_values(["team", "date"])
    pen["직전1경기"] = pen.groupby("team")["불펜투구수"].shift(1)
    pen["직전3경기"] = (pen.groupby("team")["불펜투구수"]
                      .rolling(3, min_periods=3).sum()
                      .reset_index(level=0, drop=True).groupby(pen["team"]).shift(1))

    s = box[box["is_starter"]].copy()
    s = s.merge(pen[["team", "date", "직전1경기", "직전3경기"]], on=["team", "date"], how="left")

    key = ["team", "선수명"]
    s = s.sort_values(key + ["date"])
    s["휴식일"] = s.groupby(key)["date"].diff().dt.days
    s["등판수"] = s.groupby(key)["투구수"].transform("size")
    s = s[s["등판수"] >= 10]
    return s.dropna(subset=["직전1경기", "직전3경기"])


def fit(s: pd.DataFrame, outcome: str, exposure: str) -> dict:
    d = s.copy()
    design = pd.DataFrame(index=d.index)
    design["불펜_10구당"] = d[exposure] / 10.0
    design["휴식_6일"] = (d["휴식일"] == 6).astype(float)
    design = pd.concat([design,
                        pd.get_dummies(d["선수명"], prefix="P", drop_first=True).astype(float)],
                       axis=1)
    design = sm.add_constant(design)
    res = sm.OLS(d[outcome].astype(float), design).fit(cov_type="HC1")
    ci = res.conf_int().loc["불펜_10구당"]
    return {"outcome": outcome, "exposure": exposure, "n": int(len(d)),
            "coef": res.params["불펜_10구당"], "lo": ci[0], "hi": ci[1],
            "sig": (ci[0] > 0) == (ci[1] > 0)}


def fig_bullpen(s: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    d = s.copy()
    d["불펜_구간"] = pd.qcut(d["직전3경기"], 4,
                          labels=["적게 씀\n(하위 25%)", "보통", "많이 씀", "매우 많이 씀\n(상위 25%)"])
    # 투수 실력이 섞이지 않도록 자기 평균 대비 편차로 본다.
    d["이닝_편차"] = d["이닝_소수"] - d.groupby(["team", "선수명"])["이닝_소수"].transform("mean")
    d["투구수_편차"] = d["투구수"] - d.groupby(["team", "선수명"])["투구수"].transform("mean")
    table = d.groupby("불펜_구간", observed=True).agg(
        이닝편차=("이닝_편차", "mean"), 투구수편차=("투구수_편차", "mean"),
        불펜중앙값=("직전3경기", "median"), 건수=("이닝_편차", "size"))

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    vals = table["투구수편차"]
    colors = [BLUE if v >= 0 else ORANGE for v in vals]
    ax.bar(range(len(table)), vals, color=colors, width=0.55)
    ax.axhline(0, color=MUTED, linewidth=1.2)
    span = max(abs(vals.min()), abs(vals.max()))
    ax.set_ylim(-span * 1.9, span * 1.9)
    for i, (v, n, med) in enumerate(zip(vals, table["건수"], table["불펜중앙값"])):
        ax.text(i, v + (span * 0.12 if v >= 0 else -span * 0.12), f"{v:+.1f}구",
                ha="center", va="bottom" if v >= 0 else "top",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
        ax.text(i, -span * 1.72, f"직전3경기 {med:.0f}구 · n={n}", ha="center",
                color=MUTED, fontsize=8)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table.index.astype(str), fontsize=9)
    _style_ax(ax, f"불펜을 많이 쓴 뒤에는 선발을 더 끄는가 ({SEASON})",
              "직전 3경기 불펜 투구수", "선발 투구수 편차 (자기 평균 대비)")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_bullpen_effect.png")
    plt.close(fig)
    return table


def summarize(s: pd.DataFrame, table: pd.DataFrame, results: list[dict]) -> None:
    lines = [f"=== {SEASON} 불펜 사정이 선발 교체에 주는 영향 "
             f"(선발 등판 {len(s):,}건 / 투수 {s['선수명'].nunique()}명) ===\n"]

    lines.append("[1] 직전 3경기 불펜 소모량 구간별 (투수 자기 평균 대비 편차)")
    for idx, row in table.iterrows():
        label = str(idx).replace("\n", " ")
        lines.append(f"  - {label} (중앙값 {row['불펜중앙값']:.0f}구): "
                     f"이닝 {row['이닝편차']:+.3f} · 투구수 {row['투구수편차']:+.1f}구 "
                     f"(n={int(row['건수'])})")
    lines.append("")

    lines.append("[2] 회귀 추정 (투수 고정효과 · 휴식일 통제, 이분산 강건 표준오차)")
    for r in results:
        out = {"이닝_소수": "소화 이닝", "투구수": "투구수"}[r["outcome"]]
        exp = {"직전1경기": "직전 1경기", "직전3경기": "직전 3경기"}[r["exposure"]]
        mark = "유의함" if r["sig"] else "유의하지 않음"
        lines.append(f"  - {exp} 불펜 +10구 → 선발 {out}: "
                     f"{r['coef']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {mark}")

    out = "\n".join(lines)
    path = FIG.parent / f"eda_{SEASON}_bullpen_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


def main() -> None:
    s = load()
    results = [fit(s, o, e) for o in ("이닝_소수", "투구수") for e in ("직전1경기", "직전3경기")]
    table = fig_bullpen(s, results)
    summarize(s, table, results)
    print(f"그림 1장 -> {FIG}")


if __name__ == "__main__":
    main()
