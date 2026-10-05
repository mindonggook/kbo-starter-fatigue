"""세 시즌 이상을 결합한다 — 시즌 간 이질성을 Q 통계량으로 검정한 뒤.

시즌이 둘일 때는 두 추정치 차이만 보면 됐지만, 셋부터는 '전체가 하나의 값을
가리키는가'를 봐야 한다. 표준 메타분석의 Cochran Q를 쓴다.

    Q = Σ wᵢ (estᵢ - est_pooled)²,   wᵢ = 1/seᵢ²

Q가 자유도(k-1)의 카이제곱 분포에서 크면 시즌마다 다른 값을 재고 있다는 뜻이다.
I²는 전체 분산 중 시즌 차이가 차지하는 몫으로, 크기를 직관적으로 보여준다.

동질적이면 고정효과로 합치고, 이질적이면 합치지 않고 시즌별로 남긴다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from compare_seasons import metrics

DATA = Path(__file__).resolve().parent.parent / "data"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SEASON_COLORS = {2024: "#2a78d6", 2025: "#eb6834", 2026: "#1baf7a"}
POOL_COLOR = "#4a3aa7"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

POOLABLE = [
    ("1바퀴 대비 3바퀴", "타순 1→3바퀴 피안타율"),
    ("투구수 1~25구 대비 76~100구", "투구수 1~25 → 76~100구 피안타율"),
    ("76~90구 주자 유무 강판율 차", "76~90구에서 주자 유무 강판율 차"),
    ("5일 대 6일 휴식 (이닝)", "5일 휴식의 소화 이닝 효과"),
    ("시즌 22~28등판 대 1~21 (이닝)", "시즌 막바지 등판의 이닝 효과"),
    ("5일 휴식 → 피안타율", "5일 휴식 → 피안타율"),
    ("시즌 누적 +100구 → 피안타율", "시즌 누적 +100구 → 피안타율"),
    ("불펜 +10구 → 선발 투구수", "직전 3경기 불펜 +10구 → 선발 투구수"),
]

# 신뢰구간 없이 값만 비교하는 서술 지표
DESCRIPTIVE = [
    ("리그 타율(파싱 검산)", "리그 타율"),
    ("조기강판 비율", "조기강판 비율"),
    ("조기강판 중 부진형", "조기강판 중 부진형 비중"),
    ("100구 이상 비율", "100구 이상 비율"),
    ("강판 시 주자 있던 비율", "강판 시 주자 있던 비율"),
    ("강판 시 지고 있던 비율", "강판 시 지고 있던 비율"),
    ("회전·투구수 상관계수", "회전·투구수 상관계수"),
    ("타자순번이 설명하는 투구수 R²", "타자순번이 설명하는 투구수 R²"),
    ("타순 3바퀴째 피안타율", "타순 3바퀴째 피안타율"),
]


def se_from_ci(d: dict) -> float:
    return (d["hi"] - d["lo"]) / (2 * 1.96)


def meta(ests: list[float], ses: list[float]) -> dict:
    """고정효과 메타분석 + Cochran Q 이질성 검정."""
    w = np.array([1 / s ** 2 for s in ses])
    e = np.array(ests)
    pooled = float((w * e).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))

    q = float((w * (e - pooled) ** 2).sum())
    df = len(e) - 1
    p = float(1 - stats.chi2.cdf(q, df)) if df > 0 else 1.0
    i2 = float(max(0.0, (q - df) / q * 100)) if q > 0 else 0.0

    lo, hi = pooled - 1.96 * se, pooled + 1.96 * se
    return {"pooled": pooled, "lo": lo, "hi": hi,
            "유의": bool((lo > 0) == (hi > 0)),
            "Q": q, "df": df, "p": p, "I2": i2,
            "동질적": bool(p >= 0.05)}


def fig_forest(rows: list[dict], seasons: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(10, 1.35 * len(rows) * (len(seasons) + 1) / 3 + 2.2),
                           dpi=150, facecolor=SURFACE)
    ypos, ylabels = [], []
    y = 0.0
    for r in reversed(rows):
        if r["동질적"]:
            ax.plot([r["lo"] / r["scale"], r["hi"] / r["scale"]], [y, y],
                    color=POOL_COLOR, linewidth=2.4, solid_capstyle="round")
            ax.plot(r["pooled"] / r["scale"], y, "D", color=POOL_COLOR, markersize=10)
            ax.text(r["hi"] / r["scale"] + 0.08, y, f"{r['pooled']:+.4f}".rstrip("0").rstrip("."),
                    va="center", ha="left", color=SECONDARY_INK, fontsize=8.5)
            ypos.append(y)
            ylabels.append("합산")
            y += 1
        for season in reversed(seasons):
            if season not in r["by_season"]:
                continue
            d = r["by_season"][season]
            color = SEASON_COLORS.get(season, MUTED)
            ax.plot([d["lo"] / r["scale"], d["hi"] / r["scale"]], [y, y],
                    color=color, linewidth=2.2, solid_capstyle="round")
            ax.plot(d["est"] / r["scale"], y, "o", color=color, markersize=8)
            text = f"{d['est']:+.4f}".rstrip("0").rstrip(".")
            if d["est"] >= 0:
                ax.text(d["hi"] / r["scale"] + 0.08, y, text, va="center", ha="left",
                        color=SECONDARY_INK, fontsize=8.5)
            else:
                ax.text(d["lo"] / r["scale"] - 0.08, y, text, va="center", ha="right",
                        color=SECONDARY_INK, fontsize=8.5)
            ypos.append(y)
            ylabels.append(str(season))
            y += 1
        tag = "" if r["동질적"] else f"  (I²={r['I2']:.0f}%, 합치지 않음)"
        ax.text(-4.75, y - 0.15, r["label"] + tag, va="center", ha="left",
                color=INK, fontsize=9.5, fontweight="bold")
        y += 1.4

    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlim(-4.9, 4.9)
    ax.set_ylim(-1, y)
    ax.set_xlabel("각 지표의 최대 추정치를 1로 놓은 상대 스케일", color=SECONDARY_INK, fontsize=10)
    ax.set_title("세 시즌을 나란히, 그리고 합쳤을 때 (95% 신뢰구간)",
                 color=INK, fontsize=13, loc="left", pad=14)
    handles = [plt.Line2D([], [], color=SEASON_COLORS[s], marker="o", linestyle="", label=str(s))
               for s in seasons if s in SEASON_COLORS]
    handles.append(plt.Line2D([], [], color=POOL_COLOR, marker="D", linestyle="",
                              label="합산(동질적일 때만)"))
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=SECONDARY_INK,
              loc="lower right")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / "pooled_forest_multi.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()
    seasons = a.seasons

    ms = {s: metrics(s) for s in seasons}

    rows = []
    for key, label in POOLABLE:
        by_season = {s: ms[s][key] for s in seasons if key in ms[s] and "lo" in ms[s][key]}
        if len(by_season) < 2:
            continue
        ests = [by_season[s]["est"] for s in by_season]
        ses = [se_from_ci(by_season[s]) for s in by_season]
        res = meta(ests, ses)
        rows.append({"key": key, "label": label, "by_season": by_season,
                     "scale": max(max(abs(e) for e in ests), 1e-9), **res})

    lines = [f"=== {' + '.join(str(s) for s in seasons)} 결합 분석 ===\n",
             "Cochran Q로 시즌 간 이질성을 검정하고, 동질적일 때만 고정효과로 합쳤다.\n"]

    for r in rows:
        lines.append(f"[{r['label']}]")
        for s, d in r["by_season"].items():
            lines.append(f"  {s}: {d['est']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
        lines.append(f"  이질성 Q={r['Q']:.2f} (df={r['df']}, p={r['p']:.3f}), I²={r['I2']:.0f}%")
        if r["동질적"]:
            sig = "유의함" if r["유의"] else "유의하지 않음"
            lines.append(f"  합산: {r['pooled']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {sig}")
        else:
            lines.append("  → 시즌마다 다름, 합치지 않음")
        lines.append("")

    lines.append("[서술 지표 — 시즌별 값]")
    for key, label in DESCRIPTIVE:
        vals = [f"{s} {ms[s][key]['est']:.4f}" for s in seasons if key in ms[s]]
        if vals:
            lines.append(f"  {label}: " + " · ".join(vals))

    n_het = sum(1 for r in rows if not r["동질적"])
    lines.append(f"\n요약: 검정 지표 {len(rows)}개 중 동질적 {len(rows)-n_het}개 · 이질적 {n_het}개")

    out = "\n".join(lines)
    path = DATA.parent / f"pooled_{'_'.join(str(s) for s in seasons)}.txt"
    path.write_text(out, encoding="utf-8")
    fig_forest(rows, seasons)
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
