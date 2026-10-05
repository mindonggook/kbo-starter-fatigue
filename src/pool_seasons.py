"""두 시즌 추정치를 합치기 전에, 합쳐도 되는지부터 검정한다.

2024와 2025의 추정치가 엇갈릴 때 그냥 평균 내면 두 가지를 뭉갠다 — 우연한
흔들림과 진짜 시즌 차이. 그래서 순서를 지킨다.

  1. 이질성 검정: 두 추정치 차이가 각자의 불확실성으로 설명되는가
     z = (est1 - est2) / sqrt(se1^2 + se2^2)
  2. 동질적이면(|z| < 1.96) 역분산 가중으로 합쳐 더 좁은 구간을 얻는다
  3. 이질적이면 합치지 않고 '시즌마다 다르다'로 보고한다
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_seasons import metrics

DATA = Path(__file__).resolve().parent.parent / "data"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# 신뢰구간이 있는 지표만 합칠 수 있다.
POOLABLE = [
    ("1바퀴 대비 3바퀴", "타순 1→3바퀴 피안타율", 1),
    ("투구수 1~25구 대비 76~100구", "투구수 1~25 → 76~100구 피안타율", 1),
    ("76~90구 주자 유무 강판율 차", "76~90구에서 주자 유무 강판율 차", 1),
    ("5일 대 6일 휴식 (이닝)", "5일 휴식의 소화 이닝 효과", 1),
    ("시즌 22~28등판 대 1~21 (이닝)", "시즌 막바지 등판의 이닝 효과", 1),
    ("5일 휴식 → 피안타율", "5일 휴식 → 피안타율", 1),
    ("시즌 누적 +100구 → 피안타율", "시즌 누적 +100구 → 피안타율", 1),
    ("불펜 +10구 → 선발 투구수", "직전 3경기 불펜 +10구 → 선발 투구수", 1),
]


def se_from_ci(d: dict) -> float:
    return (d["hi"] - d["lo"]) / (2 * 1.96)


def combine(a: dict, b: dict) -> dict:
    """이질성 검정 후, 동질적이면 역분산 가중 결합."""
    se_a, se_b = se_from_ci(a), se_from_ci(b)
    z = (a["est"] - b["est"]) / np.sqrt(se_a ** 2 + se_b ** 2)
    homogeneous = abs(z) < 1.96

    wa, wb = 1 / se_a ** 2, 1 / se_b ** 2
    est = (wa * a["est"] + wb * b["est"]) / (wa + wb)
    se = np.sqrt(1 / (wa + wb))
    return {
        "z_이질성": float(z), "동질적": bool(homogeneous),
        "합산_est": float(est), "합산_lo": float(est - 1.96 * se),
        "합산_hi": float(est + 1.96 * se),
        "합산_유의": bool((est - 1.96 * se > 0) == (est + 1.96 * se > 0)),
    }


def fig_forest(rows: list[dict]) -> None:
    """숲그림 — 지표마다 2024·2025·합산을 세로로 쌓는다."""
    fig, ax = plt.subplots(figsize=(9.6, 9.2), dpi=150, facecolor=SURFACE)

    ypos, ylabels = [], []
    y = len(rows) * 4
    for r in rows:
        for label, key, color in (("2024", "a", BLUE), ("2025", "b", ORANGE),
                                  ("합산", "pool", VIOLET)):
            if key == "pool" and not r["동질적"]:
                y -= 1
                continue
            est, lo, hi = r[f"{key}_est"], r[f"{key}_lo"], r[f"{key}_hi"]
            # 단위가 제각각이라 2024 추정치 크기로 나눠 상대 스케일로 맞춘다.
            scale = r["scale"]
            ax.plot([lo / scale, hi / scale], [y, y], color=color, linewidth=2.2,
                    solid_capstyle="round")
            marker = "D" if key == "pool" else "o"
            ax.plot(est / scale, y, marker, color=color,
                    markersize=10 if key == "pool" else 8)
            # 값 라벨은 구간이 뻗은 반대쪽에 둔다 — 음수 지표에서 선과 겹치지 않도록.
            text = f"{est:+.4f}".rstrip("0").rstrip(".")
            if est >= 0:
                ax.text(hi / scale + 0.08, y, text, va="center", ha="left",
                        color=SECONDARY_INK, fontsize=8.5)
            else:
                ax.text(lo / scale - 0.08, y, text, va="center", ha="right",
                        color=SECONDARY_INK, fontsize=8.5)
            ypos.append(y)
            ylabels.append(label)
            y -= 1
        ax.text(-4.7, y + 2.0, r["label"], va="center", ha="left",
                color=INK, fontsize=10, fontweight="bold")
        y -= 1

    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlim(-4.8, 4.8)
    ax.set_xlabel("2024 추정치 크기를 1로 놓은 상대 스케일", color=SECONDARY_INK, fontsize=10)
    ax.set_title("두 시즌을 나란히, 그리고 합쳤을 때 (95% 신뢰구간)",
                 color=INK, fontsize=13, loc="left", pad=14)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", linestyle="", label="2024"),
               plt.Line2D([], [], color=ORANGE, marker="o", linestyle="", label="2025"),
               plt.Line2D([], [], color=VIOLET, marker="D", linestyle="", label="합산(동질적일 때만)")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=SECONDARY_INK,
              loc="lower right")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / "pooled_forest.png")
    plt.close(fig)


def main() -> None:
    m24, m25 = metrics(2024), metrics(2025)

    rows = []
    for key, label, _ in POOLABLE:
        if key not in m24 or key not in m25:
            continue
        a, b = m24[key], m25[key]
        if "lo" not in a or "lo" not in b:
            continue
        c = combine(a, b)
        scale = max(abs(a["est"]), abs(b["est"]), 1e-6)
        rows.append({
            "key": key, "label": label, "scale": scale,
            "a_est": a["est"], "a_lo": a["lo"], "a_hi": a["hi"],
            "b_est": b["est"], "b_lo": b["lo"], "b_hi": b["hi"],
            "pool_est": c["합산_est"], "pool_lo": c["합산_lo"], "pool_hi": c["합산_hi"],
            **c,
        })

    lines = ["=== 2024 + 2025 결합 분석 ===\n",
             "각 지표마다 (1) 두 시즌이 같은 것을 재고 있는지 검정하고,",
             "(2) 같다고 판단될 때만 역분산 가중으로 합쳤다.\n"]

    for r in rows:
        lines.append(f"[{r['label']}]")
        lines.append(f"  2024: {r['a_est']:+.4f} [{r['a_lo']:+.4f}, {r['a_hi']:+.4f}]")
        lines.append(f"  2025: {r['b_est']:+.4f} [{r['b_lo']:+.4f}, {r['b_hi']:+.4f}]")
        het = "동질적" if r["동질적"] else "이질적 — 합치지 않음"
        lines.append(f"  이질성 z = {r['z_이질성']:+.2f} → {het}")
        if r["동질적"]:
            sig = "유의함" if r["합산_유의"] else "유의하지 않음"
            lines.append(f"  합산: {r['합산_est']:+.4f} "
                         f"[{r['합산_lo']:+.4f}, {r['합산_hi']:+.4f}] {sig}")
        lines.append("")

    n_het = sum(1 for r in rows if not r["동질적"])
    lines.append(f"요약: {len(rows)}개 지표 중 동질적 {len(rows)-n_het}개 · 이질적 {n_het}개")

    out = "\n".join(lines)
    path = DATA.parent / "pooled_2024_2025.txt"
    path.write_text(out, encoding="utf-8")
    fig_forest(rows)
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
