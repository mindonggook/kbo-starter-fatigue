"""한계 2 — 강판 시점의 점수차·주자 상황.

'피로해서 내렸나, 져 있어서 내렸나'를 가르려면 교체 순간의 경기 상황이 필요하다.
문자중계에서 복원한 타석별 상태로 두 가지를 본다.

  1. 선발이 내려온 순간의 상황 분포 (점수차 / 주자 / 이닝 경계 여부)
  2. 피로 수준을 고정했을 때 상황이 교체 확률을 얼마나 움직이는가

2번이 핵심이다. 같은 투구수 구간에 있는 선발들만 모아놓고, 점수차와 주자 유무에
따라 '이 타석이 마지막이었을 확률'이 어떻게 달라지는지 본다.
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
    pa = pd.read_parquet(DATA / f"kbo_pa_state_{SEASON}.parquet")
    chk = pd.read_parquet(DATA / f"kbo_state_check_{SEASON}.parquet")

    # 점수는 720경기 모두 검증됐지만 주자·아웃까지 쓰려면 정합한 경기만 골라야 한다.
    ok = chk[chk["score_ok"] & chk["outs_ok"] & chk["runner_ok"] & chk["starter_ok"]]
    pa = pa[pa["game_id"].isin(set(ok["game_id"])) & pa["is_starter"]].copy()

    pa = pa.sort_values(["game_id", "pitcher", "inning"]).reset_index(drop=True)
    # 선발의 마지막 타석 = 그 등판이 끝난 지점
    last = pa.groupby(["game_id", "pitcher"]).tail(1).index
    pa["마지막타석"] = False
    pa.loc[last, "마지막타석"] = True

    pa["점수상황"] = pd.cut(pa["점수차_투수팀기준"], [-99, -4, -1, 0, 3, 99],
                        labels=["4점 이상 뒤짐", "1~3점 뒤짐", "동점", "1~3점 앞섬", "4점 이상 앞섬"])
    pa["주자상황"] = np.where(pa["주자수"] > 0, "주자 있음", "주자 없음")
    return pa


def fig_hook_context(pa: pd.DataFrame) -> pd.DataFrame:
    """선발이 내려온 순간의 점수 상황 분포."""
    hooks = pa[pa["마지막타석"]]
    counts = hooks["점수상황"].value_counts().reindex(
        ["4점 이상 뒤짐", "1~3점 뒤짐", "동점", "1~3점 앞섬", "4점 이상 앞섬"])

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    colors = [ORANGE, ORANGE, MUTED, BLUE, BLUE]
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.6)
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v + total * 0.012, f"{v}건\n{v/total*100:.1f}%", ha="center",
                color=SECONDARY_INK, fontsize=9)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), fontsize=9)
    ax.set_ylim(0, counts.max() * 1.24)
    _style_ax(ax, f"선발이 내려온 순간의 점수 상황 ({SEASON})", "투수 팀 기준 점수차", "강판 건수")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_hook_context.png")
    plt.close(fig)
    return counts


def fig_hook_rate_by_context(pa: pd.DataFrame) -> pd.DataFrame:
    """피로 수준(투구수)을 묶어놓고, 상황별로 '이 타석이 마지막일 확률'을 본다."""
    d = pa[pa["투수_누적투구수"].between(61, 105)].copy()
    bins = [60, 75, 90, 105]
    labels = ["61~75구", "76~90구", "91~105구"]
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], bins=bins, labels=labels)

    table = (d.groupby(["투구수_구간", "점수상황"], observed=True)["마지막타석"]
               .agg(["mean", "size"]).rename(columns={"mean": "강판율", "size": "타석"}))
    table = table[table["타석"] >= 80]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150, facecolor=SURFACE)
    situations = ["4점 이상 뒤짐", "1~3점 뒤짐", "동점", "1~3점 앞섬", "4점 이상 앞섬"]
    colors = {"61~75구": BLUE, "76~90구": ORANGE, "91~105구": AQUA}
    width = 0.26
    x = np.arange(len(situations))
    for k, band in enumerate(labels):
        vals, pos = [], []
        for i, sit in enumerate(situations):
            if (band, sit) in table.index:
                vals.append(table.loc[(band, sit), "강판율"] * 100)
                pos.append(i + (k - 1) * width)
        ax.bar(pos, vals, width=width, color=colors[band], label=band)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.7, f"{v:.0f}", ha="center", color=SECONDARY_INK, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(situations, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, title="누적 투구수",
              title_fontsize=9, loc="upper right")
    _style_ax(ax, f"같은 투구수라도 상황에 따라 교체가 달라지는가 ({SEASON})",
              "투수 팀 기준 점수차", "이 타석이 마지막일 확률 (%)")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_hook_rate_context.png")
    plt.close(fig)
    return table


def fig_runner_effect(pa: pd.DataFrame) -> pd.DataFrame:
    """주자 유무가 교체를 앞당기는가 — 투구수 구간별로."""
    d = pa[pa["투수_누적투구수"].between(61, 105)].copy()
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], [60, 75, 90, 105],
                          labels=["61~75구", "76~90구", "91~105구"])
    table = (d.groupby(["투구수_구간", "주자상황"], observed=True)["마지막타석"]
               .agg(["mean", "size"]).rename(columns={"mean": "강판율", "size": "타석"}))

    fig, ax = plt.subplots(figsize=(7.8, 4.5), dpi=150, facecolor=SURFACE)
    bands = ["61~75구", "76~90구", "91~105구"]
    width = 0.34
    x = np.arange(len(bands))
    for k, (sit, color) in enumerate((("주자 없음", BLUE), ("주자 있음", ORANGE))):
        vals = [table.loc[(b, sit), "강판율"] * 100 for b in bands]
        pos = x + (k - 0.5) * width
        ax.bar(pos, vals, width=width, color=color, label=sit)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.7, f"{v:.1f}%", ha="center", color=SECONDARY_INK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")
    _style_ax(ax, f"주자를 두고 있으면 더 빨리 내려가는가 ({SEASON})",
              "그 타석 시점의 누적 투구수", "이 타석이 마지막일 확률 (%)")
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_hook_rate_runners.png")
    plt.close(fig)
    return table


def prop_ci(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return diff, lo, hi, (lo > 0) == (hi > 0)


def summarize(pa, ctx, rate_ctx, runner) -> None:
    hooks = pa[pa["마지막타석"]]
    lines = [f"=== {SEASON} 강판 시점의 점수차·주자 상황 "
             f"(검증 통과 {pa['game_id'].nunique()}경기 / 선발 강판 {len(hooks)}건) ===\n"]

    lines.append("[1] 선발이 내려온 순간의 점수 상황")
    total = ctx.sum()
    for idx, v in ctx.items():
        lines.append(f"  - {idx}: {v}건 ({v/total*100:.1f}%)")
    behind = ctx[["4점 이상 뒤짐", "1~3점 뒤짐"]].sum()
    ahead = ctx[["1~3점 앞섬", "4점 이상 앞섬"]].sum()
    lines.append(f"  → 지고 있을 때 내려간 비율 {behind/total*100:.1f}% vs "
                 f"앞서고 있을 때 {ahead/total*100:.1f}%\n")

    lines.append("[2] 선발이 내려온 순간의 주자 상황")
    r = hooks["주자상황"].value_counts()
    for idx, v in r.items():
        lines.append(f"  - {idx}: {v}건 ({v/len(hooks)*100:.1f}%)")
    lines.append("")

    lines.append("[3] 같은 투구수 구간에서 상황별 강판율")
    for (band, sit), row in rate_ctx.iterrows():
        lines.append(f"  - {band} / {sit}: {row['강판율']*100:.1f}% (타석 {int(row['타석'])})")
    lines.append("")

    lines.append("[4] 주자 유무의 효과 (투구수 고정)")
    for band in ["61~75구", "76~90구", "91~105구"]:
        no = runner.loc[(band, "주자 없음")]
        yes = runner.loc[(band, "주자 있음")]
        d, lo, hi, sig = prop_ci(no["강판율"] * no["타석"], no["타석"],
                                 yes["강판율"] * yes["타석"], yes["타석"])
        lines.append(f"  - {band}: 주자 없음 {no['강판율']*100:.1f}% → 주자 있음 {yes['강판율']*100:.1f}% "
                     f"({d*100:+.1f}%p [{lo*100:+.1f}, {hi*100:+.1f}] "
                     f"{'유의함' if sig else '유의하지 않음'})")

    out = "\n".join(lines)
    path = FIG.parent / f"eda_{SEASON}_hook_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


def main() -> None:
    pa = load()
    ctx = fig_hook_context(pa)
    rate_ctx = fig_hook_rate_by_context(pa)
    runner = fig_runner_effect(pa)
    summarize(pa, ctx, rate_ctx, runner)
    print(f"그림 3장 -> {FIG}")


if __name__ == "__main__":
    main()
