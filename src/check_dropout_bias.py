"""한계 6 — 검증에서 걸러낸 경기가 무작위인가.

문자중계에서 복원한 선발 투구수가 박스스코어와 어긋나는 경기는 분석에서 뺐다.
통과율이 시즌마다 97.1% / 80.4% / 91.0%로 달라서, 걸러진 경기가 특정 성격에
쏠려 있다면 결론이 흔들릴 수 있다.

중계 누락의 원인이 '투수 교체 기록이 빠지는 것'이므로, 교체가 많은 경기일수록
걸러질 가능성이 높다는 것이 유력한 가설이다. 그렇다면 걸러진 표본은 무작위가
아니라 '난타전'에 치우친다 — 그 방향과 크기를 재둔다.
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

SEASONS = tuple(range(2017, 2027))


def collect(season: int) -> pd.DataFrame:
    chk = pd.read_parquet(DATA / f"kbo_state_check_{season}.parquet")
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    sched = pd.read_parquet(DATA / f"kbo_schedule_{season}.parquet")

    box["투구수"] = box["투구수"].astype(float)
    per_game = box.groupby("game_id").agg(
        투수수=("선수명", "size"),
        구원투수수=("is_starter", lambda s: int((~s).sum())),
        총투구수=("투구수", "sum"),
        총실점=("실점", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
    )
    starters = box[box["is_starter"]].groupby("game_id").agg(
        선발평균이닝=("이닝_소수", "mean"), 선발평균투구수=("투구수", "mean"))

    g = (chk.set_index("game_id")
            .join(per_game).join(starters)
            .join(sched.set_index("game_id")[["date", "away_score", "home_score"]]))
    g["점수차"] = (g["away_score"] - g["home_score"]).abs()
    g["총득점"] = g["away_score"] + g["home_score"]
    g["월"] = pd.to_datetime(g["date"], format="%Y%m%d").dt.month
    g["season"] = season
    g["통과"] = g["starter_ok"]
    return g.reset_index()


def compare(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        a = df.loc[df["통과"], c].astype(float).dropna()
        b = df.loc[~df["통과"], c].astype(float).dropna()
        if len(b) < 5:
            continue
        diff = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        rows.append({"변수": c, "통과 경기": a.mean(), "걸러진 경기": b.mean(),
                     "차이": diff, "lo": lo, "hi": hi,
                     "유의": (lo > 0) == (hi > 0),
                     "n_통과": len(a), "n_걸러짐": len(b)})
    return pd.DataFrame(rows)


def fig_bias(res: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150, facecolor=SURFACE)
    # 변수마다 단위가 달라 통과 경기 평균 대비 몇 %인지로 본다.
    rel = res["차이"] / res["통과 경기"].abs() * 100
    rel_lo = res["lo"] / res["통과 경기"].abs() * 100
    rel_hi = res["hi"] / res["통과 경기"].abs() * 100

    y = np.arange(len(res))[::-1]
    for yi, (_, r), lo, hi, v in zip(y, res.iterrows(), rel_lo, rel_hi, rel):
        color = ORANGE if r["유의"] else MUTED
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.4, solid_capstyle="round")
        ax.plot(v, yi, "o", color=color, markersize=9)
        ax.text(hi + 1.2, yi, f"{v:+.1f}%", va="center", color=SECONDARY_INK,
                fontsize=9, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(res["변수"], fontsize=9.5)
    ax.set_xlabel("걸러진 경기가 통과 경기보다 (%)", color=SECONDARY_INK, fontsize=10)
    ax.set_title("걸러낸 경기는 어떤 경기였나 (세 시즌 합산, 95% 신뢰구간)",
                 color=INK, fontsize=13, loc="left", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / "dropout_bias.png")
    plt.close(fig)


def main() -> None:
    df = pd.concat([collect(s) for s in SEASONS], ignore_index=True)
    cols = ["투수수", "구원투수수", "총투구수", "총실점", "총득점", "점수차",
            "선발평균이닝", "선발평균투구수"]
    res = compare(df, cols)

    lines = ["=== 중계 결손 경기가 무작위인가 ===\n"]
    for s in SEASONS:
        d = df[df["season"] == s]
        lines.append(f"  {s}: 통과 {int(d['통과'].sum())} / 전체 {len(d)} "
                     f"({d['통과'].mean()*100:.1f}%)")
    lines.append("")

    lines.append("[통과 경기 vs 걸러진 경기]")
    for _, r in res.iterrows():
        mark = "유의함" if r["유의"] else "차이 없음"
        lines.append(f"  {r['변수']}: 통과 {r['통과 경기']:.2f} vs 걸러짐 {r['걸러진 경기']:.2f} "
                     f"({r['차이']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]) {mark}")
    lines.append("")

    sig = res[res["유의"]]
    if len(sig):
        lines.append("→ 걸러진 경기는 무작위가 아니다. 두드러진 차이:")
        for _, r in sig.iterrows():
            direction = "많은" if r["차이"] > 0 else "적은"
            lines.append(f"   · {r['변수']}가 {abs(r['차이']):.2f} 더 {direction} 경기")
    else:
        lines.append("→ 관측 가능한 특성에서 통과/걸러짐 사이에 유의한 차이가 없다.")

    out = "\n".join(lines)
    path = DATA.parent / "dropout_bias_summary.txt"
    path.write_text(out, encoding="utf-8")
    fig_bias(res)
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
