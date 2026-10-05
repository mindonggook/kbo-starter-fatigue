"""발견 13의 메커니즘을 독립적으로 검증한다 — 익숙함은 경기를 넘어서도 작동하는가.

투구 단위 분석은 한 경기 안의 성적 저하가 '투수의 쇠퇴'가 아니라 '타자의 적응'이라고
말했다. 그렇다면 그 적응은 경기가 끝나면 사라지는가, 아니면 쌓이는가.

예측이 분명하다. 적응이 기억으로 쌓인다면, 같은 투수를 여러 번 상대해온 타자는
<그 경기의 첫 타석부터> 더 잘 쳐야 한다. 그래서

  - 타순 1바퀴째 타석만 본다 (경기 안의 익숙함을 0으로)
  - 같은 (투수, 타자) 쌍 안에서 초기 대결과 후기 대결을 비교한다
    -> 타자 실력도 투수 실력도 쌍 안에서 상쇄된다

효과가 있으면 적응 가설이 강해지고, 없으면 경기 내 적응은 '기억'이 아니라
'그날 눈에 익는 것'에 가깝다는 뜻이 된다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
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

MIN_PAIR_PA = 8  # 쌍 안에서 초기/후기를 가르려면 최소한 이 정도는 만나야 한다


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


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        pa_path = DATA / f"kbo_pa_state_{s}.parquet"
        chk_path = DATA / f"kbo_state_check_{s}.parquet"
        if not pa_path.exists():
            continue
        pa = pd.read_parquet(pa_path)
        chk = pd.read_parquet(chk_path)
        ok = set(chk.loc[chk["score_ok"] & chk["starter_ok"], "game_id"])
        d = pa[pa["game_id"].isin(ok) & pa["is_starter"]].copy()
        d["season"] = s
        frames.append(d)
    pa = pd.concat(frames, ignore_index=True)
    pa["안타"] = pa["안타"].astype(int)
    pa["타수"] = pa["타수"].astype(bool)
    pa["출루"] = (pa["안타"].astype(bool) | (pa["종류"] == "볼넷·사구")).astype(int)
    pa = pa.dropna(subset=["batter", "pitcher"])

    # 대결 순번은 모든 타석으로 쌓이고(익숙함은 몇 바퀴째였든 쌓이므로),
    # 분석은 1바퀴째 타석만 쓴다.
    pa = pa.sort_values(["date", "game_id", "inning", "half"]).reset_index(drop=True)
    pair = ["pitcher", "batter"]
    pa["대결순번"] = pa.groupby(pair).cumcount()          # 이전까지 만난 횟수
    pa["시즌내_대결순번"] = pa.groupby(pair + ["season"]).cumcount()
    return pa


def mh_pooled(groups) -> dict:
    """쌍별로 초기 vs 후기 비율 차이를 구해 가중 결합 (Mantel-Haenszel 위험차)."""
    num = den = var = 0.0
    used = 0
    for _, g in groups:
        early = g[g["후기"] == 0]
        late = g[g["후기"] == 1]
        n1, n2 = early["타수"].sum(), late["타수"].sum()
        if n1 < 3 or n2 < 3:
            continue
        p1, p2 = early["안타"].sum() / n1, late["안타"].sum() / n2
        w = (n1 * n2) / (n1 + n2)
        num += w * (p2 - p1)
        den += w
        var += w ** 2 * (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        used += 1
    diff = num / den
    se = np.sqrt(var) / den
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return {"diff": diff, "lo": lo, "hi": hi, "유의": (lo > 0) == (hi > 0), "쌍": used}


def within_pair(pa: pd.DataFrame, tto1_only: bool) -> dict:
    d = pa[pa["타순회전"] == 1] if tto1_only else pa
    d = d.copy()
    counts = d.groupby(["pitcher", "batter"])["안타"].transform("size")
    d = d[counts >= MIN_PAIR_PA].copy()
    # 쌍마다 대결 순서의 중앙값을 기준으로 초기/후기를 가른다.
    med = d.groupby(["pitcher", "batter"])["대결순번"].transform("median")
    d["후기"] = (d["대결순번"] > med).astype(int)
    return mh_pooled(d.groupby(["pitcher", "batter"]))


def by_bucket(pa: pd.DataFrame) -> pd.DataFrame:
    """서술 통계 — 누적 대결 횟수 구간별 1바퀴째 피안타율(타자 실력 미통제)."""
    d = pa[pa["타순회전"] == 1].copy()
    d["대결구간"] = pd.cut(d["대결순번"], [-1, 0, 5, 15, 40, 10000],
                        labels=["첫 대결", "1~5번째", "6~15번째", "16~40번째", "41번+"])
    g = d.groupby("대결구간", observed=True).agg(안타=("안타", "sum"), 타수=("타수", "sum"))
    g["피안타율"] = g["안타"] / g["타수"]
    g["se"] = np.sqrt(g["피안타율"] * (1 - g["피안타율"]) / g["타수"])
    return g


def fig_familiarity(bucket: pd.DataFrame, res_tto1: dict, res_all: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    x = np.arange(len(bucket))
    ax.bar(x, bucket["피안타율"], color=BLUE, width=0.55)
    ax.errorbar(x, bucket["피안타율"], yerr=bucket["se"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, r in zip(x, bucket.itertuples()):
        ax.text(xi, r.피안타율 + r.se * 1.96 + 0.004, f"{r.피안타율:.3f}",
                ha="center", color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
        ax.text(xi, 0.008, f"{int(r.타수):,}타수", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket.index.astype(str), fontsize=9)
    ax.set_ylim(0, bucket["피안타율"].max() * 1.3)
    _style_ax(ax, "누적 대결 횟수별 1바퀴째 피안타율", "그 투수를 몇 번째 상대하는가", "피안타율")

    ax = axes[1]
    labels = ["1바퀴째 타석만\n(경기 간 익숙함)", "모든 타석\n(참고)"]
    vals = [res_tto1["diff"], res_all["diff"]]
    los = [res_tto1["lo"], res_all["lo"]]
    his = [res_tto1["hi"], res_all["hi"]]
    y = np.arange(len(labels))[::-1]
    for yi, v, lo, hi, r in zip(y, vals, los, his, (res_tto1, res_all)):
        color = ORANGE if r["유의"] else MUTED
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.6, solid_capstyle="round")
        ax.plot(v, yi, "o", color=color, markersize=10)
        ax.text(hi + 0.002, yi, f"{v:+.4f}", va="center", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.05, 0.05)
    _style_ax(ax, "같은 쌍 안에서 초기 대결 → 후기 대결", "피안타율 차이 (95% 신뢰구간)", "")

    fig.suptitle("타자는 같은 투수를 반복해 만나면 유리해지는가",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "familiarity_between_games.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2017, 2027)))
    a = p.parse_args()

    pa = load(a.seasons)
    bucket = by_bucket(pa)
    res_tto1 = within_pair(pa, tto1_only=True)
    res_all = within_pair(pa, tto1_only=False)

    # 시즌 안에서만 본 버전 — 노화·기량 변화가 섞이는 것을 줄인다
    season_pa = pa.copy()
    season_pa["대결순번"] = season_pa["시즌내_대결순번"]
    res_season = within_pair(season_pa, tto1_only=True)

    fig_familiarity(bucket, res_tto1, res_all)

    lines = [f"=== 경기 간 익숙함 검증 ({len(a.seasons)}시즌 · 선발 상대 타석 {len(pa):,}건) ===\n",
             "적응이 기억으로 쌓인다면, 같은 투수를 여러 번 상대한 타자는",
             "그 경기의 첫 타석(1바퀴째)부터 더 잘 쳐야 한다.\n"]

    lines.append("[1] 누적 대결 횟수별 1바퀴째 피안타율 (타자 실력 미통제 — 참고용)")
    for idx, r in bucket.iterrows():
        lines.append(f"  {idx}: {r['피안타율']:.4f} (±{r['se']*1.96:.4f}, {int(r['타수']):,}타수)")
    lines.append("")

    lines.append("[2] 같은 (투수, 타자) 쌍 안에서 초기 → 후기 대결")
    lines.append("    쌍 안에서 비교하므로 타자 실력과 투수 실력이 상쇄된다.")
    for name, r in (("1바퀴째 타석만 (경기 간 익숙함)", res_tto1),
                    ("모든 타석 (참고)", res_all),
                    ("같은 시즌 안에서만, 1바퀴째", res_season)):
        mark = "유의함" if r["유의"] else "유의하지 않음"
        lines.append(f"  - {name}: {r['diff']:+.4f} "
                     f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {mark} (쌍 {r['쌍']:,}개)")

    out = "\n".join(lines)
    path = RESULTS / "eda_familiarity_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
