"""생존 편향 통제가 만든 반대 방향의 편향을 잰다.

외부 평가의 세 번째 지적이다. 나는 생존 편향을 줄곧 '투수가 좋아 보이게 만드는 쪽'
으로만 다뤘고, 그래서 '3바퀴까지 간 등판만' 남겼다. 그런데 그 필터는 반대 방향의
편향을 만든다 -- 3바퀴까지 가려면 1·2바퀴를 잘 막았어야 하므로, 남은 등판의
1바퀴 성적이 인위적으로 좋아진다. 그러면 1->3바퀴 차이가 부풀려진다.

두 설계가 반대 방향으로 틀린다는 점을 이용한다.

  A 생존 등판만        1바퀴가 선택돼 좋아지므로 <과대>
  B 제한 없음          3바퀴는 좋은 투수만 도달하므로 <과소>
  C 투수x시즌 고정효과  같은 투수의 다른 등판끼리 이어 붙여 중간쯤

A와 B가 참값을 사이에 두는 구간이고, C가 그 안의 한 점이다.
편향의 크기 자체도 직접 잰다 -- 도달한 등판과 못 한 등판의 1바퀴 성적을 비교하면
필터가 무엇을 골라냈는지 바로 보인다.
"""
import argparse
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
VIOLET = "#7b5bd6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SIG = {True: "유의함", False: "유의하지 않음"}


def load(seasons):
    pa = pd.read_parquet(DATA / "kbo_pa_runvalue.parquet")
    pa = pa[pa["season"].isin(seasons) & pa["is_starter"]].copy()
    ok = set()
    for s in seasons:
        path = DATA / f"kbo_state_check_{s}.parquet"
        if path.exists():
            c = pd.read_parquet(path)
            ok |= set(c.loc[c["score_ok"] & c["starter_ok"], "game_id"])
    pa = pa[pa["game_id"].isin(ok)].copy()
    pa["start"] = pa["game_id"].astype(str) + "_" + pa["pitcher"].astype(str)
    pa["투수시즌"] = pa["pitcher"].astype(str) + "_" + pa["season"].astype(str)
    pa["안타"] = pa["안타"].astype(float)
    pa["타수"] = pa["타수"].astype(bool)
    reach = pa.groupby("start")["타순회전"].max()
    pa["도달3"] = pa["start"].map(reach >= 3)
    return pa


def std_mean(d, value, weights):
    """타순 구성을 공통 가중치로 맞춘 평균."""
    cells = d.groupby("batting_order")[value].mean().reindex(weights.index)
    w = weights[cells.notna()]
    return float((cells.dropna() * w).sum() / w.sum())


def diff_with_se(a, b, value, weights):
    """두 집합의 표준화 평균 차이. 표준오차는 등판 단위 군집으로."""
    d = pd.concat([a.assign(_g=0.0), b.assign(_g=1.0)], ignore_index=True)
    X = pd.DataFrame({"_g": d["_g"].astype(float)}, index=d.index)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = sm.add_constant(X)
    f = sm.OLS(d[value].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})
    ci = f.conf_int().loc["_g"]
    return f.params["_g"], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def fe_diff(d, value):
    """투수x시즌 고정효과로 1바퀴 대 3바퀴. 등판 간 비교가 섞인다."""
    X = pd.DataFrame(index=d.index)
    X["_g"] = (d["타순회전"] == 3).astype(float)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    cols = [value] + list(X.columns)
    w = pd.concat([d[[value]].astype(float), X], axis=1)
    w = w - w.groupby(d["투수시즌"].to_numpy()).transform("mean")
    f = sm.OLS(w[value], w[list(X.columns)]).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})
    ci = f.conf_int().loc["_g"]
    return f.params["_g"], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2017, 2027)))
    a = p.parse_args()

    pa = load(a.seasons)
    slots = pa["batting_order"].value_counts(normalize=True)
    starts = pa["start"].nunique()
    reach_rate = pa.groupby("start")["도달3"].first().mean()

    L = [f"=== 생존 편향 통제가 만든 반대 방향의 편향 ({len(a.seasons)}시즌) ===",
         f"선발 등판 {starts:,}개 · 3바퀴 도달 {reach_rate*100:.1f}%",
         f"선발 타석 {len(pa):,}건\n"]

    # ── [1] 필터가 무엇을 골라냈나 ─────────────────────────────
    L.append("[1] 필터의 선택 효과 — 3바퀴에 도달한 등판의 1바퀴 성적")
    t1 = pa[pa["타순회전"] == 1]
    for value, lab, scale in (("안타", "피안타율", 1000), ("득점가치", "득점가치", 1000)):
        src = t1[t1["타수"]] if value == "안타" else t1
        yes = src[src["도달3"]]
        no = src[~src["도달3"]]
        m_yes = std_mean(yes, value, slots)
        m_no = std_mean(no, value, slots)
        d_, lo, hi, sig = diff_with_se(no, yes, value, slots)
        L.append(f"  {lab}: 도달 {m_yes:.4f} vs 미도달 {m_no:.4f} "
                 f"· 도달−미도달 {d_*scale:+.1f}/1000 [{lo*scale:+.1f}, {hi*scale:+.1f}] {SIG[sig]}")
    L.append("  → 도달한 등판은 1바퀴를 더 잘 막았다. "
             "그 등판만 남기면 1바퀴 기준선이 낮아져 1→3 차이가 부풀려진다.\n")

    # ── [2] 세 설계로 1→3바퀴 ──────────────────────────────────
    L.append("[2] 1→3바퀴 페널티를 세 설계로")
    res = {}
    for value, lab, scale in (("안타", "피안타율", 1000), ("득점가치", "득점가치", 1000)):
        L.append(f"  [{lab}]")
        src = pa[pa["타수"]] if value == "안타" else pa
        t3 = src[(src["타순회전"] == 3) & src["도달3"]]

        # A: 생존 등판만 (지금까지 쓴 방식)
        a1 = src[(src["타순회전"] == 1) & src["도달3"]]
        A = diff_with_se(a1, t3, value, slots)
        # B: 1바퀴는 모든 등판
        b1 = src[src["타순회전"] == 1]
        B = diff_with_se(b1, t3, value, slots)
        # C: 투수×시즌 고정효과 (1바퀴는 모든 등판, 3바퀴는 도달 등판)
        cdf = pd.concat([b1, t3], ignore_index=True)
        C = fe_diff(cdf, value)
        res[value] = dict(A=A, B=B, C=C)
        for name, g in (("A 생존 등판만 (지금까지)", A),
                        ("B 제한 없음", B),
                        ("C 투수×시즌 고정효과", C)):
            L.append(f"    {name:24s}: {g[0]*scale:+.2f}/1000 "
                     f"[{g[1]*scale:+.2f}, {g[2]*scale:+.2f}] {SIG[g[3]]}")
        lo = min(A[0], B[0]) * scale
        hi = max(A[0], B[0]) * scale
        L.append(f"    → 두 설계가 만드는 구간: {lo:+.1f} ~ {hi:+.1f}/1000 "
                 f"(폭 {hi-lo:.1f}) · 고정효과는 {C[0]*scale:+.1f}")
    L.append("")

    # ── [3] 1·2바퀴 성적으로 가른 하위표본 ─────────────────────
    # 평균 회귀를 직접 보이는 방법 — 1바퀴를 잘 막은 등판만 모으면
    # 3바퀴에 '원래 수준으로 돌아오는' 몫이 섞인다.
    L.append("[3] 평균 회귀를 직접 보기 — 1바퀴 결과로 가른 하위표본 (도달 등판)")
    deep = pa[pa["도달3"]].copy()
    first = (deep[(deep["타순회전"] == 1) & deep["타수"]]
             .groupby("start")["안타"].mean().rename("1바퀴피안타율"))
    deep = deep.join(first, on="start")
    deep["1바퀴군"] = pd.cut(deep["1바퀴피안타율"], [-0.01, 0.0, 0.25, 1.01],
                          labels=["무안타", "0~.25", ".25 초과"])
    for grp, g in deep[deep["타수"]].groupby("1바퀴군", observed=True):
        m1 = std_mean(g[g["타순회전"] == 1], "안타", slots)
        m3 = std_mean(g[g["타순회전"] == 3], "안타", slots)
        n = g["start"].nunique()
        L.append(f"  1바퀴 {grp}: 1바퀴 {m1:.4f} → 3바퀴 {m3:.4f} "
                 f"(차이 {(m3-m1)*1000:+.1f}/1000, 등판 {n:,})")
    L.append("  → 1바퀴를 잘 막은 등판일수록 1→3 차이가 크다. 평균 회귀가 섞인다는 증거다.\n")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), dpi=150, facecolor=SURFACE)
    for ax, (value, lab, scale) in zip(axes, (("안타", "피안타율", 1000),
                                              ("득점가치", "득점가치", 1000))):
        names = ["A 생존 등판만\n(지금까지)", "B 제한 없음", "C 투수×시즌\n고정효과"]
        picks = [res[value]["A"], res[value]["B"], res[value]["C"]]
        vals = [g[0] * scale for g in picks]
        errs = [(g[0] - g[1]) * scale for g in picks]
        x = np.arange(3)
        ax.bar(x, vals, color=[MUTED, BLUE, VIOLET], width=0.56)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK,
                    elinewidth=1.3, capsize=5)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + 0.6, f"{v:+.1f}", ha="center", color=SECONDARY_INK,
                    fontsize=10, fontweight="bold")
        ax.axhline(0, color=INK, linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.3)
        ax.set_title(f"1→3바퀴 페널티 — {lab}", color=INK, fontsize=12, loc="left", pad=10)
        ax.set_ylabel("1/1000", color=SECONDARY_INK, fontsize=9.5)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_facecolor(SURFACE)
    fig.suptitle("생존 편향 통제는 어느 쪽으로 틀리나", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "survivorship.png")
    plt.close(fig)

    path = DATA.parent / "eda_survivorship_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
