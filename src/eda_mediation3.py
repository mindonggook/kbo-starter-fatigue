"""발견 23을 구종 고정 흔들림으로 다시 분해한다.

발견 23은 '릴리스 흔들림 → 피출루율'이 총효과의 16.7%를 설명한다고 했다.
그때의 흔들림은 등판 평균 릴리스 대비 거리였는데, 구종마다 릴리스가 원래 다르다.
구종 차이는 제구와 무관한 잡음이므로 경로 b를 희석시킨다 — 즉 16.7%는
과대평가가 아니라 과소평가일 수 있다(발견 24에서 실제로 그랬다).

같은 분해를 두 가지 흔들림으로 나란히 돌려 몫이 어떻게 달라지는지 본다.
타석 단위 자료를 naver_full에서 직접 만들어 두 정의가 같은 표본을 쓰게 한다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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


def load(seasons):
    frames = []
    for s in seasons:
        path = DATA / f"naver_full_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["구속", "구종", "release_x", "release_z", "타석종류"])
    d = d[d["구속"].between(100, 170)].copy()

    first = d[d["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    d = d[[c in starters.get(g, set())
           for g, c in zip(d["game_id"], d["pitcher_code"])]].copy()
    d["start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)

    reached = d[d["투수_누적투구수"] >= 90]["start"].unique()
    d = d[d["start"].isin(reached) & d["투수_누적투구수"].between(1, 100)].copy()

    # 흔들림 두 정의 — 등판 안에서, 그리고 등판×구종 안에서
    for c in ("release_x", "release_z"):
        d[f"{c}_s"] = d[c] - d.groupby("start")[c].transform("mean")
        d[f"{c}_t"] = d[c] - d.groupby(["start", "구종"])[c].transform("mean")
    d["흔들림_등판"] = np.hypot(d["release_x_s"], d["release_z_s"]) * 30.48
    d["흔들림_구종"] = np.hypot(d["release_x_t"], d["release_z_t"]) * 30.48

    # 타석 경계 — 투구번호가 1로 돌아가는 지점
    d = d.sort_values(["start", "투수_누적투구수"])
    d["pa_id"] = (d.groupby("start")["투구번호"].diff().fillna(1) <= 0).cumsum()

    pa = d.groupby(["start", "pa_id"], as_index=False).agg(
        season=("season", "first"), batting_order=("batting_order", "first"),
        종류=("타석종류", "first"), 투구수=("투구번호", "size"),
        시작투구수=("투수_누적투구수", "min"),
        흔들림_등판=("흔들림_등판", "mean"), 흔들림_구종=("흔들림_구종", "mean"))
    pa["이전_누적투구수"] = pa["시작투구수"] - 1
    pa = pa[pa["이전_누적투구수"].between(0, 100)].copy()
    pa["출루"] = pa["종류"].isin(["안타", "볼넷·사구"]).astype(float)
    pa["볼넷"] = (pa["종류"] == "볼넷·사구").astype(float)
    pa["투구수_10구당"] = pa["이전_누적투구수"] / 10.0
    return pa


def fit_ols(d, y, cols):
    X = pd.DataFrame(index=d.index)
    for c in cols:
        X[c] = d[c].astype(float)
    X["타석투구수"] = d["투구수"].astype(float)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = sm.add_constant(X)
    return sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})


def decompose(d, mediator, y):
    Xa = pd.DataFrame({"투구수_10구당": d["투구수_10구당"].astype(float),
                       "타석투구수": d["투구수"].astype(float)}, index=d.index)
    fa = sm.OLS(d[mediator].astype(float), sm.add_constant(Xa)).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})
    a_coef = fa.params["투구수_10구당"]

    total_fit = fit_ols(d, y, ["투구수_10구당"])
    both_fit = fit_ols(d, y, ["투구수_10구당", mediator])
    total = total_fit.params["투구수_10구당"]
    b_coef = both_fit.params[mediator]
    b_ci = both_fit.conf_int().loc[mediator]
    mediated = a_coef * b_coef
    return dict(a=a_coef, b=b_coef, b_lo=b_ci[0], b_hi=b_ci[1],
                b_sig=(b_ci[0] > 0) == (b_ci[1] > 0),
                total=total, direct=both_fit.params["투구수_10구당"],
                mediated=mediated,
                share=mediated / total * 100 if total else float("nan"))


def boot_share(d, mediator, y, reps=400, seed=20261004):
    """매개분의 신뢰구간은 a×b 곱이라 정규근사가 맞지 않는다 — 등판 단위 부트스트랩."""
    rng = np.random.default_rng(seed)
    starts = d["start"].unique()
    idx = {k: g.index.to_numpy() for k, g in d.groupby("start")}
    out = []
    for _ in range(reps):
        pick = rng.choice(starts, size=len(starts), replace=True)
        rows = np.concatenate([idx[k] for k in pick])
        b = d.loc[rows].copy()
        b["start"] = np.repeat(np.arange(len(pick)),
                               [len(idx[k]) for k in pick])  # 재표집 군집 재명명
        try:
            out.append(decompose(b, mediator, y)["share"])
        except Exception:
            continue
    arr = np.array([x for x in out if np.isfinite(x)])
    if len(arr) < 50:
        return np.nan, np.nan, len(arr)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), len(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()
    d = load(a.seasons)

    L = [f"=== 구종 고정 흔들림으로 발견 23 재분해 ({len(a.seasons)}시즌) ===",
         f"선발 타석 {len(d):,}건 / 등판 {d['start'].nunique():,}개\n"]

    res = {}
    for y, ylab in (("출루", "피출루율"), ("볼넷", "볼넷률")):
        L.append(f"[{ylab}]")
        for med, mlab in (("흔들림_등판", "흔들림 (등판 기준 · 발견 23)"),
                          ("흔들림_구종", "흔들림 (구종 고정)")):
            r = decompose(d, med, y)
            res[(y, med)] = r
            L.append(f"  {mlab}")
            L.append(f"    경로 a (투구수 10구 -> 흔들림): {r['a']:+.4f} cm")
            L.append(f"    경로 b (흔들림 1cm -> {ylab}) : {r['b']:+.5f} "
                     f"[{r['b_lo']:+.5f}, {r['b_hi']:+.5f}] "
                     f"{'유의함' if r['b_sig'] else '유의하지 않음'}")
            L.append(f"    총효과 {r['total']:+.5f} = 매개 {r['mediated']:+.5f} "
                     f"+ 직접 {r['direct']:+.5f}")
            lo, hi, nb = boot_share(d, med, y)
            L.append(f"    -> 설명하는 몫: {r['share']:+.1f}% "
                     f"[부트스트랩 {lo:+.1f}, {hi:+.1f}] (재표집 {nb}회)")
        L.append("")

    # 흔들림 5분위별 피출루율 — 두 정의 나란히
    L.append("[참고] 흔들림 5분위별 피출루율")
    for med, mlab in (("흔들림_등판", "등판 기준"), ("흔들림_구종", "구종 고정")):
        q = pd.qcut(d[med], 5, labels=False)
        tab = d.groupby(q, observed=True).agg(피출루율=("출루", "mean"),
                                              흔들림=(med, "median"), n=("출루", "size"))
        spread = tab["피출루율"].iloc[-1] - tab["피출루율"].iloc[0]
        L.append(f"  {mlab}: " + " / ".join(
            f"{r['피출루율']:.3f}({r['흔들림']:.1f}cm)" for _, r in tab.iterrows())
            + f"  최상-최하 {spread*1000:+.1f}/1000")
    L.append("")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=150, facecolor=SURFACE)
    for ax, (y, ylab) in zip(axes, (("출루", "피출루율"), ("볼넷", "볼넷률"))):
        labs, meds, dirs = [], [], []
        for med, mlab in (("흔들림_등판", "등판 기준\n(발견 23)"),
                          ("흔들림_구종", "구종 고정")):
            r = res[(y, med)]
            labs.append(mlab)
            meds.append(r["mediated"] * 1000)
            dirs.append(r["direct"] * 1000)
        x = np.arange(2)
        ax.bar(x, meds, color=ORANGE, width=0.5, label="릴리스 흔들림 경로")
        ax.bar(x, dirs, bottom=meds, color=MUTED, width=0.5, label="직접효과")
        for xi, (m, dr, med) in enumerate(zip(meds, dirs, ("흔들림_등판", "흔들림_구종"))):
            ax.text(xi, m + dr + 0.02, f"{res[(y, med)]['share']:.1f}%", ha="center",
                    color=INK, fontsize=11, fontweight="bold")
        ax.axhline(0, color=INK, linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=9)
        ax.set_title(f"투구수 10구당 {ylab} 변화의 분해", color=INK, fontsize=12,
                     loc="left", pad=10)
        ax.set_ylabel(f"{ylab} 변화 (1/1000)", color=SECONDARY_INK, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_facecolor(SURFACE)
    fig.suptitle("구종을 고정하면 제구 경로는 더 커진다", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "mediation_pitchtype.png")
    plt.close(fig)

    path = RESULTS / "eda_mediation3_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
