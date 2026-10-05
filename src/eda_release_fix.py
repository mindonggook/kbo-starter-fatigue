"""발견 22~24를 올바른 릴리스 지점으로 다시 잰다.

외부 평가에서 지적받았다. PITCHf/x의 (x0, z0)는 실제 릴리스가 아니라
홈플레이트에서 <50피트 지점>의 좌표다. 데이터를 확인하니 y0이 전부 50.0이었다.
지적이 맞았다.

그러면 내가 '릴리스 흔들림'이라 부른 것에는 투구 궤적이 섞인다. 볼이 될 공은
50피트 지점에서 이미 다른 곳을 지나가고 있으므로, '흔들리면 볼이 된다'의
일부가 기계적으로 생겼을 수 있다.

등가속도 궤적으로 55피트 평면까지 되돌려 다시 잰다. 네 조합을 나란히 둔다.
  평면   50피트(원래) vs 55피트(보정)
  구종   고정하지 않음 vs 고정(발견 24에서 이미 한 보정)
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
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SIG = {True: "유의함", False: "유의하지 않음"}

COMBOS = [
    ("50피트 · 구종 자유", "p50_x", "p50_z", False),
    ("50피트 · 구종 고정", "p50_x", "p50_z", True),
    ("55피트 · 구종 자유", "release_x", "release_z", False),
    ("55피트 · 구종 고정", "release_x", "release_z", True),
]


def load(seasons):
    frames = []
    for s in seasons:
        path = DATA / f"naver_full_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["구속", "구종", "투구결과",
                         "p50_x", "p50_z", "release_x", "release_z"])
    d = d[d["구속"].between(100, 170)].copy()

    first = d[d["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    d = d[[c in starters.get(g, set())
           for g, c in zip(d["game_id"], d["pitcher_code"])]].copy()
    d["start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)

    reached = d[d["투수_누적투구수"] >= 90]["start"].unique()
    d = d[d["start"].isin(reached) & d["투수_누적투구수"].between(1, 100)].copy()

    d["볼"] = (d["투구결과"] == "볼").astype(float)
    d["투구수_10구당"] = d["투수_누적투구수"] / 10.0
    d["구속_편차"] = d["구속"] - d.groupby("start")["구속"].transform("mean")
    return d


def wobble(d, cx, cz, by_type):
    keys = ["start", "구종"] if by_type else ["start"]
    dx = d[cx] - d.groupby(keys)[cx].transform("mean")
    dz = d[cz] - d.groupby(keys)[cz].transform("mean")
    return np.hypot(dx, dz) * 30.48


def model(d, w):
    X = pd.DataFrame(index=d.index)
    X["흔들림_cm"] = w.astype(float)
    X["투구수_10구당"] = d["투구수_10구당"].astype(float)
    X["구속_편차"] = d["구속_편차"].astype(float)
    X = X.join(pd.get_dummies(d["구종"], prefix="구종", drop_first=True).astype(float))
    X = sm.add_constant(X)
    f = sm.OLS(d["볼"], X).fit(cov_type="cluster", cov_kwds={"groups": d["start"]})
    ci = f.conf_int().loc["흔들림_cm"]
    return f.params["흔들림_cm"], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def spread(d, w):
    q = pd.qcut(w, 5, labels=False, duplicates="drop")
    t = d.assign(_q=q, _w=w).groupby("_q", observed=True).agg(
        볼비율=("볼", "mean"), 흔들림=("_w", "median"), n=("볼", "size"))
    t["se"] = np.sqrt(t["볼비율"] * (1 - t["볼비율"]) / t["n"])
    lo, hi = t.index[0], t.index[-1]
    diff = t.loc[hi, "볼비율"] - t.loc[lo, "볼비율"]
    se = np.hypot(t.loc[lo, "se"], t.loc[hi, "se"])
    return t, diff, diff - 1.96 * se, diff + 1.96 * se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()
    d = load(a.seasons)

    L = [f"=== 발견 22~24를 올바른 릴리스 지점으로 재측정 ({len(a.seasons)}시즌) ===",
         f"선발 투구 {len(d):,}개 · 등판 {d['start'].nunique():,}개",
         "y0은 전 투구에서 50.0 — 원래 쓰던 좌표는 50피트 평면이 맞았다.",
         "등가속도 궤적으로 55피트까지 역외삽해 다시 쟀다.\n"]

    # 두 평면이 얼마나 다른가
    L.append("[0] 두 평면의 관계")
    L.append(f"  x 상관 {d['release_x'].corr(d['p50_x']):.4f} · "
             f"z 상관 {d['release_z'].corr(d['p50_z']):.4f}")
    L.append(f"  x 차이 sd {(d['release_x']-d['p50_x']).std():.4f} ft · "
             f"z 차이 sd {(d['release_z']-d['p50_z']).std():.4f} ft")
    L.append("")

    L.append("[1] 네 조합으로 다시 잰 '흔들림 → 볼 비율'")
    res = {}
    for name, cx, cz, by_type in COMBOS:
        w = wobble(d, cx, cz, by_type)
        t, diff, lo, hi = spread(d, w)
        m = model(d, w)
        res[name] = dict(tab=t, diff=diff, lo=lo, hi=hi, m=m, med=w.median())
        L.append(f"  [{name}] 흔들림 중앙 {w.median():.2f}cm")
        L.append(f"    5분위 최상−최하: {diff*100:+.2f}%p [{lo*100:+.2f}, {hi*100:+.2f}]")
        L.append(f"    회귀 1cm당     : {m[0]*100:+.4f}%p "
                 f"[{m[1]*100:+.4f}, {m[2]*100:+.4f}] {SIG[m[3]]}")
    L.append("")

    base = res["50피트 · 구종 자유"]["m"][0]
    fixed = res["55피트 · 구종 고정"]["m"][0]
    L.append(f"  → 원래 보고값(50피트·구종 자유) 대비 "
             f"올바른 측정(55피트·구종 고정)은 {fixed/base*100:.1f}%")
    plane_only = res["55피트 · 구종 자유"]["m"][0]
    L.append(f"  → 평면만 바로잡으면 {plane_only/base*100:.1f}% "
             f"(평면 보정이 깎는 몫 {(1-plane_only/base)*100:+.1f}%p)")
    L.append("")

    # 투구수에 따라 흔들림이 커지는가 — 발견 17의 '폼이 두 배로 흔들린다'
    L.append("[2] 투구수에 따른 흔들림 증가 (발견 17의 릴리스 부분)")
    d["구간"] = pd.cut(d["투수_누적투구수"], [0, 25, 50, 75, 100],
                     labels=["1~25구", "26~50구", "51~75구", "76~100구"])
    for name, cx, cz, by_type in COMBOS:
        w = wobble(d, cx, cz, by_type)
        g = d.assign(_w=w).groupby("구간", observed=True)["_w"].mean()
        L.append(f"  [{name}] " + " / ".join(f"{k} {v:.2f}cm" for k, v in g.items())
                 + f"  차이 {g.iloc[-1]-g.iloc[0]:+.2f}cm")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    names = [c[0].replace(" · ", "\n") for c in COMBOS]
    vals = [res[c[0]]["m"][0] * 100 for c in COMBOS]
    errs = [(res[c[0]]["m"][0] - res[c[0]]["m"][1]) * 100 for c in COMBOS]
    colors = [MUTED, MUTED, BLUE, ORANGE]
    x = np.arange(4)
    ax.bar(x, vals, color=colors, width=0.56)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK,
                elinewidth=1.3, capsize=5)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.012, f"{v:+.3f}", ha="center", color=SECONDARY_INK,
                fontsize=9.5, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.25)
    ax.set_title("흔들림 1cm당 볼 비율", color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("%p", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    for (name, *_), color in zip(COMBOS, colors):
        t = res[name]["tab"]
        ax.plot(np.arange(len(t)), t["볼비율"] * 100, marker="o", markersize=6,
                linewidth=2.0, color=color, label=name)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f"{i+1}분위" for i in range(5)], fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    ax.set_title("흔들림 5분위별 볼 비율", color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("볼 비율 (%)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("릴리스 지점을 바로잡으면 발견 22는 살아남는가",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "release_fix.png")
    plt.close(fig)

    path = DATA.parent / "eda_release_fix_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
