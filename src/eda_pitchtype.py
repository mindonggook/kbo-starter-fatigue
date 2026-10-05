"""구종을 손에 넣고 발견 17·22를 다시 잰다.

두 발견은 구종 없이 측정됐고, 바로 그래서 위험하다.

발견 17 — '투구수가 쌓이면 구속이 1.6km/h 떨어진다'
    평균 구속을 쟀다. 그런데 직구(143km/h)와 커브(119km/h)는 24km/h 차이다.
    후반에 변화구를 더 섞기만 해도 평균은 내려간다. 팔이 지친 것이 아니라
    투수가 다르게 던진 것일 수 있다.

발견 22 — '릴리스가 흔들리면 볼이 16%p 늘어난다'
    등판 평균 릴리스에서 벗어난 거리를 '흔들림'이라 불렀다. 그런데 구종마다
    릴리스 포인트가 원래 다르다. 내가 잰 흔들림의 상당 부분이 폼이 무너진 것이
    아니라 그냥 다른 구종을 던진 것일 수 있다.

가르는 방법은 같다 — 구종을 고정하고 다시 재면 된다.
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

BUCKETS = [0, 25, 50, 75, 100]
LABELS = ["1~25구", "26~50구", "51~75구", "76~100구"]
SIG = {True: "유의함", False: "유의하지 않음"}


def style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.set_facecolor(SURFACE)


def load(seasons):
    frames = []
    for s in seasons:
        path = DATA / f"naver_full_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["구속", "구종", "pitcher", "release_x", "release_z"])
    d = d[d["구속"].between(100, 170)].copy()

    first = d[d["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    d["is_starter"] = [c in starters.get(g, set())
                       for g, c in zip(d["game_id"], d["pitcher_code"])]
    d["start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)
    d = d[d["is_starter"]].copy()

    # 발견 13·17·22와 같은 생존 편향 통제
    reached = d[d["투수_누적투구수"] >= 90]["start"].unique()
    d = d[d["start"].isin(reached) & d["투수_누적투구수"].between(1, 100)].copy()

    d["구간"] = pd.cut(d["투수_누적투구수"], BUCKETS, labels=LABELS)
    d["볼"] = (d["투구결과"] == "볼").astype(float)
    d["직구"] = (d["구종"] == "직구").astype(float)
    d["투구수_10구당"] = d["투수_누적투구수"] / 10.0

    # 등판 안에서 뺀 것과, 등판×구종 안에서 뺀 것 — 이 둘의 차이가 구종 혼입이다.
    d["구속_편차"] = d["구속"] - d.groupby("start")["구속"].transform("mean")
    d["구속_편차_구종고정"] = d["구속"] - d.groupby(["start", "구종"])["구속"].transform("mean")
    for c in ("release_x", "release_z"):
        d[f"{c}_s"] = d[c] - d.groupby("start")[c].transform("mean")
        d[f"{c}_t"] = d[c] - d.groupby(["start", "구종"])[c].transform("mean")
    d["흔들림"] = np.hypot(d["release_x_s"], d["release_z_s"]) * 30.48
    d["흔들림_구종고정"] = np.hypot(d["release_x_t"], d["release_z_t"]) * 30.48
    return d


def slope(d, y, col="투구수_10구당"):
    """투구수 10구당 기울기. 등판 단위 군집 보정."""
    X = sm.add_constant(pd.DataFrame({col: d[col].astype(float)}, index=d.index))
    fit = sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})
    ci = fit.conf_int().loc[col]
    b = fit.params[col]
    return b, ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def ball_model(d, wobble_col, with_type):
    X = pd.DataFrame(index=d.index)
    X["흔들림_cm"] = d[wobble_col].astype(float)
    X["투구수_10구당"] = d["투구수_10구당"].astype(float)
    X["구속_편차"] = d["구속_편차"].astype(float)
    if with_type:
        X = X.join(pd.get_dummies(d["구종"], prefix="구종", drop_first=True).astype(float))
    X = sm.add_constant(X)
    fit = sm.OLS(d["볼"], X).fit(cov_type="cluster", cov_kwds={"groups": d["start"]})
    ci = fit.conf_int().loc["흔들림_cm"]
    b = fit.params["흔들림_cm"]
    return b, ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def quintile_spread(d, col):
    q = pd.qcut(d[col], 5, labels=False, duplicates="drop")
    tab = d.groupby(q, observed=True).agg(볼비율=("볼", "mean"), 흔들림=(col, "median"),
                                          n=("볼", "size"))
    tab["se"] = np.sqrt(tab["볼비율"] * (1 - tab["볼비율"]) / tab["n"])
    lo, hi = tab.index[0], tab.index[-1]
    diff = tab.loc[hi, "볼비율"] - tab.loc[lo, "볼비율"]
    se = np.hypot(tab.loc[lo, "se"], tab.loc[hi, "se"])
    return tab, diff, diff - 1.96 * se, diff + 1.96 * se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()
    d = load(a.seasons)
    L = [f"=== 구종으로 발견 17·22 재검증 ({len(a.seasons)}시즌) ===",
         f"선발 투구 {len(d):,}개 / 등판 {d['start'].nunique():,}개 / 투수 {d['pitcher'].nunique()}명",
         "(90구 이상 간 등판, 1~100구, 등판 단위 군집 보정)\n"]

    # ── [1] 구종 분포와 구종별 구속 ─────────────────────────────
    mix = d.groupby("구종").agg(n=("구속", "size"), 구속=("구속", "mean"))
    mix["비율"] = mix["n"] / len(d) * 100
    mix = mix.sort_values("n", ascending=False)
    L.append("[1] 구종 분포와 평균 구속")
    for idx, r in mix.iterrows():
        L.append(f"  {idx:5s} {r['비율']:5.1f}%  {r['구속']:6.1f}km/h  (n={int(r['n']):,})")
    L.append(f"  -> 가장 빠른 구종과 가장 느린 구종의 차이: "
             f"{mix['구속'].max() - mix['구속'].min():.1f}km/h\n")

    # ── [2] 투구수가 쌓이면 구종 배합이 바뀌는가 ────────────────
    share = (d.groupby(["구간", "구종"], observed=True).size()
             / d.groupby("구간", observed=True).size()).unstack() * 100
    L.append("[2] 투구수 구간별 구종 비율 (%)")
    L.append("  " + "구종".ljust(7) + "".join(f"{c:>10s}" for c in LABELS))
    for t in mix.index:
        if t in share.columns:
            L.append("  " + t.ljust(7) + "".join(f"{share.loc[b, t]:>10.1f}" for b in LABELS))
    # 같은 등판 안에서 짝지어 비교 — 투수 구성 차이를 지운다.
    pair = (d[d["구간"].isin([LABELS[0], LABELS[-1]])]
            .pivot_table(index="start", columns="구간", values="직구",
                         aggfunc="mean", observed=True).dropna())
    pdiff = (pair[LABELS[-1]] - pair[LABELS[0]]) * 100
    pse = pdiff.std(ddof=1) / np.sqrt(len(pdiff))
    L.append(f"  -> 같은 등판 안에서 직구 비율 변화(76~100구 - 1~25구): "
             f"{pdiff.mean():+.2f}%p [{pdiff.mean()-1.96*pse:+.2f}, {pdiff.mean()+1.96*pse:+.2f}] "
             f"(등판 {len(pdiff):,}개)\n")

    # ── [3] 발견 17 재검증 ─────────────────────────────────────
    L.append("[3] 발견 17 재검증 - 구속 하락은 팔인가 배합인가")
    all_s = slope(d, "구속_편차")
    fast = d[d["구종"] == "직구"].copy()
    fast["구속_편차"] = fast["구속"] - fast.groupby("start")["구속"].transform("mean")
    fb_s = slope(fast, "구속_편차")
    fix_s = slope(d, "구속_편차_구종고정")
    for name, s in (("모든 구종 (발견 17의 방식)", all_s),
                    ("직구만", fb_s),
                    ("모든 구종 · 구종 고정", fix_s)):
        L.append(f"  {name:24s}: 10구당 {s[0]:+.4f}km/h [{s[1]:+.4f}, {s[2]:+.4f}] "
                 f"{SIG[s[3]]}  -> 100구 환산 {s[0]*9:+.2f}km/h")
    mixshare = (all_s[0] - fix_s[0]) / all_s[0] * 100 if all_s[0] else float("nan")
    L.append(f"  -> 구속 하락 중 구종 배합이 만든 몫: {mixshare:.1f}%\n")

    # 구종별로 따로 — 어느 구종이 느려지나
    L.append("  구종별 하락 (10구당, 자기 등판 평균 대비)")
    for t in mix.index:
        sub = d[d["구종"] == t].copy()
        if len(sub) < 2000:
            continue
        sub["y"] = sub["구속"] - sub.groupby("start")["구속"].transform("mean")
        s = slope(sub, "y")
        L.append(f"    {t:5s}: {s[0]:+.4f}km/h [{s[1]:+.4f}, {s[2]:+.4f}] "
                 f"{SIG[s[3]]} (n={len(sub):,})")
    L.append("")

    # ── [4] 발견 22 재검증 ─────────────────────────────────────
    L.append("[4] 발견 22 재검증 - 릴리스 흔들림은 폼인가 구종인가")
    var_all = d["흔들림"].var()
    var_fix = d["흔들림_구종고정"].var()
    L.append(f"  흔들림 중앙값: 등판 기준 {d['흔들림'].median():.2f}cm "
             f"-> 구종 고정 {d['흔들림_구종고정'].median():.2f}cm "
             f"(분산 {(1-var_fix/var_all)*100:.1f}% 감소)")
    tab_a, da, la, ha = quintile_spread(d, "흔들림")
    tab_b, db, lb, hb = quintile_spread(d, "흔들림_구종고정")
    L.append("  5분위 최상-최하 볼 비율 차")
    L.append(f"    등판 기준 (발견 22): {da*100:+.2f}%p [{la*100:+.2f}, {ha*100:+.2f}]")
    L.append(f"    구종 고정          : {db*100:+.2f}%p [{lb*100:+.2f}, {hb*100:+.2f}]")
    m1 = ball_model(d, "흔들림", with_type=False)
    m2 = ball_model(d, "흔들림", with_type=True)
    m3 = ball_model(d, "흔들림_구종고정", with_type=True)
    L.append("  회귀 - 흔들림 1cm당 볼 비율 (투구수·구속 통제)")
    for name, m in (("발견 22의 모형", m1),
                    ("+ 구종 더미", m2),
                    ("흔들림도 구종 고정 + 구종 더미", m3)):
        L.append(f"    {name:30s}: {m[0]*100:+.4f}%p [{m[1]*100:+.4f}, {m[2]*100:+.4f}] "
                 f"{SIG[m[3]]}")
    keep = m3[0] / m1[0] * 100 if m1[0] else float("nan")
    L.append(f"  -> 구종을 고정해도 남는 몫: {keep:.1f}%")
    L.append(f"  (참고) 5분위 볼 비율 - 구종 고정 기준")
    for idx, r in tab_b.iterrows():
        L.append(f"    {int(idx)+1}분위 (중앙 {r['흔들림']:.1f}cm): {r['볼비율']*100:.2f}% "
                 f"(n={int(r['n']):,})")
    L.append("")

    # ── [5] 리포트의 수치와 같은 방식(구간 차)으로 다시 ─────────
    # 발견 17의 -1.6km/h는 기울기가 아니라 '76~100구 − 1~25구'였다. 같은 자로 재야
    # 리포트 숫자를 얼마나 깎아야 하는지 말할 수 있다.
    L.append("[5] 구간 차로 다시 (발견 17이 쓴 자)")
    two = d[d["구간"].isin([LABELS[0], LABELS[-1]])].copy()
    two["후반"] = (two["구간"] == LABELS[-1]).astype(float)
    fast2 = two[two["구종"] == "직구"].copy()
    fast2["구속_편차"] = fast2["구속"] - fast2.groupby("start")["구속"].transform("mean")
    for name, sub, col in (("모든 구종 (발견 17의 방식)", two, "구속_편차"),
                           ("직구만", fast2, "구속_편차"),
                           ("모든 구종 · 구종 고정", two, "구속_편차_구종고정")):
        s = slope(sub, col, col="후반")
        L.append(f"  {name:24s}: {s[0]:+.3f}km/h [{s[1]:+.3f}, {s[2]:+.3f}] {SIG[s[3]]}")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    x = np.arange(len(LABELS))
    bottom = np.zeros(len(LABELS))
    order = [t for t in mix.index if t in share.columns][:6]
    colors = [BLUE, ORANGE, "#6fa8dc", "#f0a97f", "#9ec4e8", "#f6cdb4"]
    for t, c in zip(order, colors):
        v = share.loc[LABELS, t].values
        ax.bar(x, v, bottom=bottom, color=c, width=0.62, label=t)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), labelcolor=SECONDARY_INK)
    style(ax, "투구수가 쌓이면 배합이 바뀌나", "구종 비율 (%)")

    ax = axes[1]
    names = ["모든 구종\n(발견 17)", "직구만", "모든 구종\n구종 고정"]
    vals = [all_s[0] * 9, fb_s[0] * 9, fix_s[0] * 9]
    errs = [(s[0] - s[1]) * 9 for s in (all_s, fb_s, fix_s)]
    ax.bar(np.arange(3), vals, color=[MUTED, BLUE, ORANGE], width=0.56)
    ax.errorbar(np.arange(3), vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK,
                elinewidth=1.3, capsize=5)
    for i, v in enumerate(vals):
        ax.text(i, v - max(errs) - 0.05, f"{v:+.2f}", ha="center", va="top",
                color=SECONDARY_INK, fontsize=10, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(min(vals) - max(errs) - 0.5, max(0.1, max(vals) + 0.2))
    style(ax, "구속 하락 (100구 환산)", "km/h")

    ax = axes[2]
    names2 = ["발견 22\n모형", "+구종\n더미", "흔들림도\n구종 고정"]
    vals2 = [m1[0] * 100, m2[0] * 100, m3[0] * 100]
    errs2 = [(m[0] - m[1]) * 100 for m in (m1, m2, m3)]
    ax.bar(np.arange(3), vals2, color=[MUTED, BLUE, ORANGE], width=0.56)
    ax.errorbar(np.arange(3), vals2, yerr=errs2, fmt="none", ecolor=SECONDARY_INK,
                elinewidth=1.3, capsize=5)
    for i, v in enumerate(vals2):
        ax.text(i, v + max(errs2) + 0.008, f"{v:+.3f}", ha="center",
                color=SECONDARY_INK, fontsize=10, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names2, fontsize=8.5)
    ax.set_ylim(min(0, min(vals2)) - 0.03, max(vals2) + max(errs2) + 0.05)
    style(ax, "흔들림 1cm당 볼 비율", "%p")

    fig.suptitle("구종을 고정하면 발견 17·22는 살아남는가",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "pitchtype_recheck.png")
    plt.close(fig)

    path = DATA.parent / "eda_pitchtype_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
