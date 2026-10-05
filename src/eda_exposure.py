"""적응 메커니즘을 검정력 있는 설계로 다시 — 타자는 '그 구종'을 배우는가.

레퍼토리 검정(발견 25 전반부)은 기각도 확인도 못 했다. 레퍼토리는 투수마다
고정이라 교차항이 사실상 투수 146명의 비교가 되고, 신뢰구간이 페널티 전체와
맞먹게 넓어졌기 때문이다.

같은 질문을 투구 단위로 바꾸면 표본이 20,008타수에서 7만 투구가 된다.
  '타자가 이 경기에서 <이 구종>을 이미 몇 번 봤는가'
적응이 구종 학습이라면, 같은 구종을 반복해 볼수록 헛스윙이 줄고 맞혀내야 한다.

핵심은 <전체 노출>을 함께 통제하는 것이다. 그래야 '그 투수에게 익숙해진 것'과
'그 구종에 익숙해진 것'이 갈린다. 둘은 함께 늘지만 배합이 투수마다 달라
비율이 흔들리므로 식별된다.
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

SWING = ("헛스윙", "파울", "타격", "번트파울", "번트헛스윙")
WHIFF = ("헛스윙", "번트헛스윙")
SIG = {True: "유의함", False: "유의하지 않음"}


def load(seasons):
    frames = []
    for s in seasons:
        path = DATA / f"naver_full_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["구종", "투구결과", "batter", "pitcher_code"])

    first = d[d["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    d = d[[c in starters.get(g, set())
           for g, c in zip(d["game_id"], d["pitcher_code"])]].copy()
    d["start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)
    d["투수시즌"] = d["pitcher_code"].astype(str) + "_" + d["season"].astype(str)

    d = d.sort_values(["start", "투수_누적투구수"]).reset_index(drop=True)
    # 이 타자가 이 경기에서 이 투수의 공을, 그리고 이 구종을, 앞서 몇 개 봤는가
    d["전체노출"] = d.groupby(["start", "batter"]).cumcount()
    d["구종노출"] = d.groupby(["start", "batter", "구종"]).cumcount()
    d["회전"] = d.groupby(["start", "batter", "투구번호"]).cumcount() + 1
    d.loc[d["투구번호"] != 1, "회전"] = np.nan
    d["회전"] = d.groupby(["start", "batter"])["회전"].ffill().fillna(1)

    d["스윙"] = d["투구결과"].isin(SWING)
    d["헛스윙"] = d["투구결과"].isin(WHIFF).astype(float)
    d["파울"] = d["투구결과"].isin(("파울", "번트파울")).astype(float)
    return d


def fit(d, y, cols, fe=True):
    X = pd.DataFrame(index=d.index)
    for c in cols:
        X[c] = d[c].astype(float)
    X = X.join(pd.get_dummies(d["구종"], prefix="구종", drop_first=True).astype(float))
    if fe:
        X = X.join(pd.get_dummies(d["투수시즌"], prefix="P", drop_first=True).astype(float))
    X = sm.add_constant(X)
    return sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})


def grab(f, name):
    ci = f.conf_int().loc[name]
    return f.params[name], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()
    d = load(a.seasons)
    sw = d[d["스윙"]].copy()

    L = [f"=== 구종 노출과 적응 ({len(a.seasons)}시즌) ===",
         f"선발 투구 {len(d):,}개 · 스윙 {len(sw):,}개 · 등판 {d['start'].nunique():,}개",
         "(구종 더미 · 투수×시즌 고정효과 · 등판 단위 군집 보정)\n"]

    L.append("[1] 같은 구종을 다시 볼 때 헛스윙률 (스윙 대상)")
    tab = sw.groupby(sw["구종노출"].clip(upper=4)).agg(
        헛스윙률=("헛스윙", "mean"), 파울률=("파울", "mean"), n=("헛스윙", "size"))
    tab["se"] = np.sqrt(tab["헛스윙률"] * (1 - tab["헛스윙률"]) / tab["n"])
    for idx, r in tab.iterrows():
        lab = f"{int(idx)}번째" if idx < 4 else "5번째 이상"
        L.append(f"  이 구종 {lab}: 헛스윙 {r['헛스윙률']*100:.2f}% (±{r['se']*196:.2f}) · "
                 f"파울 {r['파울률']*100:.2f}% (n={int(r['n']):,})")
    L.append("")

    L.append("[2] 회귀 — 구종 노출과 전체 노출을 함께")
    for y, ylab in (("헛스윙", "헛스윙률"), ("파울", "파울 비율")):
        only = grab(fit(sw, y, ["구종노출"]), "구종노출")
        both = fit(sw, y, ["구종노출", "전체노출"])
        g1, g2 = grab(both, "구종노출"), grab(both, "전체노출")
        tto = fit(sw, y, ["구종노출", "전체노출", "회전"])
        h1, h2, h3 = (grab(tto, "구종노출"), grab(tto, "전체노출"), grab(tto, "회전"))
        L.append(f"  [{ylab}]")
        L.append(f"    구종노출만            : {only[0]*100:+.4f}%p/1회 "
                 f"[{only[1]*100:+.4f}, {only[2]*100:+.4f}] {SIG[only[3]]}")
        L.append(f"    + 전체노출 통제 · 구종 : {g1[0]*100:+.4f}%p/1회 "
                 f"[{g1[1]*100:+.4f}, {g1[2]*100:+.4f}] {SIG[g1[3]]}")
        L.append(f"    + 전체노출 통제 · 전체 : {g2[0]*100:+.4f}%p/1구 "
                 f"[{g2[1]*100:+.4f}, {g2[2]*100:+.4f}] {SIG[g2[3]]}")
        L.append(f"    + 회전까지 통제 · 구종 : {h1[0]*100:+.4f}%p/1회 "
                 f"[{h1[1]*100:+.4f}, {h1[2]*100:+.4f}] {SIG[h1[3]]}")
        L.append(f"    + 회전까지 통제 · 전체 : {h2[0]*100:+.4f}%p/1구 "
                 f"[{h2[1]*100:+.4f}, {h2[2]*100:+.4f}] {SIG[h2[3]]}")
        L.append(f"    + 회전까지 통제 · 회전 : {h3[0]*100:+.4f}%p/1바퀴 "
                 f"[{h3[1]*100:+.4f}, {h3[2]*100:+.4f}] {SIG[h3[3]]}")
    L.append("")

    # ── [3] 바퀴를 고정하고 구종 노출만 ────────────────────────
    # 같은 3바퀴째 타석 안에서도 타자마다 그 구종을 본 횟수가 다르다.
    L.append("[3] 회전을 고정하고 구종 노출만 (스윙 · 헛스윙률)")
    for t in (1, 2, 3):
        sub = sw[sw["회전"] == t]
        if len(sub) < 2000:
            continue
        g = grab(fit(sub, "헛스윙", ["구종노출", "전체노출"]), "구종노출")
        L.append(f"  {t}바퀴: {g[0]*100:+.4f}%p/1회 [{g[1]*100:+.4f}, {g[2]*100:+.4f}] "
                 f"{SIG[g[3]]} (스윙 {len(sub):,})")
    L.append("")

    # ── [4] 검정력 ─────────────────────────────────────────────
    g = grab(fit(sw, "헛스윙", ["구종노출", "전체노출"]), "구종노출")
    se = (g[2] - g[0]) / 1.96
    L.append("[4] 검정력 — 무엇을 배제할 수 있나")
    L.append(f"  구종노출 계수 표준오차: {se*100:.4f}%p/1회")
    L.append(f"  관측 범위(0~8회)에서 검출 한계: {1.96*se*8*100:.2f}%p "
             f"(헛스윙률 평균 {sw['헛스윙'].mean()*100:.1f}% 대비 "
             f"{1.96*se*8/sw['헛스윙'].mean()*100:.1f}%)")
    L.append("")

    # ── [5] 생존 편향과 군집 단위 ──────────────────────────────
    # 노출이 많이 쌓인 투구는 '오래 버틴 등판'에서 더 많이 나온다. 잘 던진 날에
    # 노출이 쌓이므로 이 편향은 학습 효과를 0 쪽으로 끌어당긴다 — 보수적이다.
    # 다만 회전 계수는 이 선택에 직접 노출되므로 따로 확인한다.
    L.append("[5] 생존 편향과 군집 단위 점검 (헛스윙률)")
    deep = d[d["투수_누적투구수"] >= 90]["start"].unique()
    sub = sw[sw["start"].isin(deep)]
    f = fit(sub, "헛스윙", ["구종노출", "전체노출", "회전"])
    for nm in ("구종노출", "전체노출", "회전"):
        g = grab(f, nm)
        L.append(f"  90구 이상 등판만 · {nm}: {g[0]*100:+.4f}%p "
                 f"[{g[1]*100:+.4f}, {g[2]*100:+.4f}] {SIG[g[3]]}")
    L.append(f"  (등판 {sub['start'].nunique():,}개 · 스윙 {len(sub):,}개)")

    X = pd.DataFrame({c: sw[c].astype(float) for c in ("구종노출", "전체노출", "회전")})
    X = X.join(pd.get_dummies(sw["구종"], prefix="구종", drop_first=True).astype(float))
    X = X.join(pd.get_dummies(sw["투수시즌"], prefix="P", drop_first=True).astype(float))
    fp = sm.OLS(sw["헛스윙"], sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": sw["투수시즌"]})
    for nm in ("구종노출", "전체노출", "회전"):
        g = grab(fp, nm)
        L.append(f"  투수×시즌 군집 · {nm}: {g[0]*100:+.4f}%p "
                 f"[{g[1]*100:+.4f}, {g[2]*100:+.4f}] {SIG[g[3]]}")
    L.append("")

    # ── [6] 회전 계수의 부호 반전은 믿을 수 있나 ───────────────
    # 발견 08에서 타순 회전과 누적 투구수를 같이 넣었을 때 계수가 8.6배로 부풀고
    # 부호가 뒤집혔다. 여기서도 회전과 전체노출은 거의 같은 것을 센다.
    L.append("[6] 회전 계수 반전의 신뢰성 (발견 08과 같은 함정인가)")
    L.append(f"  회전 ↔ 전체노출 상관: {sw['회전'].corr(sw['전체노출']):+.3f}")
    solo = grab(fit(sw, "헛스윙", ["회전"]), "회전")
    L.append(f"  회전만 넣으면      : {solo[0]*100:+.4f}%p/1바퀴 "
             f"[{solo[1]*100:+.4f}, {solo[2]*100:+.4f}] {SIG[solo[3]]}")
    g = grab(fit(sw, "헛스윙", ["회전", "전체노출"]), "회전")
    L.append(f"  전체노출과 함께     : {g[0]*100:+.4f}%p/1바퀴 "
             f"[{g[1]*100:+.4f}, {g[2]*100:+.4f}] {SIG[g[3]]} "
             f"→ {abs(g[0]/solo[0]) if solo[0] else float('nan'):.1f}배, "
             f"부호 {'반전' if (g[0] > 0) != (solo[0] > 0) else '유지'}")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    x = np.arange(len(tab))
    ax.bar(x, tab["헛스윙률"] * 100, color=BLUE, width=0.58)
    ax.errorbar(x, tab["헛스윙률"] * 100, yerr=tab["se"] * 196, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, r in zip(x, tab.itertuples()):
        ax.text(xi, r.헛스윙률 * 100 + r.se * 196 + 0.4, f"{r.헛스윙률*100:.1f}",
                ha="center", color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}번째" if i < 4 else "5번째+" for i in tab.index], fontsize=9)
    ax.set_ylim(0, tab["헛스윙률"].max() * 130)
    ax.set_title("같은 구종을 다시 볼 때 헛스윙률", color=INK, fontsize=12.5,
                 loc="left", pad=10)
    ax.set_ylabel("헛스윙률 (%, 스윙 대상)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    both = fit(sw, "헛스윙", ["구종노출", "전체노출", "회전"])
    names = ["이 구종을\n한 번 더", "이 투수 공을\n한 개 더", "타순이\n한 바퀴"]
    picks = [grab(both, "구종노출"), grab(both, "전체노출"), grab(both, "회전")]
    vals = [g[0] * 100 for g in picks]
    errs = [(g[0] - g[1]) * 100 for g in picks]
    xx = np.arange(3)
    ax.bar(xx, vals, color=[BLUE if v < 0 else ORANGE for v in vals], width=0.5)
    ax.errorbar(xx, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK,
                elinewidth=1.3, capsize=5)
    for xi, v, e in zip(xx, vals, errs):
        ax.text(xi, v + (e + 0.03) * (1 if v >= 0 else -1), f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(xx)
    ax.set_xticklabels(names, fontsize=8.5)
    span = max(abs(v) + e for v, e in zip(vals, errs)) * 1.6
    ax.set_ylim(-span, span)
    ax.set_title("헛스윙률을 움직이는 것은 어느 노출인가", color=INK, fontsize=12.5,
                 loc="left", pad=10)
    ax.set_ylabel("헛스윙률 변화 (%p)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("타자는 구종을 배우는가, 투수를 배우는가", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "exposure_learning.png")
    plt.close(fig)

    path = RESULTS / "eda_exposure_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
