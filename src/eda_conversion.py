"""구위 축은 결과 축으로 얼마나 번역되는가 — 이 리포트가 세 번 멈춘 지점.

발견 18(구속 → 피안타율), 발견 13(헛스윙률), 발견 27(노출 전이)이 모두 같은 데서
멈췄다. 구위 지표는 분명히 움직이는데 피안타율이 따라오지 않는다.
'전환 효율이 낮다'고 적어 왔지만 그 효율을 잰 적은 없다.

정확히 가를 수 있다. 피안타율은 항등식으로 쪼개진다.

    피안타율 = (1 - 삼진율) x 인플레이 안타율

앞쪽이 <방망이에 맞히는가>, 뒤쪽이 <맞힌 공이 안타가 되는가>다.
헛스윙이 줄면 삼진율이 떨어져 앞쪽 경로로 피안타율을 올린다. 그 몫이 얼마이고,
나머지가 어디로 가는지 보면 전환 효율이 숫자로 나온다.

전미분으로 분해한다.
    Δ피안타율 = -BABIP x Δ삼진율  +  (1 - 삼진율) x ΔBABIP  (+ 잔차)
       (맞히기 채널)              (타구질 채널)
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
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

SWING = ("헛스윙", "파울", "타격", "번트파울", "번트헛스윙")
WHIFF = ("헛스윙", "번트헛스윙")
AB_KINDS = ("안타", "낫아웃 출루", "삼진", "실책 출루", "야수선택", "인플레이 아웃")
K_KINDS = ("삼진", "낫아웃 출루")
SIG = {True: "유의함", False: "유의하지 않음"}


def load(seasons):
    frames = []
    for s in seasons:
        path = DATA / f"naver_full_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["구종", "투구결과", "batter", "pitcher_code", "batting_order"])

    first = d[d["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    d = d[[c in starters.get(g, set())
           for g, c in zip(d["game_id"], d["pitcher_code"])]].copy()
    d["start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)
    d["투수시즌"] = d["pitcher_code"].astype(str) + "_" + d["season"].astype(str)

    d = d.sort_values(["start", "투수_누적투구수"]).reset_index(drop=True)
    d["스윙"] = d["투구결과"].isin(SWING).astype(float)
    d["헛스윙"] = d["투구결과"].isin(WHIFF).astype(float)

    # 타석 경계와, 타석 시작 시점의 노출
    d["pa_id"] = (d.groupby("start")["투구번호"].diff().fillna(1) <= 0).cumsum()
    d["타자노출"] = d.groupby(["start", "batter"]).cumcount()

    pa = d.groupby(["start", "pa_id"], as_index=False).agg(
        season=("season", "first"), 투수시즌=("투수시즌", "first"),
        batter=("batter", "first"), batting_order=("batting_order", "first"),
        inning=("inning", "first"), 종류=("타석종류", "first"),
        투구수=("투구번호", "size"), 시작노출=("타자노출", "min"),
        시작투구수=("투수_누적투구수", "min"),
        스윙수=("스윙", "sum"), 헛스윙수=("헛스윙", "sum"))
    pa = pa.dropna(subset=["종류"])
    pa["타수"] = pa["종류"].isin(AB_KINDS).astype(float)
    pa["안타"] = (pa["종류"] == "안타").astype(float)
    pa["삼진"] = pa["종류"].isin(K_KINDS).astype(float)
    pa["노출_10개당"] = pa["시작노출"] / 10.0
    pa["투구수_10구당"] = (pa["시작투구수"] - 1) / 10.0
    pa["타석헛스윙률"] = np.where(pa["스윙수"] > 0, pa["헛스윙수"] / pa["스윙수"], np.nan)
    return pa


def fit(d, y, treat, extra=()):
    X = pd.DataFrame(index=d.index)
    X[treat] = d[treat].astype(float)
    for c in extra:
        X[c] = d[c].astype(float)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = X.join(pd.get_dummies(d["투수시즌"], prefix="P", drop_first=True).astype(float))
    X = sm.add_constant(X)
    return sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})


def eff(d, y, treat, extra=()):
    f = fit(d, y, treat, extra)
    ci = f.conf_int().loc[treat]
    b = f.params[treat]
    return b, ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()
    pa = load(a.seasons)

    # 생존 편향 통제 — 앞선 발견들과 같은 기준
    deep = pa[pa["시작투구수"] >= 90]["start"].unique()
    pa = pa[pa["start"].isin(deep)].copy()
    ab = pa[pa["타수"] == 1].copy()
    bip = ab[ab["삼진"] == 0].copy()          # 인플레이 — 삼진이 아닌 타수

    K = ab["삼진"].mean()
    BABIP = bip["안타"].mean()
    AVG = ab["안타"].mean()

    L = [f"=== 구위 축에서 결과 축으로의 전환 효율 ({len(a.seasons)}시즌) ===",
         f"선발 타석 {len(pa):,}건 · 타수 {len(ab):,}건 · 인플레이 {len(bip):,}건 "
         f"(90구 이상 간 등판)",
         f"리그 수준: 피안타율 {AVG:.4f} · 삼진율 {K:.4f} · 인플레이 안타율 {BABIP:.4f}",
         f"항등식 점검: (1-{K:.4f}) x {BABIP:.4f} = {(1-K)*BABIP:.4f} vs 실제 {AVG:.4f}\n"]

    TREATS = (("노출_10개당", "타자 노출 10개당", ()),
              ("투구수_10구당", "투수 투구수 10구당", ()))

    for treat, label, extra in TREATS:
        L.append(f"[{label}]")
        w = eff(ab[ab["스윙수"] > 0], "타석헛스윙률", treat, extra)
        k = eff(ab, "삼진", treat, extra)
        h = eff(ab, "안타", treat, extra)
        b = eff(bip, "안타", treat, extra)
        L.append(f"  헛스윙률    : {w[0]*100:+.4f}%p [{w[1]*100:+.4f}, {w[2]*100:+.4f}] {SIG[w[3]]}")
        L.append(f"  삼진율      : {k[0]*1000:+.3f}/1000 [{k[1]*1000:+.3f}, {k[2]*1000:+.3f}] {SIG[k[3]]}")
        L.append(f"  인플레이안타율: {b[0]*1000:+.3f}/1000 [{b[1]*1000:+.3f}, {b[2]*1000:+.3f}] {SIG[b[3]]}")
        L.append(f"  피안타율    : {h[0]*1000:+.3f}/1000 [{h[1]*1000:+.3f}, {h[2]*1000:+.3f}] {SIG[h[3]]}")

        # 전미분 분해
        ch_contact = -BABIP * k[0]
        ch_quality = (1 - K) * b[0]
        total = h[0]
        resid = total - ch_contact - ch_quality
        L.append(f"  분해: 맞히기 채널 {ch_contact*1000:+.3f}/1000 "
                 f"({ch_contact/total*100:.1f}%) · 타구질 채널 {ch_quality*1000:+.3f} "
                 f"({ch_quality/total*100:.1f}%) · 교차·잔차 {resid*1000:+.3f} "
                 f"({resid/total*100:.1f}%)")

        # 전환 효율 — 헛스윙률 1%p 감소가 피안타율 몇 포인트로 바뀌는가
        if w[0]:
            per = h[0] / (-w[0]) * 1000 / 100
            L.append(f"  전환 효율: 헛스윙률 1%p 감소당 피안타율 {per:+.2f}/1000 "
                     f"(맞히기 채널만 {ch_contact/(-w[0])*1000/100:+.2f})")
        L.append("")

    # ── 헛스윙률과 삼진율의 연결 강도 ──────────────────────────
    # 맞히기 채널이 작다면, 헛스윙이 줄어도 삼진이 안 줄어든다는 뜻이다. 확인한다.
    L.append("[연결 강도] 헛스윙률이 삼진율로 얼마나 전달되나")
    sw = ab[ab["스윙수"] > 0].copy()
    # 헛스윙률은 0에 몰려 있어 4분위가 두 개로 붕괴한다 — 값으로 끊는다.
    sw["헛스윙_구간"] = pd.cut(sw["타석헛스윙률"], [-0.01, 0.0, 0.34, 0.5, 1.01],
                            labels=["0%", "0~33%", "34~50%", "50% 초과"])
    t = sw.groupby("헛스윙_구간", observed=True).agg(
        헛스윙률=("타석헛스윙률", "mean"), 삼진율=("삼진", "mean"),
        피안타율=("안타", "mean"), n=("안타", "size"))
    for idx, r in t.iterrows():
        L.append(f"  {idx} (평균 {r['헛스윙률']*100:.1f}%): "
                 f"삼진율 {r['삼진율']:.4f} · 피안타율 {r['피안타율']:.4f} (n={int(r['n']):,})")
    L.append("  (같은 타석 안의 값이므로 인과가 아니라 연결 강도의 상한이다)\n")

    # ── 발견 13과 부호가 어긋나는 곳을 맞춘다 ──────────────────
    # 발견 13은 '투구수가 쌓여도 헛스윙률은 거의 안 변한다(+0.31%p)'였는데
    # 여기서는 10구당 -0.20%p로 유의하게 줄었다. 분모가 다르다.
    L.append("[대조] 발견 13과의 부호 차이 — 헛스윙률의 분모")
    base = pa.copy()
    base["구간"] = pd.cut(base["시작투구수"], [0, 25, 50, 75, 100],
                        labels=["1~25구", "26~50구", "51~75구", "76~100구"])
    g = base.dropna(subset=["구간"]).groupby("구간", observed=True).agg(
        투구당=("헛스윙수", "sum"), 스윙수=("스윙수", "sum"),
        투구수=("투구수", "sum"))
    g["투구당_헛스윙률"] = g["투구당"] / g["투구수"] * 100
    g["스윙당_헛스윙률"] = g["투구당"] / g["스윙수"] * 100
    g["스윙률"] = g["스윙수"] / g["투구수"] * 100
    for idx, r in g.iterrows():
        L.append(f"  {idx}: 투구당 {r['투구당_헛스윙률']:.2f}% · "
                 f"스윙당 {r['스윙당_헛스윙률']:.2f}% · 스윙률 {r['스윙률']:.2f}%")
    f0, f3 = g.index[0], g.index[-1]
    L.append(f"  → 1~25구 대비 76~100구: 투구당 "
             f"{g.loc[f3,'투구당_헛스윙률']-g.loc[f0,'투구당_헛스윙률']:+.2f}%p · "
             f"스윙당 {g.loc[f3,'스윙당_헛스윙률']-g.loc[f0,'스윙당_헛스윙률']:+.2f}%p · "
             f"스윙률 {g.loc[f3,'스윙률']-g.loc[f0,'스윙률']:+.2f}%p")
    L.append("")

    # ── 왜 끊기는가: 인플레이 안타율은 거의 안 움직인다 ────────
    L.append("[진단] 두 채널의 움직임 폭 비교")
    for treat, label, extra in TREATS:
        k = eff(ab, "삼진", treat, extra)
        b = eff(bip, "안타", treat, extra)
        L.append(f"  {label}: 삼진율 {k[0]*1000:+.3f}/1000 ({SIG[k[3]]}) · "
                 f"인플레이안타율 {b[0]*1000:+.3f}/1000 ({SIG[b[3]]})")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    names, contact, quality, totals = [], [], [], []
    for treat, label, extra in TREATS:
        k = eff(ab, "삼진", treat, extra)
        b = eff(bip, "안타", treat, extra)
        h = eff(ab, "안타", treat, extra)
        names.append(label.replace(" ", "\n", 1))
        contact.append(-BABIP * k[0] * 1000)
        quality.append((1 - K) * b[0] * 1000)
        totals.append(h[0] * 1000)
    x = np.arange(len(names))
    ax.bar(x - 0.21, contact, color=BLUE, width=0.2, label="맞히기 채널")
    ax.bar(x, quality, color=ORANGE, width=0.2, label="타구질 채널")
    ax.bar(x + 0.21, totals, color=MUTED, width=0.2, label="총효과")
    for xi, (c, qv, tv) in enumerate(zip(contact, quality, totals)):
        for off, v in ((-0.21, c), (0.0, qv), (0.21, tv)):
            ax.text(xi + off, v + (0.05 if v >= 0 else -0.05), f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", color=SECONDARY_INK, fontsize=8.5)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
    ax.set_title("피안타율 변화를 두 채널로 가르면", color=INK, fontsize=12.5,
                 loc="left", pad=10)
    ax.set_ylabel("피안타율 변화 (1/1000)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    xx = np.arange(len(t))
    ax.bar(xx - 0.19, t["삼진율"], color=BLUE, width=0.36, label="삼진율")
    ax.bar(xx + 0.19, t["피안타율"], color=ORANGE, width=0.36, label="피안타율")
    for xi, r in zip(xx, t.itertuples()):
        ax.text(xi - 0.19, r.삼진율 + 0.006, f"{r.삼진율:.3f}", ha="center",
                color=SECONDARY_INK, fontsize=8.5)
        ax.text(xi + 0.19, r.피안타율 + 0.006, f"{r.피안타율:.3f}", ha="center",
                color=SECONDARY_INK, fontsize=8.5)
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{r.헛스윙률*100:.0f}%" for r in t.itertuples()], fontsize=9)
    ax.set_ylim(0, max(t["삼진율"].max(), t["피안타율"].max()) * 1.35)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
    ax.set_title("타석 헛스윙률과 두 결과 지표", color=INK, fontsize=12.5,
                 loc="left", pad=10)
    ax.set_xlabel("타석 안 헛스윙률 4분위", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("구위는 결과로 얼마나 번역되는가", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "conversion.png")
    plt.close(fig)

    path = RESULTS / "eda_conversion_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
