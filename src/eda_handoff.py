"""불펜의 가치는 쉬어서인가 낯설어서인가 — 노출이 투수를 넘어 전이되는지.

발견 15·16은 '3바퀴째에 필승조가 있으면 바꿔라'로 끝났고, 그 이유를
'불펜은 쉬어서가 아니라 낯설어서 유리하다'고 적었다. 그건 해석이었다.

발견 26이 '타자는 구종을 따로 배운다'를 보여줬으니, 그 해석을 직접 검정할 수 있다.
구종 학습이 투수를 넘어 전이된다면 &mdash; 즉 선발에게서 슬라이더를 열 번 본 타자가
구원투수의 슬라이더에도 덜 속는다면 &mdash; 교체의 가치는 '선발이 남긴 노출 상태'에
달려 있다. 선발과 다른 구종을 던지는 구원투수가 더 낯설 것이다.

두 수준으로 본다.
  [A] 타석 단위 — 선발과 구종 배합이 다른 구원투수가 덜 맞는가
  [B] 투구 단위 — 타자가 '그 구종'을 선발에게서 본 횟수가 구원투수의 헛스윙을 깎는가

식별은 둘 다 <구원투수×시즌 고정효과>로 한다. 같은 구원투수가 날마다 다른
선발 뒤에 나오므로, 투수의 실력이 아니라 '선발과 얼마나 다른가'만 남는다.
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
AB_KINDS = ("안타", "낫아웃 출루", "삼진", "실책 출루", "야수선택", "인플레이 아웃")
SIG = {True: "유의함", False: "유의하지 않음"}
MIN_PITCHES = 200


def load(seasons):
    frames = []
    for s in seasons:
        path = DATA / f"naver_full_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["구종", "투구결과", "batter", "pitcher_code", "batting_order"])
    d["투수시즌"] = d["pitcher_code"].astype(str) + "_" + d["season"].astype(str)
    d = d.sort_values(["game_id", "inning", "투수_누적투구수"]).reset_index(drop=True)

    # 1회에 던진 투수 둘이 그 경기의 선발이다.
    starters = (d[d["inning"] == 1].groupby("game_id")["pitcher_code"]
                .apply(lambda s: set(s.dropna())).to_dict())
    d["is_starter"] = [c in starters.get(g, set())
                       for g, c in zip(d["game_id"], d["pitcher_code"])]

    # 타자마다 '자기가 상대한 선발'을 붙인다 — 양 팀 타순은 겹치지 않는다.
    mine = d[d["is_starter"]].groupby(["game_id", "batter"])["pitcher_code"].agg(
        lambda s: s.value_counts().idxmax())
    d = d.join(mine.rename("상대선발"), on=["game_id", "batter"])
    d = d[d["상대선발"].notna()].copy()

    d["스윙"] = d["투구결과"].isin(SWING)
    d["헛스윙"] = d["투구결과"].isin(WHIFF).astype(float)
    return d


def mix_table(d):
    """투수×시즌 구종 배합 벡터. 구원은 투구가 적으니 기준을 낮춘다."""
    cnt = d.groupby(["투수시즌", "구종"]).size().unstack(fill_value=0)
    tot = cnt.sum(axis=1)
    mix = cnt.div(tot, axis=0)
    return mix[tot >= MIN_PITCHES]


def exposures(d):
    """선발에게서 본 노출을 구원 투구에 붙인다."""
    # 선발 구간에서 타자별 누적 노출 (경기 전체 합)
    st = d[d["is_starter"]]
    by_type = st.groupby(["game_id", "batter", "구종"]).size().rename("선발노출")
    by_all = st.groupby(["game_id", "batter"]).size().rename("선발전체노출")

    rl = d[~d["is_starter"]].copy()
    rl = rl.join(by_type, on=["game_id", "batter", "구종"])
    rl = rl.join(by_all, on=["game_id", "batter"])
    rl["선발노출"] = rl["선발노출"].fillna(0.0)
    rl["선발전체노출"] = rl["선발전체노출"].fillna(0.0)
    # 구원투수 자신에 대한 노출 — 발견 26에서 쓴 것과 같은 방식
    rl = rl.sort_values(["game_id", "inning", "투수_누적투구수"])
    rl["구원노출"] = rl.groupby(["game_id", "pitcher_code", "batter", "구종"]).cumcount()
    rl["구원전체노출"] = rl.groupby(["game_id", "pitcher_code", "batter"]).cumcount()
    return rl


def fit(d, y, cols, fe="투수시즌", cluster="game_id"):
    X = pd.DataFrame(index=d.index)
    for c in cols:
        X[c] = d[c].astype(float)
    if "구종" in d.columns:      # 타석 단위로 모으면 구종은 사라진다
        X = X.join(pd.get_dummies(d["구종"], prefix="구종", drop_first=True).astype(float))
    X = X.join(pd.get_dummies(d[fe], prefix="F", drop_first=True).astype(float))
    X = sm.add_constant(X)
    return sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster]})


def grab(f, name):
    ci = f.conf_int().loc[name]
    return f.params[name], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    d = load(a.seasons)
    mix = mix_table(d)
    rl = exposures(d)

    L = [f"=== 교체 이득은 '낯섦'에서 오는가 ({len(a.seasons)}시즌) ===",
         f"전체 투구 {len(d):,}개 · 구원 투구 {len(rl):,}개 · "
         f"배합을 잴 수 있는 투수×시즌 {len(mix)}개 ({MIN_PITCHES}구 이상)\n"]

    # ── [B] 투구 단위 — 노출은 투수를 넘어 전이되는가 ───────────
    sw = rl[rl["스윙"]].copy()
    L.append("[B] 투구 단위 — 선발에게서 본 구종이 구원투수에게도 통하는가")
    L.append(f"  구원 스윙 {len(sw):,}개 · 구원투수×시즌 고정효과 · 경기 단위 군집")
    tab = sw.groupby(sw["선발노출"].clip(upper=8) // 2 * 2).agg(
        헛스윙률=("헛스윙", "mean"), n=("헛스윙", "size"))
    tab["se"] = np.sqrt(tab["헛스윙률"] * (1 - tab["헛스윙률"]) / tab["n"])
    for idx, r in tab.iterrows():
        lab = f"{int(idx)}~{int(idx)+1}개" if idx < 8 else "8개 이상"
        L.append(f"    선발에게서 이 구종 {lab}: 헛스윙 {r['헛스윙률']*100:.2f}% "
                 f"(±{r['se']*196:.2f}, n={int(r['n']):,})")
    cols = ["선발노출", "선발전체노출", "구원노출", "구원전체노출"]
    f = fit(sw, "헛스윙", cols)
    for c in cols:
        g = grab(f, c)
        L.append(f"    {c}: {g[0]*100:+.4f}%p/1개 [{g[1]*100:+.4f}, {g[2]*100:+.4f}] {SIG[g[3]]}")
    L.append("")

    # 선발과 같은 구종일 때만 전이가 일어나야 한다 — 배합 거리로 교차 확인
    L.append("  검정력")
    g = grab(f, "선발노출")
    se = (g[2] - g[0]) / 1.96
    L.append(f"    선발노출 표준오차 {se*100:.4f}%p · 관측 범위(0~12개) 검출 한계 "
             f"{1.96*se*12*100:.2f}%p (헛스윙률 평균 {sw['헛스윙'].mean()*100:.1f}%)")
    L.append("")

    # 위조 검정 — 구종 라벨을 엉뚱한 것으로 바꿔 노출을 찾으면 효과가 사라져야 한다.
    # 사라지지 않으면 내가 잰 것은 구종 학습이 아니라 '많이 본 타자' 효과다.
    L.append("  위조 검정 (구종 라벨을 섞어 노출을 조회)")
    by_type = (d[d["is_starter"]].groupby(["game_id", "batter", "구종"]).size()
               .rename("가짜노출"))
    types = sorted(d["구종"].unique())
    rng = np.random.default_rng(20261004)
    fake_hits = []
    for rep_i in range(5):
        shuffled = dict(zip(types, rng.permutation(types)))
        probe = sw.copy()
        probe["가짜구종"] = probe["구종"].map(shuffled)
        probe = probe.drop(columns=["가짜노출"], errors="ignore").join(
            by_type.rename_axis(["game_id", "batter", "가짜구종"]),
            on=["game_id", "batter", "가짜구종"])
        probe["가짜노출"] = probe["가짜노출"].fillna(0.0)
        gp = grab(fit(probe, "헛스윙",
                      ["가짜노출", "선발전체노출", "구원노출", "구원전체노출"]), "가짜노출")
        fake_hits.append(gp)
        L.append(f"    {rep_i+1}회: {gp[0]*100:+.4f}%p/1개 "
                 f"[{gp[1]*100:+.4f}, {gp[2]*100:+.4f}] {SIG[gp[3]]}")
    L.append(f"    → 위조 5회 중 유의 {sum(x[3] for x in fake_hits)}회 · "
             f"평균 계수 {np.mean([x[0] for x in fake_hits])*100:+.4f}%p "
             f"(실제 {g[0]*100:+.4f}%p)")
    L.append("")

    # ── [A] 타석 단위 — 배합이 다른 구원투수가 덜 맞는가 ────────
    L.append("[A] 타석 단위 — 선발과 배합이 먼 구원투수가 덜 맞는가")
    rl2 = rl.sort_values(["game_id", "inning", "투수_누적투구수"])
    rl2["pa_id"] = (rl2.groupby(["game_id", "pitcher_code"])["투구번호"]
                    .diff().fillna(1) <= 0).cumsum()
    pa = rl2.groupby(["game_id", "pitcher_code", "pa_id"], as_index=False).agg(
        season=("season", "first"), 투수시즌=("투수시즌", "first"),
        상대선발=("상대선발", "first"), batter=("batter", "first"),
        batting_order=("batting_order", "first"), inning=("inning", "first"),
        종류=("타석종류", "first"), 투구수=("투구번호", "size"),
        선발전체노출=("선발전체노출", "first"))
    pa = pa.dropna(subset=["종류"])
    # 구원투수의 1바퀴째만 — 발견 15와 같은 기준
    pa = pa.sort_values(["game_id", "pitcher_code", "inning"])
    pa["구원회전"] = pa.groupby(["game_id", "pitcher_code", "batter"]).cumcount() + 1
    pa = pa[pa["구원회전"] == 1].copy()
    pa["타수"] = pa["종류"].isin(AB_KINDS).astype(float)
    pa["안타"] = (pa["종류"] == "안타").astype(float)

    # 배합 거리 — 총변동거리. '몇 할의 공을 다시 배정해야 같아지나'로 읽힌다.
    pa["선발시즌"] = pa["상대선발"].astype(str) + "_" + pa["season"].astype(str)
    both = pa["투수시즌"].isin(mix.index) & pa["선발시즌"].isin(mix.index)
    pa = pa[both].copy()
    A = mix.loc[pa["투수시즌"]].to_numpy()
    B = mix.loc[pa["선발시즌"]].to_numpy()
    pa["배합거리"] = 0.5 * np.abs(A - B).sum(axis=1)

    ab = pa[pa["타수"] == 1].copy()
    L.append(f"  구원 1바퀴째 타수 {len(ab):,}건 · 구원투수×시즌 {ab['투수시즌'].nunique()}개 · "
             f"경기 {ab['game_id'].nunique():,}개")
    L.append(f"  배합거리 분포: 중앙 {ab['배합거리'].median():.3f} · "
             f"25~75% {ab['배합거리'].quantile(0.25):.3f}~{ab['배합거리'].quantile(0.75):.3f} · "
             f"범위 {ab['배합거리'].min():.3f}~{ab['배합거리'].max():.3f}")

    q = pd.qcut(ab["배합거리"], 4, labels=False)
    t2 = ab.groupby(q, observed=True).agg(피안타율=("안타", "mean"),
                                          거리=("배합거리", "median"), n=("안타", "size"))
    t2["se"] = np.sqrt(t2["피안타율"] * (1 - t2["피안타율"]) / t2["n"])
    for idx, r in t2.iterrows():
        L.append(f"    {int(idx)+1}분위 (거리 {r['거리']:.2f}): 피안타율 {r['피안타율']:.4f} "
                 f"(±{r['se']*1.96:.4f}, n={int(r['n']):,})")
    diff = t2["피안타율"].iloc[-1] - t2["피안타율"].iloc[0]
    sd = np.hypot(t2["se"].iloc[0], t2["se"].iloc[-1])
    L.append(f"    → 가장 먼 쪽 − 가장 가까운 쪽: {diff*1000:+.1f}/1000 "
             f"[{(diff-1.96*sd)*1000:+.1f}, {(diff+1.96*sd)*1000:+.1f}]")

    X = ["배합거리", "선발전체노출", "inning", "투구수"]
    for slot in range(2, 10):
        ab[f"타순_{slot}"] = (ab["batting_order"] == slot).astype(float)
    fa = fit(ab, "안타", X + [f"타순_{s}" for s in range(2, 10)])
    g = grab(fa, "배합거리")
    L.append(f"  회귀 (구원투수×시즌 고정효과 · 이닝·타순·선발노출 통제)")
    L.append(f"    배합거리 0.1당: {g[0]*0.1*1000:+.2f}/1000 "
             f"[{g[1]*0.1*1000:+.2f}, {g[2]*0.1*1000:+.2f}] {SIG[g[3]]}")
    se = (g[2] - g[0]) / 1.96 * 0.1 * 1000
    L.append(f"    표준오차 {se:.2f}/1000 (0.1당) · 관측 범위(0.2~0.9) 검출 한계 "
             f"{1.96*se*7:.1f}/1000")
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
    ax.set_xticklabels([f"{int(i)}~{int(i)+1}" if i < 8 else "8+" for i in tab.index],
                       fontsize=9)
    ax.set_ylim(0, tab["헛스윙률"].max() * 130)
    ax.set_title("선발에게서 이 구종을 본 횟수별\n구원투수 상대 헛스윙률", color=INK,
                 fontsize=11.5, loc="left", pad=10)
    ax.set_ylabel("헛스윙률 (%, 스윙 대상)", color=SECONDARY_INK, fontsize=9.5)
    ax.set_xlabel("선발에게서 본 같은 구종 (개)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    xx = np.arange(len(t2))
    ax.bar(xx, t2["피안타율"], color=ORANGE, width=0.58)
    ax.errorbar(xx, t2["피안타율"], yerr=t2["se"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, r in zip(xx, t2.itertuples()):
        ax.text(xi, r.피안타율 + r.se * 1.96 + 0.004, f"{r.피안타율:.3f}", ha="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{r.거리:.2f}" for r in t2.itertuples()], fontsize=9)
    ax.set_ylim(0, t2["피안타율"].max() * 1.35)
    ax.set_title("선발과 배합이 멀수록 덜 맞는가\n(구원 1바퀴째)", color=INK,
                 fontsize=11.5, loc="left", pad=10)
    ax.set_ylabel("피안타율", color=SECONDARY_INK, fontsize=9.5)
    ax.set_xlabel("선발과의 구종 배합 거리 (중앙값)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("불펜의 가치는 낯섦에서 오는가", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "handoff_novelty.png")
    plt.close(fig)

    path = RESULTS / "eda_handoff_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
