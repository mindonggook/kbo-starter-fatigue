"""남은 한계 07을 닫는다 — 필승조를 오늘 쓰는 값은 얼마인가.

발견 29는 '5~8회에 필승조로 바꾸면 9타자에 0.26점을 되찾는다'로 끝났다.
그런데 그 필승조는 내일 없거나 나빠진다. 리포트는 줄곧 일찍 내리라고 조언하면서
그 대가를 한 번도 세지 않았다.

득점가치가 생겼으니 셀 수 있다. 비용은 두 갈래다.

  성능 비용   연투하거나 최근 많이 던진 구원투수가 실제로 나빠지는가
  가용성 비용 오늘 쓰면 내일 못 쓰게 되는가

득점가치는 RE24라서 <물려받은 주자·아웃 상황을 이미 뺀> 값이다. 그래서 주자를
따로 통제할 필요가 없다 -- 구원투수를 평가할 때 늘 걸리던 문제가 여기서는 없다.

식별은 투수x시즌 고정효과다. 같은 투수가 어떤 날은 연투로, 어떤 날은 푹 쉬고
나오므로, '좋은 투수가 많이 쓰인다'는 역인과가 지워진다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import tiers as T

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
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

TIERS = ["추격조", "중간", "필승조"]
SIG = {True: "유의함", False: "유의하지 않음"}


def load_appearances(seasons):
    rows = []
    for s in seasons:
        path = DATA / f"kbo_pitcher_appearances_{s}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        d["season"] = s
        rows.append(d)
    app = pd.concat(rows, ignore_index=True)
    app["date"] = pd.to_datetime(app["date"].astype(str), format="%Y%m%d")
    app["투수키"] = (app["season"].astype(str) + "_" + app["team"].astype(str)
                  + "_" + app["선수명"].astype(str))
    return app


def workload(app):
    """구원 등판마다 직전 소모 상태를 붙인다."""
    rel = app[~app["is_starter"]].copy()
    rel = rel.sort_values(["투수키", "date"])
    g = rel.groupby("투수키")
    rel["직전등판일"] = g["date"].shift(1)
    rel["직전투구수"] = g["투구수"].shift(1)
    rel["휴식일"] = (rel["date"] - rel["직전등판일"]).dt.days
    rel["연투"] = (rel["휴식일"] == 1).astype(float)

    # 직전 3일·7일 투구수 -- 날짜 창으로 센다(등판 수가 아니라 투구 수로).
    out = []
    for key, grp in rel.groupby("투수키", sort=False):
        grp = grp.sort_values("date")
        idx = grp.set_index("date")["투구수"]
        prev3, prev7 = [], []
        for dt in grp["date"]:
            w3 = idx[(idx.index >= dt - pd.Timedelta(days=3)) & (idx.index < dt)]
            w7 = idx[(idx.index >= dt - pd.Timedelta(days=7)) & (idx.index < dt)]
            prev3.append(float(w3.sum()))
            prev7.append(float(w7.sum()))
        grp = grp.assign(직전3일투구=prev3, 직전7일투구=prev7)
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def tiers_of(seasons):
    """시점 기준 누적(최근 365일)으로 등급을 매긴다.

    같은 시즌 기록을 쓰면 결과 누수이고, 직전 시즌만 쓰면 신인이 통째로 빠진다.
    등판 날짜마다 그 직전까지의 누적을 세면 둘 다 피한다(결측 5.8%).
    """
    t = T.build_pit(seasons).dropna(subset=["등급"])
    return t.rename(columns={"pitcher_team": "team", "pitcher": "선수명"})[
        ["game_id", "season", "team", "선수명", "등급"]]


def within(d, cols, group):
    """고정효과를 더미 대신 집단 평균 차감으로 흡수한다(투수x시즌이 수천 개다)."""
    out = d[cols].astype(float).copy()
    return out - out.groupby(d[group].to_numpy()).transform("mean")


def fe_ols(d, y, xs, group, cluster):
    cols = [y] + list(xs)
    w = within(d, cols, group)
    f = sm.OLS(w[y], w[list(xs)]).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster]})
    return f


def grab(f, name, scale=1.0):
    ci = f.conf_int().loc[name]
    return f.params[name] * scale, ci[0] * scale, ci[1] * scale, (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2017, 2027)))
    a = p.parse_args()

    app = load_appearances(a.seasons)
    rel = workload(app)
    tiers = tiers_of(a.seasons)
    # 등급이 등판 단위이므로 game_id까지 넣어야 한다 — 빼면 다대다로 행이 폭증한다.
    before = len(rel)
    rel = rel.merge(tiers[["game_id", "season", "team", "선수명", "등급"]],
                    on=["game_id", "season", "team", "선수명"], how="left")
    assert len(rel) <= before * 1.01, f"등급 조인에서 행이 늘었다: {before} -> {len(rel)}"

    pa = pd.read_parquet(DATA / "kbo_pa_runvalue.parquet")
    pa = pa[pa["season"].isin(a.seasons) & ~pa["is_starter"]].copy()
    pa = pa.merge(
        rel[["game_id", "선수명", "등급", "휴식일", "연투", "직전투구수",
             "직전3일투구", "직전7일투구", "투수키", "opponent"]],
        left_on=["game_id", "pitcher"], right_on=["game_id", "선수명"], how="inner")
    pa = pa.dropna(subset=["휴식일", "득점가치"])
    pa = pa[pa["휴식일"].between(1, 14)].copy()
    pa["직전3일투구_10구당"] = pa["직전3일투구"] / 10.0
    pa["점수차"] = pa["점수차_투수팀기준"].astype(float)

    L = [f"=== 불펜 소모 비용 ({len(a.seasons)}시즌) ===",
         f"구원 등판 {len(rel):,}건 · 소모 상태를 붙인 구원 타석 {len(pa):,}건 "
         f"· 투수×시즌 {pa['투수키'].nunique():,}개",
         "득점가치는 RE24라 물려받은 주자·아웃 상황이 이미 빠져 있다.\n"]

    # ── [1] 연투는 얼마나 흔한가 ───────────────────────────────
    L.append("[1] 구원 등판의 휴식 상태")
    rr = rel.dropna(subset=["휴식일"])
    rr = rr[rr["휴식일"].between(1, 14)]
    for t in TIERS:
        sub = rr[rr["등급"] == t]
        if sub.empty:
            continue
        L.append(f"  {t}: 등판 {len(sub):,}건 · 연투 비율 {sub['연투'].mean()*100:.1f}% "
                 f"· 중앙 휴식 {sub['휴식일'].median():.0f}일 "
                 f"· 직전3일 투구 중앙 {sub['직전3일투구'].median():.0f}개")
    L.append("")

    # ── [2] 성능 비용 ──────────────────────────────────────────
    L.append("[2] 성능 비용 — 소모된 구원투수가 실제로 나빠지는가")
    L.append("    (투수×시즌 고정효과 · 이닝·점수차 통제 · 투수×시즌 군집)")
    pa["이닝수"] = pa["inning"].astype(float)
    XS = ["연투", "직전3일투구_10구당", "이닝수", "점수차"]
    for t in ["전체"] + TIERS:
        sub = pa if t == "전체" else pa[pa["등급"] == t]
        if len(sub) < 3000:
            continue
        f = fe_ols(sub, "득점가치", XS, "투수키", "투수키")
        g1 = grab(f, "연투", 1000)
        g2 = grab(f, "직전3일투구_10구당", 1000)
        L.append(f"  [{t}] 타석 {len(sub):,}건")
        L.append(f"    연투          : {g1[0]:+.2f}/1000점 [{g1[1]:+.2f}, {g1[2]:+.2f}] "
                 f"{SIG[g1[3]]}  (9타자 {g1[0]*9/1000:+.3f}점)")
        L.append(f"    직전3일 10구당 : {g2[0]:+.2f}/1000점 [{g2[1]:+.2f}, {g2[2]:+.2f}] "
                 f"{SIG[g2[3]]}")
    L.append("")

    # 휴식일 구간별로도 — 선형이 아닐 수 있다
    L.append("  휴식일 구간별 득점가치 (필승조, 자기 평균 대비)")
    wn = pa[pa["등급"] == "필승조"].copy()
    wn["dev"] = wn["득점가치"] - wn.groupby("투수키")["득점가치"].transform("mean")
    wn["휴식구간"] = pd.cut(wn["휴식일"], [0, 1, 2, 3, 5, 14],
                        labels=["연투(1일)", "2일", "3일", "4~5일", "6일 이상"])
    tb = wn.groupby("휴식구간", observed=True).agg(
        편차=("dev", "mean"), n=("dev", "size"), sd=("dev", "std"))
    tb["se"] = tb["sd"] / np.sqrt(tb["n"])
    for idx, r in tb.iterrows():
        L.append(f"    {idx}: {r['편차']*1000:+.2f}/1000점 (±{r['se']*1960:.2f}, "
                 f"n={int(r['n']):,})")
    L.append("")

    # ── [3] 가용성 비용 ────────────────────────────────────────
    L.append("[3] 가용성 비용 — 오늘 쓰면 내일 못 쓰는가")
    # 팀이 경기한 날짜 = 등판 기록에 나타난 (팀, 날짜)
    team_days = app[["season", "team", "date"]].drop_duplicates()
    team_days = team_days.sort_values(["season", "team", "date"])
    team_days["다음경기"] = team_days.groupby(["season", "team"])["date"].shift(-1)
    team_days["간격"] = (team_days["다음경기"] - team_days["date"]).dt.days

    relc = rel.merge(team_days, on=["season", "team", "date"], how="left")
    relc = relc[(relc["간격"] == 1)].copy()        # 다음날 바로 경기가 있는 경우만
    nxt = rel[["투수키", "date"]].assign(다음날_등판표시=1.0)
    relc = relc.merge(nxt.rename(columns={"date": "다음경기"}),
                      on=["투수키", "다음경기"], how="left")
    relc["다음날등판"] = relc["다음날_등판표시"].fillna(0.0)
    relc["투구수_구간"] = pd.cut(relc["투구수"], [0, 10, 20, 30, 100],
                            labels=["1~10구", "11~20구", "21~30구", "31구 이상"])
    L.append("  오늘 던진 투구수별, 다음날 등판 확률 (다음날 경기가 있는 경우만)")
    for t in TIERS:
        sub = relc[relc["등급"] == t]
        if sub.empty:
            continue
        g = sub.groupby("투구수_구간", observed=True).agg(
            확률=("다음날등판", "mean"), n=("다음날등판", "size"))
        L.append(f"    {t}: " + " / ".join(
            f"{idx} {r['확률']*100:.1f}%(n={int(r['n']):,})" for idx, r in g.iterrows()))
    base_rate = relc.groupby("등급")["다음날등판"].mean()
    L.append(f"  등급별 전체 다음날 등판 확률: " + " / ".join(
        f"{k} {v*100:.1f}%" for k, v in base_rate.items()))

    # 어제 던졌으면 오늘 등판 확률이 얼마나 떨어지나 (같은 투수 안에서)
    wn2 = relc[relc["등급"] == "필승조"].copy()
    wn2["dev"] = wn2["다음날등판"] - wn2.groupby("투수키")["다음날등판"].transform("mean")
    gg = wn2.groupby("투구수_구간", observed=True).agg(
        편차=("dev", "mean"), n=("dev", "size"), sd=("dev", "std"))
    gg["se"] = gg["sd"] / np.sqrt(gg["n"])
    L.append("  필승조, 자기 평균 대비 (투구수가 많을수록 다음날 빠지는가)")
    for idx, r in gg.iterrows():
        L.append(f"    {idx}: {r['편차']*100:+.2f}%p (±{r['se']*196:.2f}, n={int(r['n']):,})")
    L.append("")

    # ── [4] 종합 ───────────────────────────────────────────────
    L.append("[4] 종합 — 오늘 필승조를 쓰는 값")
    f = fe_ols(pa[pa["등급"] == "필승조"], "득점가치", XS, "투수키", "투수키")
    g1 = grab(f, "연투", 1000)
    perf = g1[0] * 9 / 1000
    hi = g1[2] * 9 / 1000
    L.append(f"  오늘의 이득 (발견 38 · 시점 기준 등급) : +0.092점/9타자")
    L.append(f"  연투 시 성능 손실            : {perf:+.3f}점/9타자 ({SIG[g1[3]]})")
    L.append(f"    → 검정력: 신뢰구간 상한이 {hi:+.3f}점이므로 "
             f"'오늘 이득의 {abs(hi)/0.092*100:.0f}%보다 큰 성능 비용'은 배제된다")
    L.append("")

    # 가용성 비용을 점수로 — 필승조가 빠지면 누가 대신 나오나로 환산한다.
    # 시점 기준 등급으로 다시 잰 교체 이득(발견 38).
    # 변천: 0.258/0.016/-0.359(누수) → 0.058/0.001/-0.108(직전 시즌) → 아래(시점 기준).
    GAIN = {"필승조": 0.092, "중간": -0.010, "추격조": -0.149}
    L.append("  가용성 비용을 점수로 환산 (뒤 문단의 가정 아래)")
    rate = relc[relc["등급"] == "필승조"].groupby("투구수_구간", observed=True)["다음날등판"].mean()
    ref = rate.iloc[0]
    for idx, v in rate.items():
        drop = ref - v
        for alt in ("중간", "추격조"):
            if alt == "중간":
                L.append(f"    {idx}: 다음날 등판 확률 {v*100:.1f}% "
                         f"(1~10구 대비 {-drop*100:+.1f}%p)")
            cost = drop * (GAIN["필승조"] - GAIN[alt])
            L.append(f"      대체가 {alt}라면 기대 비용 {cost:+.3f}점")
    L.append("    가정: 내일도 같은 교체 지점이 생기고, 필승조가 빠지면 그 자리를"
             " 중간 또는 추격조가 메운다. 확률 하락 전부가 '쓸 수 없음'이라고 본 것이므로"
             " 비용의 상한에 가깝다.")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    x = np.arange(len(tb))
    ax.bar(x, tb["편차"] * 1000, color=[ORANGE if v > 0 else BLUE for v in tb["편차"]],
           width=0.58)
    ax.errorbar(x, tb["편차"] * 1000, yerr=tb["se"] * 1960, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, r in zip(x, tb.itertuples()):
        ax.text(xi, r.편차 * 1000 + (r.se * 1960 + 1) * (1 if r.편차 >= 0 else -1),
                f"{r.편차*1000:+.1f}", ha="center",
                va="bottom" if r.편차 >= 0 else "top",
                color=SECONDARY_INK, fontsize=9)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(tb.index.astype(str), fontsize=8.5)
    ax.set_title("필승조, 휴식일별 득점가치", color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("자기 평균 대비 (1/1000점)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    xs2 = np.arange(len(gg))
    rates = relc[relc["등급"] == "필승조"].groupby("투구수_구간", observed=True)["다음날등판"].mean()
    ax.bar(xs2, rates * 100, color=VIOLET, width=0.58)
    for xi, v in zip(xs2, rates * 100):
        ax.text(xi, v + 0.7, f"{v:.1f}", ha="center", color=SECONDARY_INK, fontsize=9.5,
                fontweight="bold")
    ax.set_xticks(xs2)
    ax.set_xticklabels(rates.index.astype(str), fontsize=8.5)
    ax.set_ylim(0, rates.max() * 125)
    ax.set_title("오늘 던진 투구수별\n다음날 등판 확률 (필승조)", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_ylabel("다음날 등판 확률 (%)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[2]
    labels = ["오늘\n이득", "연투 시\n성능 손실"]
    vals = [0.258, perf]
    ax.bar(np.arange(2), vals, color=[BLUE, ORANGE], width=0.5)
    err = [0, (g1[0] - g1[1]) * 9 / 1000]
    ax.errorbar(np.arange(2), vals, yerr=err, fmt="none", ecolor=SECONDARY_INK,
                elinewidth=1.3, capsize=5)
    for xi, v in zip(range(2), vals):
        ax.text(xi, v + (0.012 if v >= 0 else -0.012), f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("필승조 교체의 값과 대가", color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("점 / 9타자", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("필승조를 오늘 쓰는 값은 얼마인가", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "bullpen_cost.png")
    plt.close(fig)

    path = RESULTS / "eda_bullpen_cost_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
