"""피안타율로 쓴 결론들을 실점으로 번역한다.

이 리포트의 모든 숫자가 피안타율 위에 서 있다. 그런데 피안타율은 삼진과
인플레이 아웃을 똑같은 '아웃'으로 묶고, 볼넷을 아예 세지 않는다.
교체 판단은 실점으로 해야 하므로 번역이 필요하다.

가중치는 우리 10시즌 데이터에서 직접 뽑았다(runvalue.py).
  안타 +0.669 / 볼넷·사구 +0.390 / 삼진 -0.314 / 인플레이 아웃 -0.316

마지막 두 줄이 핵심이다. 삼진과 인플레이 아웃은 득점가치가 사실상 같다
(차이 +0.0013점). 그래서 발견 28의 '맞히기 채널'은 피안타율은 올리지만
실점은 거의 올리지 않는다. 그 몫이 얼마나 줄어드는지가 이 분석의 질문이다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from kbo_client import TEAM_CODE

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
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

TIERS = ["추격조", "중간", "필승조"]
SIG = {True: "유의함", False: "유의하지 않음"}


def load(seasons):
    pa = pd.read_parquet(DATA / "kbo_pa_runvalue.parquet")
    pa = pa[pa["season"].isin(seasons)].copy()
    ok = set()
    for s in seasons:
        chk = DATA / f"kbo_state_check_{s}.parquet"
        if chk.exists():
            c = pd.read_parquet(chk)
            ok |= set(c.loc[c["score_ok"] & c["starter_ok"], "game_id"])
    pa = pa[pa["game_id"].isin(ok)].copy()
    away = pa["game_id"].str[8:10].map(TEAM_CODE)
    home = pa["game_id"].str[10:12].map(TEAM_CODE)
    pa["pitcher_team"] = np.where(pa["half"] == "초", home, away)
    pa["start"] = pa["game_id"].astype(str) + "_" + pa["pitcher"].astype(str)
    pa["안타"] = pa["안타"].astype(float)
    pa["타수"] = pa["타수"].astype(bool)
    return pa


def build_tiers(seasons):
    rows = []
    for s in seasons:
        path = DATA / f"kbo_pitcher_appearances_{s}.parquet"
        if not path.exists():
            continue
        box = pd.read_parquet(path)
        rel = box[~box["is_starter"]].copy()
        rel["결과"] = rel["결과"].fillna("")
        rel["성공"] = rel["결과"].str.contains("세|홀드", regex=True).astype(int)
        g = rel.groupby(["team", "선수명"])["성공"].sum().reset_index()
        g["season"] = s
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    t["등급"] = np.select([t["성공"] >= 15, t["성공"] >= 5],
                        ["필승조", "중간"], default="추격조")
    return t.rename(columns={"선수명": "pitcher", "team": "pitcher_team"})


def standardize(d, level, value, weights):
    """타순 구성을 공통 가중치로 맞춘 평균."""
    cells = d.groupby([level, "batting_order"], observed=True)[value].mean().unstack()
    cells = cells.reindex(columns=weights.index)
    w = cells.notna().mul(weights, axis=1)
    return (cells * weights).sum(axis=1) / w.sum(axis=1)


def slope(d, y, x):
    X = sm.add_constant(pd.DataFrame({x: d[x].astype(float)}, index=d.index))
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    f = sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})
    ci = f.conf_int().loc[x]
    return f.params[x], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2017, 2027)))
    a = p.parse_args()

    pa = load(a.seasons)
    w = pd.read_parquet(DATA / "linear_weights.parquet")["가중치"]
    slots = pa["batting_order"].value_counts(normalize=True)

    L = [f"=== 피안타율을 실점으로 번역하다 ({len(a.seasons)}시즌) ===",
         f"타석 {len(pa):,}건 · 선발 타석 {int(pa['is_starter'].sum()):,}건",
         f"가중치: 안타 {w['안타']:+.3f} · 볼넷·사구 {w['볼넷·사구']:+.3f} · "
         f"삼진 {w['삼진']:+.3f} · 인플레이 아웃 {w['인플레이 아웃']:+.3f}\n"]

    # ── [1] 발견 06을 실점으로 ─────────────────────────────────
    st = pa[pa["is_starter"]].copy()
    deep = st[st["타순회전"] >= 3]["start"].unique()
    tto = st[st["start"].isin(deep) & st["타순회전"].between(1, 3)].copy()
    L.append("[1] 발견 06 — 타순 회전 (3바퀴까지 간 등판, 타순 구성 표준화)")
    rv = standardize(tto, "타순회전", "득점가치", slots)
    avg = standardize(tto[tto["타수"]], "타순회전", "안타", slots)
    for t in (1, 2, 3):
        L.append(f"  {t}바퀴: 득점가치 {rv[t]:+.4f}점/타석 · 피안타율 {avg[t]:.4f}")
    d_rv, d_avg = rv[3] - rv[1], avg[3] - avg[1]
    L.append(f"  1→3바퀴: 득점가치 {d_rv*1000:+.1f}/1000점 · 피안타율 {d_avg*1000:+.1f}/1000")
    L.append(f"  → 9타자면 한 바퀴당 {d_rv/2*9:+.3f}점, 1→3바퀴 누적 {d_rv*9:+.3f}점\n")

    # ── [2] 발견 13·28을 실점으로 ──────────────────────────────
    st2 = st.copy()
    reach = st2[st2["투수_누적투구수"] >= 90]["start"].unique()
    pc = st2[st2["start"].isin(reach) & st2["투수_누적투구수"].between(1, 100)].copy()
    pc["투구수_10구당"] = pc["투수_누적투구수"] / 10.0
    L.append("[2] 발견 28 — 투구수 축 (90구 이상 간 등판)")
    s_rv = slope(pc, "득점가치", "투구수_10구당")
    s_avg = slope(pc[pc["타수"]], "안타", "투구수_10구당")
    L.append(f"  10구당 득점가치: {s_rv[0]*1000:+.3f}/1000점 "
             f"[{s_rv[1]*1000:+.3f}, {s_rv[2]*1000:+.3f}] {SIG[s_rv[3]]}")
    L.append(f"  10구당 피안타율 : {s_avg[0]*1000:+.3f}/1000 "
             f"[{s_avg[1]*1000:+.3f}, {s_avg[2]*1000:+.3f}] {SIG[s_avg[3]]}")
    L.append("")

    # 종류별 구성 변화로 득점가치 변화를 분해한다
    L.append("  득점가치 변화를 타석 종류별로 분해 (1~25구 → 76~100구)")
    pc["구간"] = pd.cut(pc["투수_누적투구수"], [0, 25, 50, 75, 100],
                      labels=["1~25구", "26~50구", "51~75구", "76~100구"])
    share = (pc.groupby(["구간", "종류"], observed=True).size()
             / pc.groupby("구간", observed=True).size()).unstack(fill_value=0.0)
    delta = share.loc["76~100구"] - share.loc["1~25구"]
    contrib = (delta * w.reindex(delta.index)).dropna().sort_values()
    for k, v in contrib.items():
        L.append(f"    {k:9s}: 비중 {delta[k]*100:+.3f}%p × 가중치 {w[k]:+.3f} "
                 f"= {v*1000:+.3f}/1000점")
    L.append(f"    합계: {contrib.sum()*1000:+.3f}/1000점")
    L.append("")

    # 분해는 '구간 차'이므로 같은 자로 잰 원값과 맞춰야 읽힌다.
    raw_rv = (pc.loc[pc["구간"] == "76~100구", "득점가치"].mean()
              - pc.loc[pc["구간"] == "1~25구", "득점가치"].mean())
    ab_pc = pc[pc["타수"]]
    raw_avg = (ab_pc.loc[ab_pc["구간"] == "76~100구", "안타"].mean()
               - ab_pc.loc[ab_pc["구간"] == "1~25구", "안타"].mean())
    L.append(f"    같은 구간 차의 실제값: 득점가치 {raw_rv*1000:+.3f}/1000점 · "
             f"피안타율 {raw_avg*1000:+.3f}/1000")
    L.append(f"    → 구성 변화로 설명되는 몫 {contrib.sum()*1000:+.3f}, 실제 {raw_rv*1000:+.3f} · "
             f"차이 {(raw_rv-contrib.sum())*1000:+.3f}는 같은 종류 안에서 "
             f"주자·아웃 상황이 달라진 몫이다(후반에는 주자가 적은 상황이 더 많다)")
    L.append("")

    # 발견 28은 '타수' 안에서 쟀고 이 분해는 '타석' 전체다. 볼넷이 들어오는지가 다르다.
    L.append("  발견 28과 분모가 다르다 — 볼넷 채널이 피안타율에는 없다")
    L.append(f"    타석 기준 비중 변화: 삼진 {delta.get('삼진',0)*100:+.3f}%p · "
             f"인플레이 아웃 {delta.get('인플레이 아웃',0)*100:+.3f}%p · "
             f"볼넷·사구 {delta.get('볼넷·사구',0)*100:+.3f}%p")
    ab_share = (ab_pc.groupby(["구간", "종류"], observed=True).size()
                / ab_pc.groupby("구간", observed=True).size()).unstack(fill_value=0.0)
    ab_delta = ab_share.loc["76~100구"] - ab_share.loc["1~25구"]
    L.append(f"    타수 기준 비중 변화: 삼진 {ab_delta.get('삼진',0)*100:+.3f}%p · "
             f"인플레이 아웃 {ab_delta.get('인플레이 아웃',0)*100:+.3f}%p "
             f"(볼넷은 타수에 안 들어간다)")
    L.append(f"    삼진과 인플레이 아웃의 득점가치 차이는 "
             f"{w['삼진']-w['인플레이 아웃']:+.4f}점에 불과하다 — 둘 사이의 이동은 "
             f"피안타율을 움직이지만 실점은 거의 못 움직인다. 실점을 움직이는 것은 "
             f"아웃이 <출루>로 바뀌는 쪽이다.")
    L.append("")

    # ── [3] 발견 15·16을 실점으로 ──────────────────────────────
    L.append("[3] 발견 15·16 — 5~8회 교체 이득 (실점 기준)")
    mid = pa[pa["inning"].between(5, 8)].copy()
    s3 = mid[mid["is_starter"] & (mid["타순회전"] >= 3)].copy()
    s3["역할"] = "선발 3바퀴 이상"
    rel = mid[~mid["is_starter"]].copy()
    tiers = build_tiers(a.seasons)
    rel = rel.merge(tiers, on=["season", "pitcher", "pitcher_team"], how="left")
    rel = rel.dropna(subset=["등급"])
    # 구원투수의 1바퀴째만 — 발견 15와 같은 기준
    rel = rel.sort_values(["game_id", "pitcher", "inning"])
    rel["구원회전"] = rel.groupby(["game_id", "pitcher", "batter"]).cumcount() + 1
    rel = rel[rel["구원회전"] == 1].copy()
    rel["역할"] = "구원 " + rel["등급"]

    both = pd.concat([s3, rel], ignore_index=True)
    rv2 = standardize(both, "역할", "득점가치", slots)
    avg2 = standardize(both[both["타수"]], "역할", "안타", slots)
    n2 = both.groupby("역할").size()
    base = "선발 3바퀴 이상"
    L.append(f"  {base}: 득점가치 {rv2[base]:+.4f}점 · 피안타율 {avg2[base]:.4f} "
             f"(n={n2[base]:,})")
    for t in TIERS:
        key = f"구원 {t}"
        if key not in rv2.index:
            continue
        g_rv = rv2[base] - rv2[key]
        g_avg = avg2[base] - avg2[key]
        L.append(f"  {key}로 교체: 득점가치 {rv2[key]:+.4f} · 피안타율 {avg2[key]:.4f} "
                 f"(n={n2[key]:,})")
        L.append(f"    → 이득 {g_rv*1000:+.1f}/1000점/타석 (피안타율 {g_avg*1000:+.1f}/1000) "
                 f"· 9타자 환산 {g_rv*9:+.3f}점")
    L.append("")

    # 두 자가 어긋나는 이유 — 피안타율이 못 보는 것
    L.append("  두 자가 어긋나는 이유 (역할별 타석 종류 비중, %)")
    comp = (both.groupby(["역할", "종류"], observed=True).size()
            / both.groupby("역할", observed=True).size()).unstack(fill_value=0.0) * 100
    keys = [base] + [f"구원 {t}" for t in TIERS if f"구원 {t}" in comp.index]
    cols = [c for c in ("안타", "볼넷·사구", "삼진", "인플레이 아웃") if c in comp.columns]
    L.append("    " + "역할".ljust(16) + "".join(f"{c:>12s}" for c in cols))
    for k in keys:
        L.append("    " + k.ljust(16) + "".join(f"{comp.loc[k, c]:>12.2f}" for c in cols))
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    order = w.sort_values().index
    vals = w.reindex(order)
    y = np.arange(len(order))
    ax.barh(y, vals, color=[BLUE if v < 0 else ORANGE for v in vals], height=0.6)
    for yi, (k, v) in enumerate(vals.items()):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                color=SECONDARY_INK, fontsize=9)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(vals.min() * 1.5, vals.max() * 1.45)
    ax.set_title("KBO 선형 가중치 (10시즌 자체 계산)", color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("타석당 득점가치 (점)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    x = np.arange(len(contrib))
    ax.bar(x, contrib.values * 1000, color=[BLUE if v < 0 else ORANGE for v in contrib.values],
           width=0.6)
    for xi, v in zip(x, contrib.values * 1000):
        ax.text(xi, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", color=SECONDARY_INK, fontsize=8.5)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(contrib.index, fontsize=8, rotation=30, ha="right")
    ax.set_title("투구수 축 득점가치 변화의 출처", color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("기여 (1/1000점)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[2]
    present = [t for t in TIERS if f"구원 {t}" in rv2.index]
    gains_rv = [(rv2[base] - rv2[f"구원 {t}"]) * 1000 for t in present]
    gains_avg = [(avg2[base] - avg2[f"구원 {t}"]) * 1000 for t in present]
    xx = np.arange(len(present))
    ax.bar(xx - 0.19, gains_avg, color=MUTED, width=0.36, label="피안타율 (1/1000)")
    ax.bar(xx + 0.19, gains_rv, color=VIOLET, width=0.36, label="득점가치 (1/1000점)")
    for xi, (va, vr) in enumerate(zip(gains_avg, gains_rv)):
        for off, v in ((-0.19, va), (0.19, vr)):
            ax.text(xi + off, v + (1.0 if v >= 0 else -1.0), f"{v:+.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", color=SECONDARY_INK, fontsize=9)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{t}로\n교체" for t in present], fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
    ax.set_title("교체 이득: 두 자로 재면", color=INK, fontsize=12, loc="left", pad=10)
    ax.set_ylabel("선발 3바퀴 대비 이득", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("피안타율을 실점으로 번역하다", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "runvalue.png")
    plt.close(fig)

    path = RESULTS / "eda_runvalue_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
