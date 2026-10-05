"""교체 판단을 승리확률로 다시 — 1차 평가 ④를 끝까지 닫는다.

평가는 '실점이 아니라 승리확률로 재야 실전 조언이 된다'고 했고, 나는 접전만
따로 보는 것으로 대신했다. 승리기대값 표를 만들었으니 이제 제대로 답할 수 있다.

WPA는 득점가치와 두 가지가 다르다.
  - 같은 1점이라도 점수차에 따라 값이 10배 넘게 다르다(5~8회: 동점 13.6%p vs 5점차 1.3%p).
  - 그래서 '추격조는 크게 지고 있을 때 나온다'는 사실이 자동으로 반영된다.
    실점으로 재면 그 등판의 손해가 그대로 잡히지만, 승률로 재면 거의 0이 된다.

등급은 시점 기준 누적(발견 38)을 쓴다. 신뢰구간은 경기 단위 군집보정.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import tiers as T
from kbo_client import TEAM_CODE

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
BASE = "선발 3바퀴 이상"


def load(seasons):
    pa = pd.read_parquet(DATA / "kbo_pa_wpa.parquet")
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
    return pa


def roles(pa, tier_df):
    mid = pa[pa["inning"].between(5, 8)].copy()
    s3 = mid[mid["is_starter"] & (mid["타순회전"] >= 3)].copy()
    s3["역할"] = BASE
    rel = mid[~mid["is_starter"]].merge(
        tier_df, on=["game_id", "season", "pitcher_team", "pitcher"], how="left")
    rel = rel.dropna(subset=["등급"])
    rel = rel.sort_values(["game_id", "pitcher", "inning"])
    rel["구원회전"] = rel.groupby(["game_id", "pitcher", "batter"]).cumcount() + 1
    rel = rel[rel["구원회전"] == 1].copy()
    rel["역할"] = "구원 " + rel["등급"].astype(str)
    return pd.concat([s3, rel], ignore_index=True)


def fit(d, y="WPA"):
    X = pd.DataFrame(index=d.index)
    cols = []
    for t in T.TIERS:
        key = f"구원 {t}"
        if (d["역할"] == key).any():
            X[key] = (d["역할"] == key).astype(float)
            cols.append(key)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = sm.add_constant(X)
    f = sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["game_id"]})
    return f, cols


def gain(f, key, scale=900):
    """이득 = 선발 − 구원 = −계수. 9타자 환산 후 %p로."""
    ci = f.conf_int().loc[key]
    return -f.params[key] * scale, -ci[1] * scale, -ci[0] * scale, (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2018, 2027)))
    a = p.parse_args()

    pa = load(a.seasons)
    tier_df = T.build_pit(a.seasons).dropna(subset=["등급"])

    L = ["=== 교체 이득을 승리확률로 (9시즌) ===",
         "5~8회 · 선발 3바퀴 이상 대비 구원 1바퀴째 · 시점 기준 등급",
         "단위: 9타자당 승리확률 %p · 경기 단위 군집보정\n"]

    cuts = [("전체", pa),
            ("접전 (2점 이내)", pa[pa["점수차_투수팀기준"].abs() <= 2]),
            ("한 점 차", pa[pa["점수차_투수팀기준"].abs() <= 1]),
            ("5점 이상 차", pa[pa["점수차_투수팀기준"].abs() >= 5])]

    store = {}
    for lab, sub in cuts:
        both = roles(sub, tier_df)
        f, cols = fit(both)
        n = both.groupby("역할").size()
        store[lab] = (f, cols, n)
        L.append(f"[{lab}]")
        for c in cols:
            g = gain(f, c)
            L.append(f"  {c:10s} {g[0]:+.3f}%p [{g[1]:+.3f}, {g[2]:+.3f}] {SIG[g[3]]} "
                     f"(n={n.get(c, 0):,})")
        L.append("")

    # 실점과 승률이 갈리는 지점
    L.append("[대조] 같은 비교를 두 자로 (전체 · 9타자 환산)")
    rv = pd.read_parquet(DATA / "kbo_pa_runvalue.parquet",
                         columns=["game_id", "inning", "half", "batting_order",
                                  "batter", "득점가치"])
    both_all = roles(pa, tier_df).merge(
        rv, on=["game_id", "inning", "half", "batting_order", "batter"], how="inner")
    f_rv, cols_rv = fit(both_all, "득점가치")
    f_wp, _ = fit(both_all, "WPA")
    L.append("  " + "교체 대상".ljust(12) + f"{'실점(점)':>14s}{'승률(%p)':>16s}")
    for c in cols_rv:
        g_rv = gain(f_rv, c, scale=9)
        g_wp = gain(f_wp, c)
        L.append("  " + c.replace("구원 ", "").ljust(12)
                 + f"{g_rv[0]:>+10.3f}{'*' if g_rv[3] else ' ':>2s}"
                 + f"{g_wp[0]:>+12.3f}{'*' if g_wp[3] else ' ':>2s}")
    L.append("  (* 유의)")
    L.append("")

    # 역할마다 '등판 상황의 무게'가 다르면 WPA 평균을 그대로 맞대면 안 된다.
    L.append("[레버리지] 역할별로 등판 상황의 무게가 같은가")
    both_all["|WPA|"] = both_all["WPA"].abs()
    g = both_all.groupby("역할").agg(
        평균_절대WPA=("|WPA|", "mean"), 중앙_점수차=("점수차_투수팀기준",
                                              lambda x: x.abs().median()),
        WPA_표준편차=("WPA", "std"), n=("WPA", "size"))
    ref = g.loc[BASE, "평균_절대WPA"]
    for idx, r in g.iterrows():
        L.append(f"  {idx:16s} 평균 |WPA| {r['평균_절대WPA']*100:.3f}%p "
                 f"(선발 대비 {r['평균_절대WPA']/ref:.2f}배) · "
                 f"중앙 |점수차| {r['중앙_점수차']:.0f} · n={int(r['n']):,}")
    L.append("  → 배수가 1에서 멀수록 같은 자로 맞대기 어렵다. "
             "레버리지로 나눈 WPA/LI가 정석이지만 여기서는 하지 않았다.")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150, facecolor=SURFACE)
    for ax, lab in zip(axes, ("전체", "접전 (2점 이내)")):
        f, cols, n = store[lab]
        present = [c for c in [f"구원 {t}" for t in T.TIERS] if c in cols]
        vals = [gain(f, c)[0] for c in present]
        errs = [(gain(f, c)[0] - gain(f, c)[1]) for c in present]
        x = np.arange(len(present))
        ax.bar(x, vals, color=[BLUE if v >= 0 else ORANGE for v in vals], width=0.5)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK,
                    elinewidth=1.4, capsize=6)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + (e + 0.06) * (1 if v >= 0 else -1), f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
        ax.axhline(0, color=INK, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("구원 ", "") + "로\n교체" for c in present], fontsize=9)
        span = max(abs(v) + e for v, e in zip(vals, errs)) * 1.45
        ax.set_ylim(-span, span)
        ax.set_title(lab, color=INK, fontsize=12, loc="left", pad=10)
        ax.set_ylabel("9타자당 승리확률 (%p)", color=SECONDARY_INK, fontsize=9.5)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_facecolor(SURFACE)
    fig.suptitle("교체 이득을 승리확률로 재면", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "tier_wpa.png")
    plt.close(fig)

    path = DATA.parent / "eda_tier_wpa_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
