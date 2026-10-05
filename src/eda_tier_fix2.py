"""2차 평가 ④⑤ — 등급 정의 세 가지를 신뢰구간과 함께, 그리고 필승조-중간 차이.

평가가 두 가지를 지적했다.
  ④ 발견 32에 신뢰구간이 없다. 이 리포트의 유일한 실전 조언이 점추정뿐이다.
  ⑤ 직전 시즌 정의는 신인·첫 시즌 33.2%를 버린다. 시점 기준 누적이 더 낫다.

세 정의를 나란히 돌린다.
  same  같은 시즌 세이브+홀드      누수 있음 (원래)
  prev  직전 시즌 세이브+홀드      누수 없음, 표본 33.2% 손실
  pit   최근 365일 누적 (시점 기준) 누수 없음, 표본 5.8% 손실

그리고 <필승조 − 중간> 차이를 직접 검정한다. 발견 30의 가용성 비용이
'내일 필승조를 못 쓸 확률 × (필승조 − 대체 투수)'이므로, 이 차이가 0이면
비용 자체가 0이 된다. 평가가 지적한 바로 그 지점이다.

신뢰구간은 경기 단위 군집보정으로 낸다 -- 같은 경기의 타석들은 독립이 아니다.
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
    pa["안타"] = pa["안타"].astype(float)
    pa["타수"] = pa["타수"].astype(bool)
    return pa


def roles(pa, tier_df, on):
    mid = pa[pa["inning"].between(5, 8)].copy()
    s3 = mid[mid["is_starter"] & (mid["타순회전"] >= 3)].copy()
    s3["역할"] = BASE
    rel = mid[~mid["is_starter"]].merge(tier_df, on=on, how="left")
    rel = rel.dropna(subset=["등급"])
    rel = rel.sort_values(["game_id", "pitcher", "inning"])
    rel["구원회전"] = rel.groupby(["game_id", "pitcher", "batter"]).cumcount() + 1
    rel = rel[rel["구원회전"] == 1].copy()
    rel["역할"] = "구원 " + rel["등급"].astype(str)
    return pd.concat([s3, rel], ignore_index=True)


def fit_roles(both, y):
    """역할 더미 + 타순 더미. 기준은 선발 3바퀴 이상. 경기 단위 군집보정."""
    d = both if y != "안타" else both[both["타수"]]
    d = d.copy()
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
    return f, cols, d


def gain(f, key):
    """이득 = 선발 − 구원 = −계수. 부호와 경계를 뒤집는다."""
    ci = f.conf_int().loc[key]
    return -f.params[key], -ci[1], -ci[0], (ci[0] > 0) == (ci[1] > 0)


def contrast(f, a, b):
    """a − b 차이와 그 신뢰구간 (둘 다 같은 기준 대비 계수)."""
    v = f.cov_params()
    d = f.params[a] - f.params[b]
    se = np.sqrt(v.loc[a, a] + v.loc[b, b] - 2 * v.loc[a, b])
    return d, d - 1.96 * se, d + 1.96 * se, abs(d) > 1.96 * se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2018, 2027)))
    a = p.parse_args()
    pa = load(a.seasons)

    defs = {
        "same": ("같은 시즌 (누수 있음)",
                 T.build(a.seasons, "same").dropna(subset=["등급"]),
                 ["season", "pitcher_team", "pitcher"]),
        "prev": ("직전 시즌 (누수 없음 · 표본 33% 손실)",
                 T.build(a.seasons, "prev").dropna(subset=["등급"]),
                 ["season", "pitcher_team", "pitcher"]),
        "pit": ("시점 기준 누적 365일 (누수 없음 · 표본 6% 손실)",
                T.build_pit(a.seasons).dropna(subset=["등급"]),
                ["game_id", "season", "pitcher_team", "pitcher"]),
    }

    L = [f"=== 등급 정의 세 가지 · 신뢰구간 · 필승조−중간 차이 ({len(a.seasons)}시즌) ===",
         f"5~8회 · 선발 3바퀴 이상 대비 구원 1바퀴째 · 경기 단위 군집보정",
         "9타자 환산 = 타석당 값 × 9\n"]

    store = {}
    for key, (label, tdf, on) in defs.items():
        both = roles(pa, tdf, on)
        f, cols, d = fit_roles(both, "득점가치")
        store[key] = dict(f=f, cols=cols, n=d.groupby("역할").size(), both=both)
        L.append(f"[{label}]")
        for c in cols:
            g = gain(f, c)
            L.append(f"  {c:10s} 이득 {g[0]*9:+.3f}점/9타자 "
                     f"[{g[1]*9:+.3f}, {g[2]*9:+.3f}] {SIG[g[3]]} "
                     f"(n={store[key]['n'].get(c, 0):,})")
        if "구원 필승조" in cols and "구원 중간" in cols:
            c0 = contrast(f, "구원 필승조", "구원 중간")
            L.append(f"  필승조 − 중간: {-c0[0]*9:+.3f}점/9타자 "
                     f"[{-c0[2]*9:+.3f}, {-c0[1]*9:+.3f}] {SIG[c0[3]]}")
        L.append("")

    # ── 접전에서 (평가 ④의 핵심) ───────────────────────────────
    L.append("[접전 · 점수차 2점 이내 · 시점 기준 등급]")
    close = pa[pa["점수차_투수팀기준"].abs() <= 2]
    both_c = roles(close, defs["pit"][1], defs["pit"][2])
    fc, colsc, dc = fit_roles(both_c, "득점가치")
    nc = dc.groupby("역할").size()
    for c in colsc:
        g = gain(fc, c)
        L.append(f"  {c:10s} 이득 {g[0]*9:+.3f}점/9타자 "
                 f"[{g[1]*9:+.3f}, {g[2]*9:+.3f}] {SIG[g[3]]} (n={nc.get(c, 0):,})")
    if "구원 필승조" in colsc and "구원 중간" in colsc:
        c0 = contrast(fc, "구원 필승조", "구원 중간")
        L.append(f"  필승조 − 중간: {-c0[0]*9:+.3f}점/9타자 "
                 f"[{-c0[2]*9:+.3f}, {-c0[1]*9:+.3f}] {SIG[c0[3]]}")
        L.append("  → 이 차이가 0이면 발견 30의 가용성 비용도 0이 된다"
                 " (비용 = 못 쓸 확률 × 이 차이).")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150, facecolor=SURFACE)
    for ax, (f_, cols_, lab) in zip(axes, ((store["pit"]["f"], store["pit"]["cols"],
                                            "전체 (5~8회)"),
                                           (fc, colsc, "접전 (2점 이내)"))):
        present = [c for c in [f"구원 {t}" for t in T.TIERS] if c in cols_]
        vals = [gain(f_, c)[0] * 9 for c in present]
        errs = [(gain(f_, c)[0] - gain(f_, c)[1]) * 9 for c in present]
        x = np.arange(len(present))
        ax.bar(x, vals, color=[BLUE if v >= 0 else ORANGE for v in vals], width=0.5)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=SECONDARY_INK,
                    elinewidth=1.4, capsize=6)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + (e + 0.012) * (1 if v >= 0 else -1), f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
        ax.axhline(0, color=INK, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("구원 ", "") + "로\n교체" for c in present],
                           fontsize=9)
        span = max(abs(v) + e for v, e in zip(vals, errs)) * 1.45
        ax.set_ylim(-span, span)
        ax.set_title(f"{lab} · 시점 기준 등급", color=INK, fontsize=12, loc="left", pad=10)
        ax.set_ylabel("9타자당 실점 이득 (점)", color=SECONDARY_INK, fontsize=9.5)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_facecolor(SURFACE)
    fig.suptitle("교체 이득에 신뢰구간을 붙이면", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "tier_ci.png")
    plt.close(fig)

    path = DATA.parent / "eda_tier_fix2_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
