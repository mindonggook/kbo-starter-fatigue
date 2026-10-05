"""불펜 등급의 결과 누수를 걷어내고 발견 16·29·30을 다시 잰다.

외부 평가의 최우선 지적이다. 같은 시즌의 세이브·홀드로 등급을 매기면
'그 시즌 잘 막은 투수'를 필승조라 부른 뒤 같은 시즌 성적으로 확인하는 꼴이다.

직전 시즌 보직으로 다시 매겨 같은 분석을 돌린다. 두 정의의 일치율은 67.5%뿐이고
33.2%는 직전 기록이 없어 빠지므로, 결과가 달라질 여지가 충분하다.

평가의 ④번(승리확률)도 여기서 부분적으로 다룬다 -- 등급별 등판 상황의 점수차를
같이 보고해서, 추격조의 손해가 '큰 점수차에서만 나오는 것'인지 확인한다.
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


def standardize(d, level, value, weights):
    cells = d.groupby([level, "batting_order"], observed=True)[value].mean().unstack()
    cells = cells.reindex(columns=weights.index)
    w = cells.notna().mul(weights, axis=1)
    return (cells * weights).sum(axis=1) / w.sum(axis=1)


def role_table(pa, tier_df, slots):
    """5~8회, 선발 3바퀴 이상 vs 구원 1바퀴째(등급별)."""
    mid = pa[pa["inning"].between(5, 8)].copy()
    s3 = mid[mid["is_starter"] & (mid["타순회전"] >= 3)].copy()
    s3["역할"] = "선발 3바퀴 이상"
    rel = mid[~mid["is_starter"]].copy()
    rel = rel.merge(tier_df, on=["season", "pitcher_team", "pitcher"], how="left")
    rel = rel.dropna(subset=["등급"])
    rel = rel.sort_values(["game_id", "pitcher", "inning"])
    rel["구원회전"] = rel.groupby(["game_id", "pitcher", "batter"]).cumcount() + 1
    rel = rel[rel["구원회전"] == 1].copy()
    rel["역할"] = "구원 " + rel["등급"]
    both = pd.concat([s3, rel], ignore_index=True)
    rv = standardize(both, "역할", "득점가치", slots)
    avg = standardize(both[both["타수"]], "역할", "안타", slots)
    n = both.groupby("역할").size()
    bb = both.assign(볼넷=(both["종류"] == "볼넷·사구").astype(float)).groupby("역할")["볼넷"].mean()
    gap = both.groupby("역할")["점수차_투수팀기준"].agg(["mean", "median"])
    return rv, avg, n, bb, gap, both


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2018, 2027)))
    a = p.parse_args()

    pa = load(a.seasons)
    slots = pa["batting_order"].value_counts(normalize=True)
    base = "선발 3바퀴 이상"

    L = [f"=== 불펜 등급의 결과 누수를 걷어내고 다시 ({len(a.seasons)}시즌) ===",
         f"2018~2026 (직전 시즌 정의를 쓰려면 2017은 기준 시즌으로만 쓴다)",
         f"타석 {len(pa):,}건\n"]

    results = {}
    for mode, label in (("same", "같은 시즌 (원래 · 누수 있음)"),
                        ("prev", "직전 시즌 (보정 · 누수 없음)")):
        tier_df = T.build(a.seasons, mode=mode).dropna(subset=["등급"])
        rv, avg, n, bb, gap, both = role_table(pa, tier_df, slots)
        results[mode] = dict(rv=rv, avg=avg, n=n, bb=bb, gap=gap)
        L.append(f"[{label}]")
        L.append(f"  {base}: 득점가치 {rv[base]:+.4f} · 피안타율 {avg[base]:.4f} "
                 f"· 볼넷 {bb[base]*100:.2f}% (n={n[base]:,})")
        for t in T.TIERS:
            key = f"구원 {t}"
            if key not in rv.index:
                continue
            g_rv = rv[base] - rv[key]
            g_avg = avg[base] - avg[key]
            L.append(f"  {key}: 득점가치 {rv[key]:+.4f} · 피안타율 {avg[key]:.4f} "
                     f"· 볼넷 {bb[key]*100:.2f}% (n={n[key]:,})")
            L.append(f"    → 이득 {g_rv*1000:+.1f}/1000점 "
                     f"(피안타율 {g_avg*1000:+.1f}/1000) · 9타자 {g_rv*9:+.3f}점")
        L.append("")

    # 두 정의의 차이를 한눈에
    L.append("[비교] 9타자 환산 교체 이득 (점)")
    L.append("  " + "등급".ljust(10) + f"{'같은 시즌':>12s}{'직전 시즌':>12s}{'차이':>10s}")
    for t in T.TIERS:
        key = f"구원 {t}"
        vals = []
        for mode in ("same", "prev"):
            rv = results[mode]["rv"]
            vals.append((rv[base] - rv[key]) * 9 if key in rv.index else np.nan)
        L.append("  " + t.ljust(10) + f"{vals[0]:>12.3f}{vals[1]:>12.3f}"
                 f"{vals[1]-vals[0]:>10.3f}")
    L.append("")

    # 평가 ④ — 추격조 손해가 큰 점수차에서만 나오는가
    L.append("[상황] 등급별 등판 시점 점수차 (투수팀 기준 · 양수면 이기는 중)")
    for mode, label in (("same", "같은 시즌"), ("prev", "직전 시즌")):
        gap = results[mode]["gap"]
        L.append(f"  [{label}] " + " / ".join(
            f"{k.replace('구원 ','')} 중앙 {int(r['median']):+d}" for k, r in gap.iterrows()))
    L.append("")

    # 점수차 구간별로 나눠 — 접전에서도 같은 결론인가
    L.append("[접전만] 점수차 2점 이내에서 다시 (직전 시즌 정의)")
    tier_df = T.build(a.seasons, mode="prev").dropna(subset=["등급"])
    close = pa[pa["점수차_투수팀기준"].abs() <= 2]
    rv, avg, n, bb, gap, _ = role_table(close, tier_df, slots)
    L.append(f"  {base}: 득점가치 {rv[base]:+.4f} (n={n[base]:,})")
    for t in T.TIERS:
        key = f"구원 {t}"
        if key not in rv.index:
            continue
        L.append(f"  {key}: 이득 {(rv[base]-rv[key])*1000:+.1f}/1000점 "
                 f"· 9타자 {(rv[base]-rv[key])*9:+.3f}점 (n={n[key]:,})")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    x = np.arange(len(T.TIERS))
    for off, (mode, color, lab) in ((-0.19, ("same", MUTED, "같은 시즌 (누수)")),
                                    (0.19, ("prev", VIOLET, "직전 시즌 (보정)"))):
        rv = results[mode]["rv"]
        vals = [(rv[base] - rv[f"구원 {t}"]) * 9 if f"구원 {t}" in rv.index else np.nan
                for t in T.TIERS]
        ax.bar(x + off, vals, color=color, width=0.36, label=lab)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi + off, v + (0.012 if v >= 0 else -0.012), f"{v:+.2f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        color=SECONDARY_INK, fontsize=8.5)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}로\n교체" for t in T.TIERS], fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
    ax.set_title("등급 정의를 바꾸면 교체 이득이 달라지나", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_ylabel("9타자당 실점 이득 (점)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    gap = results["prev"]["gap"]
    keys = [base] + [f"구원 {t}" for t in T.TIERS if f"구원 {t}" in gap.index]
    vals = [gap.loc[k, "median"] for k in keys]
    y = np.arange(len(keys))[::-1]
    ax.barh(y, vals, color=[MUTED] + [BLUE, ORANGE, VIOLET][:len(keys) - 1], height=0.55)
    for yi, v in zip(y, vals):
        ax.text(v + (0.06 if v >= 0 else -0.06), yi, f"{v:+.0f}", va="center",
                ha="left" if v >= 0 else "right", color=SECONDARY_INK, fontsize=9.5)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("구원 ", "") for k in keys], fontsize=9)
    ax.set_title("등급별 등판 시점 점수차 (중앙값)", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("투수팀 기준 점수차", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("등급 정의의 누수를 걷어내면", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "tier_fix.png")
    plt.close(fig)

    path = DATA.parent / "eda_tier_fix_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
