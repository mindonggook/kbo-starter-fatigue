"""적응 메커니즘을 직접 때린다 — 구종이 적으면 더 빨리 읽히는가.

발견 06·14는 '타자가 적응한다'고 했지만 적응의 내용은 추론이었다.
적응이 <패턴 학습>이라면 읽을 패턴이 적은 투수가 더 빨리 읽혀야 한다.
구종 라벨이 생겼으니 그 예측을 직접 검정할 수 있다.

예측: 레퍼토리가 좁은 투수의 타순 회전 페널티가 더 크다.
기각되면 적응은 '구종 조합을 외우는 것'이 아니라 다른 무엇이다
(타이밍, 릴리스 포인트, 그날의 컨디션 읽기 등).

식별 전략 — 레퍼토리는 투수마다 거의 고정이므로 투수×시즌 고정효과를 넣어도
<회전 × 레퍼토리> 교차항은 식별된다. 투수의 실력 차이는 고정효과가 흡수한다.
생존 편향은 '3바퀴까지 간 등판만' 남겨 통제한다.
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

AB_KINDS = ("안타", "낫아웃 출루", "삼진", "실책 출루", "야수선택", "인플레이 아웃")
SIG = {True: "유의함", False: "유의하지 않음"}
MIN_PITCHES = 300


def load(seasons):
    frames = []
    for s in seasons:
        path = DATA / f"naver_full_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["구종", "pitcher_code", "batter", "batting_order"])

    first = d[d["inning"] == 1]
    starters = first.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    d = d[[c in starters.get(g, set())
           for g, c in zip(d["game_id"], d["pitcher_code"])]].copy()
    d["start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)
    d["투수시즌"] = d["pitcher_code"].astype(str) + "_" + d["season"].astype(str)
    return d


def repertoire(d):
    """투수×시즌 단위 레퍼토리. 등판 단위로 재면 짧은 등판이 자동으로 좁아 보인다."""
    cnt = d.groupby(["투수시즌", "구종"]).size().rename("n").reset_index()
    tot = cnt.groupby("투수시즌")["n"].transform("sum")
    cnt["share"] = cnt["n"] / tot
    g = cnt.groupby("투수시즌")
    rep = pd.DataFrame({
        "총투구": g["n"].sum(),
        "구종수": g["share"].apply(lambda s: int((s >= 0.10).sum())),
        "엔트로피": g["share"].apply(lambda s: float(-(s * np.log(s)).sum())),
    })
    # 투심을 주 패스트볼로 쓰는 투수는 '직구'가 1%대로 잡힌다 — 계열로 묶어야 한다.
    for name, types in (("직구비율", ("직구",)),
                        ("속구비율", ("직구", "투심")),
                        ("속구비율_커터", ("직구", "투심", "커터"))):
        sel = cnt[cnt["구종"].isin(types)].groupby("투수시즌")["share"].sum()
        rep[name] = sel.reindex(rep.index).fillna(0.0)
    return rep[rep["총투구"] >= MIN_PITCHES]


def pa_table(d):
    d = d.sort_values(["start", "투수_누적투구수"])
    d["pa_id"] = (d.groupby("start")["투구번호"].diff().fillna(1) <= 0).cumsum()
    pa = d.groupby(["start", "pa_id"], as_index=False).agg(
        season=("season", "first"), 투수시즌=("투수시즌", "first"),
        pitcher=("pitcher", "first"), batter=("batter", "first"),
        batting_order=("batting_order", "first"), 종류=("타석종류", "first"),
        투구수=("투구번호", "size"), 시작투구수=("투수_누적투구수", "min"))
    pa = pa.dropna(subset=["종류"])
    pa = pa.sort_values(["start", "시작투구수"])
    # 같은 타자를 몇 번째로 만나는가 — 이것이 타순 회전이다.
    pa["회전"] = pa.groupby(["start", "batter"]).cumcount() + 1
    pa["타수"] = pa["종류"].isin(AB_KINDS).astype(float)
    pa["안타"] = (pa["종류"] == "안타").astype(float)
    return pa


def fit(d, inter_cols=()):
    """회전 더미 + 회전×레퍼토리 + 타순 더미 + 투수×시즌 고정효과."""
    X = pd.DataFrame(index=d.index)
    X["회전2"] = (d["회전"] == 2).astype(float)
    X["회전3"] = (d["회전"] == 3).astype(float)
    for col in inter_cols:
        z = d[col].astype(float)
        z = z - z.mean()          # 중심화 — 회전 주효과를 평균 레퍼토리에서 읽게
        X[f"회전2×{col}"] = X["회전2"] * z
        X[f"회전3×{col}"] = X["회전3"] * z
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = X.join(pd.get_dummies(d["투수시즌"], prefix="P", drop_first=True).astype(float))
    X = sm.add_constant(X)
    return sm.OLS(d["안타"].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})


def grab(f, name):
    ci = f.conf_int().loc[name]
    return f.params[name], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    raw = load(a.seasons)
    rep = repertoire(raw)
    pa = pa_table(raw)
    pa = pa.join(rep, on="투수시즌", how="inner")

    # 3바퀴까지 간 등판만 — 1바퀴와 3바퀴를 같은 등판에서 비교한다.
    deep = pa[pa["회전"] >= 3]["start"].unique()
    pa = pa[pa["start"].isin(deep) & pa["회전"].between(1, 3)].copy()
    ab = pa[pa["타수"] == 1].copy()

    L = [f"=== 레퍼토리와 타순 회전 페널티 ({len(a.seasons)}시즌) ===",
         f"투수×시즌 {rep.shape[0]}개 (시즌 {MIN_PITCHES}구 이상) · "
         f"3바퀴까지 간 등판 {ab['start'].nunique():,}개 · 타수 {len(ab):,}건\n"]

    L.append("[1] 레퍼토리 분포 (투수×시즌)")
    L.append(f"  구종수(10% 이상): " + " / ".join(
        f"{k}개 {v}명" for k, v in rep["구종수"].value_counts().sort_index().items()))
    for c in ("엔트로피", "직구비율", "속구비율", "속구비율_커터", "총투구"):
        q = rep[c].describe()
        L.append(f"  {c}: 중앙 {rep[c].median():.3f} · "
                 f"25~75% {q['25%']:.3f}~{q['75%']:.3f} · 범위 {q['min']:.3f}~{q['max']:.3f}")
    L.append("")

    # ── [2] 그룹으로 나눠 눈으로 ────────────────────────────────
    ab["그룹"] = np.where(ab["구종수"] <= 2, "좁음 (2개 이하)",
                        np.where(ab["구종수"] >= 4, "넓음 (4개 이상)", "보통 (3개)"))
    L.append("[2] 레퍼토리 그룹별 회전 피안타율 (타순 구성 표준화)")
    # 직접 표준화 — 전체 타순 분포를 공통 가중치로 쓴다.
    w = ab["batting_order"].value_counts(normalize=True)
    rows = {}
    for grp in ("좁음 (2개 이하)", "보통 (3개)", "넓음 (4개 이상)"):
        sub = ab[ab["그룹"] == grp]
        if sub.empty:
            continue
        cells = sub.groupby(["회전", "batting_order"])["안타"].mean().unstack()
        cells = cells.reindex(columns=w.index)
        std = (cells * w).sum(axis=1) / cells.notna().mul(w, axis=1).sum(axis=1)
        n = sub.groupby("회전").size()
        rows[grp] = (std, n)
        L.append(f"  {grp}: " + " / ".join(
            f"{int(t)}바퀴 {std.get(t, np.nan):.4f}(n={n.get(t, 0):,})" for t in (1, 2, 3))
            + f"  1→3 {(std.get(3, np.nan)-std.get(1, np.nan))*1000:+.1f}/1000")
    L.append("")

    # ── [3] 교차항 회귀 ────────────────────────────────────────
    L.append("[3] 회전 × 레퍼토리 교차항 (투수×시즌 고정효과 · 등판 군집)")
    base = fit(ab)
    b2, b3 = grab(base, "회전2"), grab(base, "회전3")
    L.append(f"  기준 모형: 2바퀴 {b2[0]*1000:+.1f}/1000 [{b2[1]*1000:+.1f}, {b2[2]*1000:+.1f}] "
             f"{SIG[b2[3]]} · 3바퀴 {b3[0]*1000:+.1f} [{b3[1]*1000:+.1f}, {b3[2]*1000:+.1f}] "
             f"{SIG[b3[3]]}")
    SPECS = (("구종수", "구종 1개당", 1.0),
             ("엔트로피", "엔트로피 0.1당", 0.1),
             ("직구비율", "직구 10%p당", 0.1),
             ("속구비율", "속구(직구+투심) 10%p당", 0.1),
             ("속구비율_커터", "속구(+커터) 10%p당", 0.1))
    for col, unit, step in SPECS:
        f = fit(ab, [col])
        L.append(f"  {col} ({unit})")
        for nm, g in (("2바퀴 교차", grab(f, f"회전2×{col}")),
                      ("3바퀴 교차", grab(f, f"회전3×{col}"))):
            L.append(f"    {nm}: {g[0]*step*1000:+.2f}/1000 "
                     f"[{g[1]*step*1000:+.2f}, {g[2]*step*1000:+.2f}] {SIG[g[3]]}")
    # 구종수와 속구비율은 음의 상관이 있다 — 같이 넣어 어느 쪽이 남는지 본다.
    f = fit(ab, ["구종수", "속구비율"])
    L.append("  둘을 같이 넣으면 (3바퀴 교차)")
    for col, step in (("구종수", 1.0), ("속구비율", 0.1)):
        g = grab(f, f"회전3×{col}")
        L.append(f"    {col}: {g[0]*step*1000:+.2f}/1000 "
                 f"[{g[1]*step*1000:+.2f}, {g[2]*step*1000:+.2f}] {SIG[g[3]]}")
    L.append("")

    # ── [4] 구속을 통제해도 남는가 ─────────────────────────────
    # 레퍼토리가 넓은 투수가 그냥 좋은 투수일 수 있다. 고정효과가 실력 수준은
    # 흡수하지만, '회전에 따라 다르게 무너지는' 성향까지 흡수하지는 않는다.
    L.append("[4] 유보 — 레퍼토리와 함께 움직이는 것들")
    for col in ("구종수", "엔트로피", "속구비율"):
        L.append(f"  {col} ↔ 시즌 총투구 상관: {rep[col].corr(rep['총투구']):+.3f}")
    L.append(f"  구종수 ↔ 엔트로피 상관: {rep['구종수'].corr(rep['엔트로피']):+.3f}")
    L.append(f"  구종수 ↔ 속구비율 상관: {rep['구종수'].corr(rep['속구비율']):+.3f}")
    L.append(f"  직구비율 ↔ 속구비율 상관: {rep['직구비율'].corr(rep['속구비율']):+.3f}")
    # 극단값이 끌고 가는지 — 속구비율 상하 5%를 잘라 다시
    lo, hi = rep["속구비율"].quantile([0.05, 0.95])
    trim = ab[ab["속구비율"].between(lo, hi)]
    g = grab(fit(trim, ["속구비율"]), "회전3×속구비율")
    L.append(f"  속구비율 상하 5% 절단 후 3바퀴 교차: {g[0]*100:+.2f}/1000 "
             f"[{g[1]*100:+.2f}, {g[2]*100:+.2f}] {SIG[g[3]]} (타수 {len(trim):,})")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    for (grp, (std, n)), color in zip(rows.items(), (ORANGE, MUTED, BLUE)):
        ts = [t for t in (1, 2, 3) if not np.isnan(std.get(t, np.nan))]
        ys = np.array([std[t] for t in ts])
        ns = np.array([max(int(n.get(t, 1)), 1) for t in ts])
        se = np.sqrt(ys * (1 - ys) / ns)          # 표본이 작은 그룹을 숨기지 않는다
        ax.plot(ts, ys, color=color, linewidth=2.3, marker="o", markersize=7, label=grp)
        ax.fill_between(ts, ys - 1.96 * se, ys + 1.96 * se, color=color, alpha=0.16,
                        linewidth=0)
        ax.text(ts[-1] + 0.08, ys[-1], f"{ys[-1]:.3f}", color=color,
                fontsize=9.5, va="center", fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1바퀴", "2바퀴", "3바퀴"], fontsize=9.5)
    ax.set_xlim(0.85, 3.45)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
    ax.set_title("레퍼토리가 좁으면 더 빨리 읽히나", color=INK, fontsize=12.5,
                 loc="left", pad=10)
    ax.set_ylabel("표준화 피안타율", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    names, vals, errs = [], [], []
    for col, unit, step in (("구종수", "구종\n1개당", 1.0),
                            ("엔트로피", "엔트로피\n0.1당", 0.1),
                            ("속구비율", "속구\n10%p당", 0.1)):
        g = grab(fit(ab, [col]), f"회전3×{col}")
        names.append(unit)
        vals.append(g[0] * step * 1000)
        errs.append((g[0] - g[1]) * step * 1000)
    x = np.arange(len(names))
    # 구종 1개당 / 엔트로피 0.1당 / 속구 10%p당은 단위가 다르다.
    # 같은 축에 막대로 놓으면 비교가 성립하지 않으므로 t값(계수/표준오차)으로 바꾼다.
    tvals = [v / (e / 1.96) if e else np.nan for v, e in zip(vals, errs)]
    ax.barh(x[::-1], tvals, color=[BLUE if v < 0 else ORANGE for v in tvals], height=0.5)
    for xi, (t, v, e) in zip(x[::-1], zip(tvals, vals, errs)):
        ax.text(t + (0.08 if t >= 0 else -0.08), xi,
                f"{v:+.1f}±{e:.1f}  (t={t:+.2f})",
                va="center", ha="left" if t >= 0 else "right",
                color=SECONDARY_INK, fontsize=9)
    ax.axvline(0, color=INK, linewidth=0.9)
    for b in (-1.96, 1.96):
        ax.axvline(b, color=GRID, linewidth=1.1, linestyle="--")
    ax.set_yticks(x[::-1])
    ax.set_yticklabels([" ".join(n.split()) for n in names], fontsize=9)
    span = max(2.4, max(abs(t) for t in tvals if np.isfinite(t)) * 1.9)
    ax.set_xlim(-span, span)
    ax.set_title("3바퀴 페널티가 레퍼토리에 따라 달라지나", color=INK, fontsize=12.5,
                 loc="left", pad=10)
    ax.set_xlabel("표준화한 교차항 (t값 · 점선은 ±1.96)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("타자는 무엇에 적응하는가 — 구종 조합인가", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "repertoire_tto.png")
    plt.close(fig)

    path = RESULTS / "eda_repertoire_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
