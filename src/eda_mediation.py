"""피로는 성적 저하의 몇 퍼센트를 설명하는가 — 구속을 매개로 한 분해.

발견 17은 두 사실을 나란히 놓았을 뿐 잇지는 못했다.
  (1) 76구 이후 구속이 1.6km/h 떨어진다
  (2) 그런데 성적 저하는 타자 적응 쪽 지표가 더 크게 움직인다

이제 같은 타석에 구속과 결과가 함께 있으므로 직접 이을 수 있다.

    경로 a : 투구수가 쌓이면 구속이 얼마나 떨어지는가        (관측됨, -1.6km/h)
    경로 b : 구속이 1km/h 떨어지면 피안타율이 얼마나 오르는가  (여기서 추정)
    매개분 = a × b
    총효과 = 투구수가 쌓일 때 실제로 오른 피안타율

    피로 기여분 = 매개분 / 총효과

경로 b를 추정할 때 두 가지를 조심해야 한다.
  - 구종 혼재: 체인지업이 많은 타석은 평균 구속이 낮다. 피로가 아니라 선택이다.
    그래서 '그 타석의 최고 구속'(직구에 가깝다)을 쓰고, 타석 투구수를 통제한다.
  - 투수 차이: 느린 투수가 더 맞는 것은 피로와 무관하다.
    그래서 등판별 자기 평균 대비 편차로 바꾼다.
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


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"naver_pa_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    pa = pd.concat(frames, ignore_index=True)
    pa = pa.dropna(subset=["타수", "최고구속", "pitcher_code"])
    pa = pa[pa["최고구속"].between(100, 170)].copy()
    pa["타수"] = pa["타수"].astype(bool)
    pa["안타"] = pa["안타"].astype(bool)

    # 선발 = 1회에 처음 등판한 투수
    first_inn = pa[pa["inning"] == 1]
    starters = first_inn.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    pa["is_starter"] = [c in starters.get(g, set())
                        for g, c in zip(pa["game_id"], pa["pitcher_code"])]

    pa["start"] = pa["game_id"].astype(str) + "_" + pa["pitcher_code"].astype(str)
    return pa


def prep(pa: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    """선발 중 90구 이상 간 등판만 — 생존 편향 통제는 앞선 분석들과 동일하게."""
    d = pa[pa["is_starter"]].copy()
    reached = d[d["투수_누적투구수"] >= threshold]["start"].unique()
    d = d[d["start"].isin(reached) & d["타수"]].copy()
    d = d[d["이전_누적투구수"].between(0, 100)].copy()

    # 등판별 자기 평균 대비 편차 — 투수 간 구속·실력 차이를 지운다.
    d["구속_편차"] = d["최고구속"] - d.groupby("start")["최고구속"].transform("mean")
    d["투구수_10구당"] = d["이전_누적투구수"] / 10.0
    d["안타_i"] = d["안타"].astype(float)
    return d


def fit(d: pd.DataFrame, formula_cols: list[str]) -> sm.regression.linear_model.RegressionResults:
    X = pd.DataFrame(index=d.index)
    for c in formula_cols:
        X[c] = d[c].astype(float)
    # 타석 투구수를 통제한다 — 최고 구속은 공을 많이 던질수록 높게 나오기 쉽다.
    X["타석투구수"] = d["투구수"].astype(float)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = sm.add_constant(X)
    return sm.OLS(d["안타_i"], X).fit(cov_type="cluster", cov_kwds={"groups": d["start"]})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    pa = load(a.seasons)
    d = prep(pa)

    # 경로 a — 투구수 → 구속
    Xa = pd.DataFrame({"투구수_10구당": d["투구수_10구당"].astype(float),
                       "타석투구수": d["투구수"].astype(float)}, index=d.index)
    fit_a = sm.OLS(d["구속_편차"].astype(float), sm.add_constant(Xa)).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})
    a_coef = fit_a.params["투구수_10구당"]
    a_ci = fit_a.conf_int().loc["투구수_10구당"]

    # 총효과 — 투구수 → 피안타율 (구속을 넣지 않은 모형)
    fit_total = fit(d, ["투구수_10구당"])
    total = fit_total.params["투구수_10구당"]
    total_ci = fit_total.conf_int().loc["투구수_10구당"]

    # 경로 b + 직접효과 — 구속을 같이 넣는다
    fit_both = fit(d, ["투구수_10구당", "구속_편차"])
    b_coef = fit_both.params["구속_편차"]
    b_ci = fit_both.conf_int().loc["구속_편차"]
    direct = fit_both.params["투구수_10구당"]
    direct_ci = fit_both.conf_int().loc["투구수_10구당"]

    mediated = a_coef * b_coef
    share = mediated / total * 100 if total else float("nan")

    # 그림 — 구속 편차 구간별 피안타율
    d["구속_구간"] = pd.qcut(d["구속_편차"], 5,
                          labels=["평소보다\n많이 느림", "느림", "평소 수준", "빠름", "평소보다\n많이 빠름"])
    g = d.groupby("구속_구간", observed=True).agg(
        피안타율=("안타_i", "mean"), n=("안타_i", "size"),
        중앙편차=("구속_편차", "median"))
    g["se"] = np.sqrt(g["피안타율"] * (1 - g["피안타율"]) / g["n"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    x = np.arange(len(g))
    ax.bar(x, g["피안타율"], color=BLUE, width=0.58)
    ax.errorbar(x, g["피안타율"], yerr=g["se"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    for xi, r in zip(x, g.itertuples()):
        ax.text(xi, r.피안타율 + r.se * 1.96 + 0.004, f"{r.피안타율:.3f}", ha="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
        ax.text(xi, 0.008, f"{r.중앙편차:+.1f}km/h", ha="center", color=SURFACE, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index.astype(str), fontsize=8.5)
    ax.set_ylim(0, g["피안타율"].max() * 1.3)
    ax.set_title("그 투수가 평소보다 느리게 던진 타석의 결과", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("피안타율", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    parts = ["총효과\n(투구수 10구당)", "그중 구속이\n설명하는 몫", "나머지\n(직접효과)"]
    vals = [total * 1000, mediated * 1000, direct * 1000]
    colors = [MUTED, ORANGE, AQUA]
    xb = np.arange(len(parts))
    ax.bar(xb, vals, color=colors, width=0.55)
    ax.axhline(0, color=MUTED, linewidth=1.2)
    for xi, v in zip(xb, vals):
        ax.text(xi, v + (0.08 if v >= 0 else -0.08), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", color=SECONDARY_INK,
                fontsize=10, fontweight="bold")
    ax.set_xticks(xb)
    ax.set_xticklabels(parts, fontsize=9)
    ax.set_title(f"피로(구속)가 설명하는 몫: {share:.1f}%", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("피안타율 변화 (1/1000)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("구속 저하는 성적 저하의 얼마를 설명하는가",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "mediation_velocity.png")
    plt.close(fig)

    lines = [f"=== 구속을 매개로 한 분해 ({len(a.seasons)}시즌) ===",
             f"선발 타석 {len(d):,}건 / 등판 {d['start'].nunique():,}개 "
             f"(90구 이상 간 등판, 타수만, 등판 단위 군집 보정)\n"]

    lines.append("[경로 a] 투구수 10구 증가 → 구속 변화")
    lines.append(f"  {a_coef:+.4f} km/h [{a_ci[0]:+.4f}, {a_ci[1]:+.4f}]")
    lines.append(f"  → 100구 누적 시 {a_coef*10:+.2f} km/h\n")

    lines.append("[경로 b] 구속 1km/h 상승 → 피안타율 변화 (투구수 통제)")
    lines.append(f"  {b_coef:+.5f} [{b_ci[0]:+.5f}, {b_ci[1]:+.5f}]"
                 f" {'유의함' if (b_ci[0]>0)==(b_ci[1]>0) else '유의하지 않음'}\n")

    lines.append("[총효과] 투구수 10구 증가 → 피안타율")
    lines.append(f"  {total:+.5f} [{total_ci[0]:+.5f}, {total_ci[1]:+.5f}]"
                 f" {'유의함' if (total_ci[0]>0)==(total_ci[1]>0) else '유의하지 않음'}\n")

    lines.append("[분해]")
    lines.append(f"  매개분 (a × b) = {mediated:+.5f}")
    lines.append(f"  직접효과       = {direct:+.5f} [{direct_ci[0]:+.5f}, {direct_ci[1]:+.5f}]")
    lines.append(f"  → 구속이 설명하는 몫: {share:.1f}%\n")

    lines.append("[참고] 구속 편차 5분위별 피안타율")
    for idx, r in g.iterrows():
        label = str(idx).replace("\n", " ")
        lines.append(f"  {label} (중앙 {r['중앙편차']:+.1f}km/h): "
                     f"{r['피안타율']:.4f} (n={int(r['n']):,})")

    out = "\n".join(lines)
    path = RESULTS / "eda_mediation_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
