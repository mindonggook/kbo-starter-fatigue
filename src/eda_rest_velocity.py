"""등판 간 피로를 구속으로 다시 검증한다 — 발견 03·04·11 재검정.

지금까지 등판 간 피로를 '소화 이닝'과 '피안타율'로 쟀다. 둘 다 둔한 지표다.
소화 이닝은 감독의 결정이고(발견 09), 피안타율은 타자 적응에 더 크게 반응한다(발견 18).
구속은 투수의 몸 상태를 직접 재므로 훨씬 예민하다.

핵심은 <그날의 출발 구속>을 쓰는 것이다. 등판 초반 25구의 구속은 그 경기의
피로가 아직 쌓이기 전 값이라, 순수하게 '등판에 들어올 때의 몸 상태'를 나타낸다.

  휴식일 5일 vs 6일        -> 발견 03의 미결(피로인가 선제적 관리인가)
  직전 등판 투구수          -> 발견 11(이월 피로 ≈ 0)
  시즌 누적 투구수          -> 발견 04(시즌마다 다름)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).resolve().parent.parent / "data"
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


def start_velocity(seasons: list[int], early_pitches: int = 25) -> pd.DataFrame:
    """등판마다 '출발 구속' — 초반 N구 안의 최고구속 평균."""
    frames = []
    for s in seasons:
        path = DATA / f"naver_pa_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    pa = pd.concat(frames, ignore_index=True)
    pa = pa.dropna(subset=["최고구속", "pitcher"])
    pa = pa[pa["최고구속"].between(100, 170)]

    first_inn = pa[pa["inning"] == 1]
    starters = first_inn.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    pa = pa[[c in starters.get(g, set())
             for g, c in zip(pa["game_id"], pa["pitcher_code"])]].copy()

    early = pa[pa["이전_누적투구수"] < early_pitches]
    out = (early.groupby(["season", "game_id", "pitcher"])
                .agg(출발구속=("최고구속", "mean"), 타석수=("최고구속", "size"))
                .reset_index())
    return out[out["타석수"] >= 3]


def start_context(seasons: list[int]) -> pd.DataFrame:
    """KBO 박스스코어에서 휴식일·직전 투구수·시즌 누적을 만든다."""
    frames = []
    for s in seasons:
        box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{s}.parquet")
        d = box[box["is_starter"]].copy()
        d["season"] = s
        frames.append(d)
    s_all = pd.concat(frames, ignore_index=True)
    s_all["date"] = pd.to_datetime(s_all["date"], format="%Y%m%d")
    s_all["투구수"] = s_all["투구수"].astype(float)

    key = ["team", "선수명"]
    s_all = s_all.sort_values(key + ["date"])
    s_all["휴식일"] = s_all.groupby(key)["date"].diff().dt.days
    s_all["직전_투구수"] = s_all.groupby(key)["투구수"].shift(1)
    s_all["등판순번"] = s_all.groupby(key + ["season"]).cumcount() + 1
    s_all["시즌누적투구수"] = (s_all.groupby(key + ["season"])["투구수"].cumsum()
                          - s_all["투구수"])
    return s_all[["season", "game_id", "선수명", "휴식일", "직전_투구수",
                  "등판순번", "시즌누적투구수", "투구수"]].rename(columns={"선수명": "pitcher"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    vel = start_velocity(a.seasons)
    ctx = start_context(a.seasons)
    d = vel.merge(ctx, on=["season", "game_id", "pitcher"], how="inner")

    # 투수마다 구속 수준이 다르므로 자기 평균 대비 편차로 본다.
    d["구속_편차"] = d["출발구속"] - d.groupby("pitcher")["출발구속"].transform("mean")
    n_before = len(d)
    d = d.dropna(subset=["휴식일", "직전_투구수"])
    # 등판이 적은 투수는 '자기 평균'이 불안정하다.
    cnt = d.groupby("pitcher")["구속_편차"].transform("size")
    d = d[cnt >= 5].copy()

    X = pd.DataFrame(index=d.index)
    X["짧은휴식"] = (d["휴식일"] == 5).astype(float)
    X["긴휴식"] = (d["휴식일"] >= 7).astype(float)
    X["직전투구수_10구당"] = d["직전_투구수"].astype(float) / 10.0
    X["시즌누적_100구당"] = d["시즌누적투구수"].astype(float) / 100.0
    for s in a.seasons[1:]:
        X[f"시즌_{s}"] = (d["season"] == s).astype(float)
    X = sm.add_constant(X)
    fit = sm.OLS(d["구속_편차"].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["pitcher"]})

    def grab(name):
        ci = fit.conf_int().loc[name]
        return fit.params[name], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)

    short = grab("짧은휴식")
    long_ = grab("긴휴식")
    prev = grab("직전투구수_10구당")
    cum = grab("시즌누적_100구당")

    # 그림
    rest_tab = (d[d["휴식일"].between(4, 10)]
                .assign(휴식=lambda x: x["휴식일"].astype(int))
                .groupby("휴식")["구속_편차"].agg(["mean", "size", "std"]))
    rest_tab = rest_tab[rest_tab["size"] >= 20]
    rest_tab["se"] = rest_tab["std"] / np.sqrt(rest_tab["size"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    x = np.arange(len(rest_tab))
    vals = rest_tab["mean"]
    ax.bar(x, vals, color=[BLUE if v >= 0 else ORANGE for v in vals], width=0.55)
    ax.errorbar(x, vals, yerr=rest_tab["se"] * 1.96, fmt="none",
                ecolor=SECONDARY_INK, elinewidth=1.3, capsize=5)
    ax.axhline(0, color=MUTED, linewidth=1.2)
    span = max(abs(vals.min()), abs(vals.max())) or 1
    ax.set_ylim(-span * 2.0, span * 2.0)
    for xi, v, n in zip(x, vals, rest_tab["size"]):
        ax.text(xi, v + (span * 0.15 if v >= 0 else -span * 0.15), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
        ax.text(xi, -span * 1.8, f"n={int(n)}", ha="center", color=MUTED, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}일" for i in rest_tab.index])
    ax.set_title("휴식일별 출발 구속 (자기 평균 대비)", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("km/h", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    rows = [("5일 휴식\n(6일 대비)", short), ("7일 이상 휴식", long_),
            ("직전 등판 +10구", prev), ("시즌 누적 +100구", cum)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r) in zip(y, rows):
        color = ORANGE if r[3] else MUTED
        ax.plot([r[1], r[2]], [yi, yi], color=color, linewidth=2.6, solid_capstyle="round")
        ax.plot(r[0], yi, "o", color=color, markersize=9)
        ax.text(r[2] + 0.02, yi, f"{r[0]:+.3f}", va="center", color=SECONDARY_INK,
                fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r_[0] for r_ in rows], fontsize=9)
    ax.set_title("등판 간 피로가 출발 구속을 떨어뜨리나", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_xlabel("출발 구속 변화 (km/h, 95% 신뢰구간)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("등판 간 피로를 구속으로 다시 재다", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "rest_velocity.png")
    plt.close(fig)

    lines = [f"=== 등판 간 피로를 출발 구속으로 재검정 ({len(a.seasons)}시즌) ===",
             f"등판 {len(d):,}개 / 투수 {d['pitcher'].nunique()}명 "
             f"(초반 25구 안의 최고구속 평균 = 출발 구속, 투수 단위 군집 보정)\n"]

    lines.append("[출발 구속에 대한 효과, km/h]")
    for lab, r in (("5일 휴식 (6일 대비)", short), ("7일 이상 휴식", long_),
                   ("직전 등판 +10구", prev), ("시즌 누적 +100구", cum)):
        mark = "유의함" if r[3] else "유의하지 않음"
        lines.append(f"  {lab}: {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}] {mark}")
    lines.append("")

    lines.append("[참고] 휴식일별 출발 구속 편차")
    for idx, r in rest_tab.iterrows():
        lines.append(f"  {idx}일: {r['mean']:+.3f} km/h (±{r['se']*1.96:.3f}, n={int(r['size'])})")

    out = "\n".join(lines)
    path = DATA.parent / "eda_rest_velocity_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
