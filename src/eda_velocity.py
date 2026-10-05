"""한계 1의 직접 검증 — 투구수가 쌓이면 구속이 떨어지는가.

지금까지는 헛스윙률이라는 대리지표로 '구위 저하가 없다'고 추론했다(발견 13).
네이버 중계의 추적 데이터로 구속을 직접 재면 그 추론을 확인하거나 뒤집을 수 있다.

두 가지를 본다.
  구속          — 구위 그 자체
  릴리스 포인트 흔들림 — 투구 메커니즘이 무너지는지. 피로의 고전적 지표로,
                   구속이 유지돼도 폼이 흔들리면 피로의 증거가 된다.

생존 편향은 여기서도 똑같이 통제한다 — 구속이 떨어진 투수는 애초에 깊은 구간에
도달하지 못하므로, 90구 이상 던진 등판만 남겨 같은 투수 안에서 비교한다.
투수마다 구속 수준이 다르므로 '자기 평균 대비 편차'로 바꾼다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=SECONDARY_INK, fontsize=9.5)
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"naver_pts_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    pts = pd.concat(frames, ignore_index=True)
    pts = pts.dropna(subset=["구속", "pitcher"])
    # 추적 오류로 보이는 극단값 제거
    pts = pts[pts["구속"].between(100, 170)].copy()

    # 선발은 1회에 처음 던진 투수 — 네이버 데이터만으로 식별된다.
    first = (pts.sort_values(["game_id", "inning", "투수_누적투구수"])
                .groupby(["game_id", "inning"]).head(1))
    starters = (first[first["inning"] == 1]
                .groupby("game_id")["pitcher_code"].apply(set).to_dict())
    pts["is_starter"] = [c in starters.get(g, set())
                         for g, c in zip(pts["game_id"], pts["pitcher_code"])]
    return pts


def survivors(pts: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    """그 구간까지 간 등판만 남긴다 — 통제 없이는 생존 편향이 신호를 뒤집는다."""
    reached = (pts[pts["투수_누적투구수"] >= threshold][["game_id", "pitcher_code"]]
               .drop_duplicates())
    return pts.merge(reached, on=["game_id", "pitcher_code"], how="inner")


def demean(d: pd.DataFrame, col: str) -> pd.Series:
    """등판마다 자기 평균을 뺀다 — 투수 간·날짜 간 구속 차이를 지운다."""
    return d[col] - d.groupby(["game_id", "pitcher_code"])[col].transform("mean")


def table_by_bucket(d: pd.DataFrame, col: str) -> pd.DataFrame:
    g = d.groupby("투구수_구간", observed=True)[col]
    out = pd.DataFrame({"평균": g.mean(), "n": g.size(), "sd": g.std()})
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out


def fig_velocity(vel: pd.DataFrame, rel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=150, facecolor=SURFACE)

    for ax, t, label, color, unit in (
            (axes[0], vel, "구속 편차 (자기 등판 평균 대비)", BLUE, "km/h"),
            (axes[1], rel, "릴리스 포인트 흔들림 (등판 평균에서 떨어진 거리)", ORANGE, "cm")):
        x = np.arange(len(t))
        vals = t["평균"]
        ax.bar(x, vals, color=color, width=0.55)
        ax.errorbar(x, vals, yerr=t["se"] * 1.96, fmt="none",
                    ecolor=SECONDARY_INK, elinewidth=1.4, capsize=5)
        ax.axhline(0, color=MUTED, linewidth=1.2)
        span = max(abs(vals.min()), abs(vals.max())) or 1
        ax.set_ylim(min(0, vals.min()) - span * 0.9, max(0, vals.max()) + span * 0.9)
        for xi, v, n in zip(x, vals, t["n"]):
            ax.text(xi, v + (span * 0.12 if v >= 0 else -span * 0.12), f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
            ax.text(xi, min(0, vals.min()) - span * 0.75, f"n={int(n):,}",
                    ha="center", color=MUTED, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(t.index.astype(str), fontsize=9)
        _style_ax(ax, label, "그 투구 시점의 누적 투구수", unit)

    fig.suptitle("투구수가 쌓이면 구위가 떨어지는가 — 구속으로 직접 재다",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "velocity_decline.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024])
    a = p.parse_args()

    pts = load(a.seasons)
    d = survivors(pts[pts["is_starter"]].copy())
    d = d[d["투수_누적투구수"].between(1, 100)].copy()
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], [0, 25, 50, 75, 100],
                           labels=["1~25구", "26~50구", "51~75구", "76~100구"])

    d["구속_편차"] = demean(d, "구속")
    # 릴리스 포인트는 '얼마나 흔들렸나'가 관심사이므로 등판 평균에서의 거리를 쓴다.
    for c in ("release_x", "release_z"):
        d[f"{c}_dev"] = demean(d, c)
    d["릴리스_흔들림"] = np.sqrt(d["release_x_dev"] ** 2 + d["release_z_dev"] ** 2) * 30.48  # ft→cm

    vel = table_by_bucket(d, "구속_편차")
    rel = table_by_bucket(d, "릴리스_흔들림")
    fig_velocity(vel, rel)

    # 같은 등판의 투구들은 독립이 아니다(그날 컨디션을 공유한다).
    # 등판 단위로 군집 보정하지 않으면 신뢰구간이 실제보다 좁게 나온다 —
    # 릴리스 포인트에서는 그 차이가 2.7배였다.
    import statsmodels.api as sm

    d["_start"] = d["game_id"].astype(str) + "_" + d["pitcher_code"].astype(str)
    design = sm.add_constant(pd.get_dummies(d["투구수_구간"], drop_first=True).astype(float))

    def diff(col, last_label="76~100구"):
        fit = sm.OLS(d[col].astype(float), design).fit(
            cov_type="cluster", cov_kwds={"groups": d["_start"]})
        ci = fit.conf_int().loc[last_label]
        est = fit.params[last_label]
        return est, ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)

    lines = [f"=== 구속으로 직접 본 구위 저하 ({len(a.seasons)}시즌 표본) ===",
             f"선발 투구 {len(d):,}개 / {d['game_id'].nunique()}경기 "
             f"(90구 이상 던진 등판만, 등판별 자기 평균 대비 편차)\n"]

    lines.append("[1] 구속 편차 (km/h)")
    for idx, r in vel.iterrows():
        lines.append(f"  {idx}: {r['평균']:+.3f} (±{r['se']*1.96:.3f}, n={int(r['n']):,})")
    dd, lo, hi, sig = diff("구속_편차")
    lines.append(f"  → 1~25구 대비 76~100구: {dd:+.3f} km/h [{lo:+.3f}, {hi:+.3f}] "
                 f"{'유의함' if sig else '유의하지 않음'}\n")

    lines.append("[2] 릴리스 포인트 흔들림 (cm, 등판 평균에서의 거리)")
    for idx, r in rel.iterrows():
        lines.append(f"  {idx}: {r['평균']:.2f} (±{r['se']*1.96:.2f}, n={int(r['n']):,})")
    dd, lo, hi, sig = diff("릴리스_흔들림")
    lines.append(f"  → 1~25구 대비 76~100구: {dd:+.2f} cm [{lo:+.2f}, {hi:+.2f}] "
                 f"{'유의함' if sig else '유의하지 않음'} (등판 단위 군집 보정)")
    lines.append(f"\n등판(군집) 수: {d['_start'].nunique():,}")

    # 시즌마다 따로도 재현되는지 — 한 시즌의 우연이 아님을 확인한다.
    if d["season"].nunique() > 1:
        lines.append("\n[3] 시즌별 재현 (1~25구 대비 76~100구, 군집 보정)")
        for season, g in d.groupby("season"):
            if g["_start"].nunique() < 30:
                lines.append(f"  {season}: 등판 {g['_start'].nunique()}개 — 표본 부족")
                continue
            dz = sm.add_constant(pd.get_dummies(g["투구수_구간"], drop_first=True).astype(float))
            res = []
            for col, unit in (("구속_편차", "km/h"), ("릴리스_흔들림", "cm")):
                fit = sm.OLS(g[col].astype(float), dz).fit(
                    cov_type="cluster", cov_kwds={"groups": g["_start"]})
                ci = fit.conf_int().loc["76~100구"]
                mark = "유의" if (ci[0] > 0) == (ci[1] > 0) else "비유의"
                name = col.replace("_편차", "").replace("_흔들림", "")
                res.append(f"{name} {fit.params['76~100구']:+.2f}{unit} "
                           f"[{ci[0]:+.2f},{ci[1]:+.2f}] {mark}")
            lines.append(f"  {season} (등판 {g['_start'].nunique()}개): " + " · ".join(res))

    out = "\n".join(lines)
    path = DATA.parent / "eda_velocity_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
