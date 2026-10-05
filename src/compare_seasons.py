"""2024와 2025를 나란히 놓고 발견들이 재현되는지 확인한다.

2024에서 얻은 결론을 2025에 그대로 적용한다. 방법론은 손대지 않는다 —
검증의 의미는 '건드리지 않은 데이터에서도 같은 값이 나오는가'에 있기 때문이다.

각 지표는 (2024, 2025) 쌍으로 계산하고, 재현 여부는 두 가지로 본다.
  - 방향이 같은가
  - 2025 추정치가 2024의 95% 신뢰구간 안에 들어오는가
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

MIN_STARTS = 10
MIN_PA = 200


# ---------------------------------------------------------------- 공용 통계
def welch(a: pd.Series, b: pd.Series) -> dict:
    a, b = np.asarray(a.dropna(), float), np.asarray(b.dropna(), float)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"est": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
            "n": len(a) + len(b)}


def prop_diff(x1, n1, x2, n2) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"est": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se, "n": int(n1 + n2)}


def standardize(cells: pd.DataFrame, level_col: str, weights: pd.Series) -> pd.DataFrame:
    out = []
    for level, grp in cells.groupby(level_col):
        g = grp.set_index("batting_order")
        common = weights.index.intersection(g.index)
        w = weights.loc[common] / weights.loc[common].sum()
        p = (g.loc[common, "안타"] / g.loc[common, "타수"]).astype(float)
        var = (w ** 2 * p * (1 - p) / g.loc[common, "타수"]).sum()
        out.append({level_col: level, "rate": float((w * p).sum()),
                    "se": float(np.sqrt(var)), "타수": int(g.loc[common, "타수"].sum())})
    return pd.DataFrame(out).set_index(level_col)


# ---------------------------------------------------------------- 데이터
def load_starts(season: int) -> pd.DataFrame:
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    s = box[box["is_starter"]].copy()
    s["date"] = pd.to_datetime(s["date"], format="%Y%m%d")
    for c in ("투구수", "타자", "자책", "실점"):
        s[c] = s[c].astype(float)
    key = ["team", "선수명"]
    s = s.sort_values(key + ["date"])
    s["휴식일"] = s.groupby(key)["date"].diff().dt.days
    s["직전_투구수"] = s.groupby(key)["투구수"].shift(1)
    s["등판순번"] = s.groupby(key).cumcount() + 1
    s["시즌누적투구수"] = s.groupby(key)["투구수"].cumsum() - s["투구수"]
    s["등판수"] = s.groupby(key)["투구수"].transform("size")
    s["조기강판"] = s["이닝_소수"] < 5
    return s


def load_pa(season: int) -> pd.DataFrame | None:
    path = DATA / f"kbo_pa_state_{season}.parquet"
    chk_path = DATA / f"kbo_state_check_{season}.parquet"
    if not path.exists() or not chk_path.exists():
        return None
    pa = pd.read_parquet(path)
    chk = pd.read_parquet(chk_path)
    pa.attrs["check"] = chk
    return pa


# ---------------------------------------------------------------- 지표 계산
def metrics(season: int) -> dict:
    m: dict[str, dict] = {}
    s = load_starts(season)

    # 발견 01 — 조기강판과 그 원인
    early = s[s["조기강판"]]
    m["조기강판 비율"] = {"est": len(early) / len(s), "unit": "비율"}
    blow = ((early["자책"] >= 4).sum()) / len(early)
    burn = (((early["자책"] <= 3) & (early["투구수"] >= 80)).sum()) / len(early)
    m["조기강판 중 부진형"] = {"est": blow, "unit": "비율"}
    m["조기강판 중 소진형"] = {"est": burn, "unit": "비율"}

    # 발견 05 — 교체 임계값
    m["상대 타자 수 중앙값"] = {"est": float(s["타자"].median()), "unit": "명"}
    m["100구 이상 비율"] = {"est": float((s["투구수"] >= 100).mean()), "unit": "비율"}
    m["선발 평균 투구수"] = {"est": float(s["투구수"].mean()), "unit": "구"}

    # 발견 03 — 휴식일 (같은 투수 안에서)
    reg = s[s["등판수"] >= MIN_STARTS].copy()
    key = ["team", "선수명"]
    reg["이닝_편차"] = reg["이닝_소수"] - reg.groupby(key)["이닝_소수"].transform("mean")
    d = reg.dropna(subset=["휴식일"])
    m["5일 대 6일 휴식 (이닝)"] = welch(d[d["휴식일"] == 5]["이닝_편차"],
                                   d[d["휴식일"] == 6]["이닝_편차"])

    # 발견 04 — 시즌 누적
    m["시즌 22~28등판 대 1~21 (이닝)"] = welch(
        reg[reg["등판순번"].between(22, 28)]["이닝_편차"],
        reg[reg["등판순번"].between(1, 21)]["이닝_편차"])

    # 불펜 사정
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{season}.parquet")
    box["date"] = pd.to_datetime(box["date"], format="%Y%m%d")
    box["투구수"] = box["투구수"].astype(float)
    pen = (box[~box["is_starter"]].groupby(["team", "date"])["투구수"].sum()
           .reset_index().rename(columns={"투구수": "불펜"}).sort_values(["team", "date"]))
    pen["직전3"] = (pen.groupby("team")["불펜"].rolling(3, min_periods=3).sum()
                  .reset_index(level=0, drop=True).groupby(pen["team"]).shift(1))
    b = reg.merge(pen[["team", "date", "직전3"]], on=["team", "date"], how="left").dropna(subset=["직전3"])
    if len(b) > 100:
        X = sm.add_constant(pd.concat([
            (b["직전3"] / 10.0).rename("불펜_10구당"),
            pd.get_dummies(b["선수명"], prefix="P", drop_first=True).astype(float)], axis=1))
        fit = sm.OLS(b["투구수"].astype(float).values, X.astype(float).values).fit(cov_type="HC1")
        ci = fit.conf_int()[1]
        m["불펜 +10구 → 선발 투구수"] = {"est": fit.params[1], "lo": ci[0], "hi": ci[1], "n": len(b)}

    # ---- 타석 단위 (문자중계가 있을 때만)
    pa_all = load_pa(season)
    if pa_all is None:
        return m
    chk = pa_all.attrs["check"]

    ok_basic = set(chk.loc[chk["score_ok"] & chk["starter_ok"], "game_id"])
    pa = pa_all[pa_all["game_id"].isin(ok_basic) & pa_all["is_starter"]].copy()
    pa["안타"] = pa["안타"].astype(int)
    pa["타수"] = pa["타수"].astype(bool)
    pa["출루"] = (pa["안타"].astype(bool) | (pa["종류"] == "볼넷·사구")).astype(int)

    m["검증 통과 경기 비율"] = {"est": float(chk["score_ok"].mean()), "unit": "비율"}
    m["리그 타율(파싱 검산)"] = {"est": float(pa_all["안타"].sum() / pa_all["타수"].sum()), "unit": "타율"}

    # 발견 06 — 타순 회전 (생존 편향 + 타순 구성 보정)
    reached3 = (pa[pa["타순회전"] >= 3].groupby(["game_id", "pitcher"]).size()
                .reset_index()[["game_id", "pitcher"]])
    dd = pa.merge(reached3, on=["game_id", "pitcher"], how="inner")
    dd = dd[dd["타순회전"].between(1, 3)]
    cells = (dd.groupby(["타순회전", "batting_order"], observed=True)
               .agg(안타=("안타", "sum"), 타수=("타수", "sum")).reset_index())
    w = cells[cells["타순회전"] == 3].set_index("batting_order")["타수"]
    tbl = standardize(cells, "타순회전", w)
    for t in (1, 2, 3):
        m[f"타순 {t}바퀴째 피안타율"] = {"est": float(tbl.loc[t, "rate"]), "unit": "타율"}
    diff = tbl.loc[3, "rate"] - tbl.loc[1, "rate"]
    se = np.sqrt(tbl.loc[1, "se"] ** 2 + tbl.loc[3, "se"] ** 2)
    m["1바퀴 대비 3바퀴"] = {"est": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
                        "n": int(tbl["타수"].sum())}

    # 발견 07 — 투구수 구간 (90구 이상 간 등판만)
    reached90 = (pa[pa["투수_누적투구수"] >= 90].groupby(["game_id", "pitcher"]).size()
                 .reset_index()[["game_id", "pitcher"]])
    p90 = pa.merge(reached90, on=["game_id", "pitcher"], how="inner").copy()
    p90["구간"] = pd.cut(p90["투수_누적투구수"], [0, 25, 50, 75, 100],
                       labels=["1~25구", "26~50구", "51~75구", "76~100구"])
    g = p90.dropna(subset=["구간"]).groupby("구간", observed=True).agg(
        안타=("안타", "sum"), 타수=("타수", "sum"))
    m["투구수 1~25구 대비 76~100구"] = prop_diff(
        g.loc["1~25구", "안타"], g.loc["1~25구", "타수"],
        g.loc["76~100구", "안타"], g.loc["76~100구", "타수"])

    # 발견 08 — 공선성
    m["회전·투구수 상관계수"] = {"est": float(pa[["타순회전", "투수_누적투구수"]].corr().iloc[0, 1]),
                          "unit": "상관"}
    X = sm.add_constant(pd.get_dummies(
        pa[["타순회전", "batting_order"]].astype(int), columns=["batting_order"],
        drop_first=True).astype(float))
    m["타자순번이 설명하는 투구수 R²"] = {
        "est": float(sm.OLS(pa["투수_누적투구수"].astype(float), X).fit().rsquared), "unit": "R²"}

    # 발견 09 — 강판 시점 (아웃·주자까지 정합한 경기만)
    ok_full = set(chk.loc[chk["score_ok"] & chk["outs_ok"] & chk["runner_ok"]
                          & chk["starter_ok"], "game_id"])
    ps = pa[pa["game_id"].isin(ok_full)].sort_values(["game_id", "pitcher", "inning"]).copy()
    if len(ps) > 1000:
        last = ps.groupby(["game_id", "pitcher"]).tail(1).index
        ps["마지막타석"] = False
        ps.loc[last, "마지막타석"] = True
        hooks = ps[ps["마지막타석"]]
        m["강판 시 지고 있던 비율"] = {"est": float((hooks["점수차_투수팀기준"] < 0).mean()), "unit": "비율"}
        m["강판 시 주자 있던 비율"] = {"est": float((hooks["주자수"] > 0).mean()), "unit": "비율"}
        band = ps[ps["투수_누적투구수"].between(76, 90)]
        no = band[band["주자수"] == 0]["마지막타석"]
        yes = band[band["주자수"] > 0]["마지막타석"]
        m["76~90구 주자 유무 강판율 차"] = prop_diff(no.sum(), len(no), yes.sum(), len(yes))

    # 발견 10 — 누적 피로를 타석 성적으로
    ctx = s[["game_id", "선수명", "휴식일", "직전_투구수", "시즌누적투구수"]].rename(
        columns={"선수명": "pitcher"})
    fp = pa.merge(ctx, on=["game_id", "pitcher"], how="inner")
    fp["평균투구수"] = fp.groupby("pitcher")["투수_누적투구수"].transform("max")
    fp = fp.dropna(subset=["휴식일", "직전_투구수"])
    counts = fp.groupby("pitcher")["안타"].transform("size")
    fp = fp[counts >= MIN_PA]
    if len(fp) > 2000:
        design = pd.DataFrame(index=fp.index)
        design["짧은휴식"] = (fp["휴식일"] == 5).astype(float)
        design["시즌누적_100구당"] = fp["시즌누적투구수"].astype(float) / 100.0
        design["회전_2"] = (fp["타순회전"] == 2).astype(float)
        design["회전_3plus"] = (fp["타순회전"] >= 3).astype(float)
        for slot in range(2, 10):
            design[f"타순_{slot}"] = (fp["batting_order"] == slot).astype(float)
        design = pd.concat([design,
                            pd.get_dummies(fp["pitcher"], prefix="P", drop_first=True).astype(float),
                            pd.get_dummies(fp["batting_team"], prefix="O", drop_first=True).astype(float)],
                           axis=1)
        design = sm.add_constant(design)
        groups = fp["game_id"] + "_" + fp["pitcher"]
        res = sm.OLS(fp["안타"].astype(float), design).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        for name, label in (("짧은휴식", "5일 휴식 → 피안타율"),
                            ("시즌누적_100구당", "시즌 누적 +100구 → 피안타율")):
            ci = res.conf_int().loc[name]
            m[label] = {"est": res.params[name], "lo": ci[0], "hi": ci[1], "n": len(fp)}
    return m


# ---------------------------------------------------------------- 비교 출력
def verdict(a: dict, b: dict) -> str:
    """2025 추정치가 2024 신뢰구간 안에 들어오는지로 재현 여부를 본다."""
    if "lo" not in a:
        rel = abs(b["est"] - a["est"]) / (abs(a["est"]) + 1e-9)
        return "재현" if rel < 0.15 else ("근접" if rel < 0.30 else "차이")
    inside = a["lo"] <= b["est"] <= a["hi"]
    same_sign = (a["est"] >= 0) == (b["est"] >= 0)
    if inside and same_sign:
        return "재현"
    if same_sign:
        return "방향 일치"
    return "불일치"


def fig_compare(rows: list[dict]) -> None:
    picks = [r for r in rows if r["key"] in (
        "타순 1바퀴째 피안타율", "타순 2바퀴째 피안타율", "타순 3바퀴째 피안타율",
        "조기강판 비율", "조기강판 중 부진형", "100구 이상 비율",
        "강판 시 주자 있던 비율", "강판 시 지고 있던 비율", "리그 타율(파싱 검산)")]
    if not picks:
        return
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    y = np.arange(len(picks))[::-1]
    for yi, r in zip(y, picks):
        ax.plot([r["2024"], r["2025"]], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.plot(r["2024"], yi, "o", color=BLUE, markersize=9, zorder=2)
        ax.plot(r["2025"], yi, "o", color=ORANGE, markersize=9, zorder=2)
        ax.text(max(r["2024"], r["2025"]) + 0.012, yi,
                f"{r['2024']:.3f} → {r['2025']:.3f}", va="center",
                color=SECONDARY_INK, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r["key"] for r in picks], fontsize=9)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", linestyle="", label="2024"),
               plt.Line2D([], [], color=ORANGE, marker="o", linestyle="", label="2025")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")
    ax.set_title("2024 vs 2025 — 주요 지표 재현 여부", color=INK, fontsize=13, loc="left", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, max(max(r["2024"], r["2025"]) for r in picks) * 1.45)
    fig.tight_layout()
    fig.savefig(FIG / "compare_2024_2025.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs=2, type=int, default=[2024, 2025])
    a = p.parse_args()
    s1, s2 = a.seasons

    m1, m2 = metrics(s1), metrics(s2)
    rows = []
    for k in m1:
        if k not in m2:
            continue
        row = {"key": k, str(s1): m1[k]["est"], str(s2): m2[k]["est"],
               "판정": verdict(m1[k], m2[k])}
        if "lo" in m1[k]:
            row["2024_ci"] = f"[{m1[k]['lo']:+.4f}, {m1[k]['hi']:+.4f}]"
            row["2025_ci"] = f"[{m2[k]['lo']:+.4f}, {m2[k]['hi']:+.4f}]"
        rows.append(row)

    lines = [f"=== {s1} vs {s2} 재현 검증 ===\n"]
    for r in rows:
        base = f"  {r['key']}: {r[str(s1)]:+.4f} → {r[str(s2)]:+.4f}  [{r['판정']}]"
        lines.append(base)
        if "2024_ci" in r:
            lines.append(f"      {s1} {r['2024_ci']} / {s2} {r['2025_ci']}")
    counts = pd.Series([r["판정"] for r in rows]).value_counts()
    lines.append("\n판정 요약: " + " · ".join(f"{k} {v}건" for k, v in counts.items()))

    out = "\n".join(lines)
    path = DATA.parent / f"compare_{s1}_{s2}.txt"
    path.write_text(out, encoding="utf-8")
    fig_compare(rows)
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
