"""한계 보강 — 누적 피로를 '소화 이닝'이 아니라 '타석 성적'으로 다시 검정한다.

앞선 검정(발견 02·04)은 결과변수로 소화 이닝을 썼다. 그런데 소화 이닝은 피로의
결과가 아니라 감독의 교체 결정이다. 발견 09에서 확인했듯 교체 확률은 같은
투구수에서도 주자 유무에 따라 2~3배 달라진다 — 피로 신호가 두꺼운 결정 노이즈를
통과해야 보이는 구조였고, 그래서 신뢰구간이 넓었다.

여기서는 결과를 타석 단위 성적(피안타·피출루)으로 바꾼다. 표본이 1,162등판에서
약 26,000타석으로 늘고, 감독 결정을 우회한다. 통제도 함께 강화한다 —
투수 고정효과에 더해 상대 팀 고정효과, 타순, 타순 회전을 넣는다.
"""
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

SEASON = 2024
MIN_PA = 200


def build_start_context() -> pd.DataFrame:
    """등판 단위 피로 이력 — 휴식일, 직전 등판 투구수, 그 시점까지의 시즌 누적 투구수."""
    box = pd.read_parquet(DATA / f"kbo_pitcher_appearances_{SEASON}.parquet")
    s = box[box["is_starter"]].copy()
    s["date"] = pd.to_datetime(s["date"], format="%Y%m%d")
    s["투구수"] = s["투구수"].astype(float)

    key = ["team", "선수명"]
    s = s.sort_values(key + ["date"])
    s["휴식일"] = s.groupby(key)["date"].diff().dt.days
    s["직전_투구수"] = s.groupby(key)["투구수"].shift(1)
    s["등판순번"] = s.groupby(key).cumcount() + 1
    # 이번 등판 직전까지 시즌 동안 던진 총 투구수
    s["시즌누적투구수"] = s.groupby(key)["투구수"].cumsum() - s["투구수"]
    s["평균투구수"] = s.groupby(key)["투구수"].transform("mean")
    s["직전_투구수_편차"] = s["직전_투구수"] - s["평균투구수"]

    return s[["game_id", "선수명", "휴식일", "직전_투구수_편차",
              "시즌누적투구수", "등판순번"]].rename(columns={"선수명": "pitcher"})


def load() -> pd.DataFrame:
    pa = pd.read_parquet(DATA / f"kbo_pa_state_{SEASON}.parquet")
    chk = pd.read_parquet(DATA / f"kbo_state_check_{SEASON}.parquet")
    ok = set(chk.loc[chk["score_ok"] & chk["starter_ok"], "game_id"])
    pa = pa[pa["game_id"].isin(ok) & pa["is_starter"]].copy()

    ctx = build_start_context()
    pa = pa.merge(ctx, on=["game_id", "pitcher"], how="inner")

    pa["안타"] = pa["안타"].astype(int)
    pa["출루"] = (pa["안타"].astype(bool) | (pa["종류"] == "볼넷·사구")).astype(int)
    pa["짧은휴식"] = (pa["휴식일"] == 5).astype(float)
    pa["정상휴식"] = (pa["휴식일"] == 6).astype(float)
    pa["시즌누적_100구당"] = pa["시즌누적투구수"] / 100.0
    pa["직전편차_10구당"] = pa["직전_투구수_편차"] / 10.0

    counts = pa.groupby("pitcher")["안타"].transform("size")
    pa = pa[counts >= MIN_PA]
    # 휴식일 5·6일만 남겨 비교를 깨끗하게 한다(우천 순연 등 변칙 제외).
    return pa.dropna(subset=["휴식일", "직전_투구수_편차"])


def fit(pa: pd.DataFrame, outcome: str) -> dict:
    d = pa.copy()
    design = pd.DataFrame(index=d.index)
    design["짧은휴식"] = d["짧은휴식"]
    design["직전편차_10구당"] = d["직전편차_10구당"]
    design["시즌누적_100구당"] = d["시즌누적_100구당"]
    # 등판 안에서의 진행도 통제 — 이게 빠지면 누적 피로 계수가 이걸 흡수한다.
    design["회전_2"] = (d["타순회전"] == 2).astype(float)
    design["회전_3plus"] = (d["타순회전"] >= 3).astype(float)
    for slot in range(2, 10):
        design[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)

    design = pd.concat([
        design,
        pd.get_dummies(d["pitcher"], prefix="P", drop_first=True).astype(float),
        pd.get_dummies(d["batting_team"], prefix="OPP", drop_first=True).astype(float),
    ], axis=1)
    design = sm.add_constant(design)

    groups = d["game_id"] + "_" + d["pitcher"]
    res = sm.OLS(d[outcome].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": groups})

    out = {"outcome": outcome, "n": int(len(d)), "n_starts": int(groups.nunique())}
    for name in ("짧은휴식", "직전편차_10구당", "시즌누적_100구당"):
        ci = res.conf_int().loc[name]
        out[name] = {"coef": res.params[name], "lo": ci[0], "hi": ci[1],
                     "sig": (ci[0] > 0) == (ci[1] > 0)}
    return out


def fig_effects(results: list[dict]) -> None:
    rows = []
    names = [("짧은휴식", "5일 휴식 (6일 대비)"),
             ("직전편차_10구당", "직전 등판 +10구 (자기 평균 대비)"),
             ("시즌누적_100구당", "시즌 누적 +100구")]
    for res in results:
        metric = {"안타": "피안타율", "출루": "피출루율"}[res["outcome"]]
        color = BLUE if res["outcome"] == "안타" else ORANGE
        for key, label in names:
            rows.append((f"{label}\n→ {metric}", res[key], color))

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=150, facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    for yi, (label, r, color) in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, linewidth=2.4, solid_capstyle="round")
        ax.plot(r["coef"], yi, "o", color=color, markersize=9)
        ax.text(r["hi"] + 0.0018, yi, f"{r['coef']:+.4f}", va="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=9)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2.4, marker="o", label="피안타율"),
               plt.Line2D([], [], color=ORANGE, linewidth=2.4, marker="o", label="피출루율")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")
    ax.set_title("누적 피로를 타석 성적으로 다시 검정 (투수·상대팀 고정효과, 95% 신뢰구간)",
                 color=INK, fontsize=12.5, loc="left", pad=12)
    ax.set_xlabel("타석당 확률 변화", color=SECONDARY_INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / f"{SEASON}_fatigue_pa_effects.png")
    plt.close(fig)


def summarize(pa: pd.DataFrame, results: list[dict]) -> None:
    lines = [f"=== {SEASON} 누적 피로 재검정 — 결과변수를 타석 성적으로 "
             f"(투수 {pa['pitcher'].nunique()}명 / 타석 {len(pa):,}건 / "
             f"등판 {results[0]['n_starts']}건) ===\n"]
    lines.append("통제: 투수 고정효과 · 상대팀 고정효과 · 타순 · 타순 회전, 등판 군집 표준오차\n")

    labels = [("짧은휴식", "5일 휴식 (6일 대비)"),
              ("직전편차_10구당", "직전 등판 자기평균 +10구"),
              ("시즌누적_100구당", "시즌 누적 +100구")]
    for res in results:
        metric = {"안타": "피안타율(타석당 피안타 확률)", "출루": "피출루율"}[res["outcome"]]
        lines.append(f"[{metric}]")
        for key, label in labels:
            r = res[key]
            mark = "유의함" if r["sig"] else "유의하지 않음"
            lines.append(f"  - {label}: {r['coef']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {mark}")
        lines.append("")

    out = "\n".join(lines)
    path = FIG.parent / f"eda_{SEASON}_fatigue_pa_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


def main() -> None:
    pa = load()
    results = [fit(pa, "안타"), fit(pa, "출루")]
    fig_effects(results)
    summarize(pa, results)
    print(f"그림 1장 -> {FIG}")


if __name__ == "__main__":
    main()
