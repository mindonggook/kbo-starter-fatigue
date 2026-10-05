"""감독은 구속을 보고 교체를 결정하는가.

발견 09는 교체 방아쇠가 '루상의 주자'라고 했지만, 그때는 구속이 없었다.
전광판에 구속이 찍히니 벤치가 본다는 것은 통념인데, 실제로 결정에 반영되는지는
재본 적이 없다.

그리고 이 질문은 발견 18과 묶일 때 날카로워진다 — 구속은 성적을 예측하지
못하는 것으로 드러났다(매개 몫 ≈ 0%). 그런데도 감독이 구속을 보고 내린다면,
<예측력 없는 신호에 반응하는 것>이 된다.

설계: 같은 투구수·같은 주자·같은 아웃 상황에서, 그날 자기 평균보다 구속이
떨어진 투수가 더 빨리 교체되는가. 교체 사건은 등판당 하나뿐이므로 유효 표본은
투구 수가 아니라 등판 수다.
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


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"naver_pa_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    pa = pd.concat(frames, ignore_index=True)
    pa = pa.dropna(subset=["최고구속", "pitcher_code"])
    pa = pa[pa["최고구속"].between(100, 170)].copy()

    first_inn = pa[pa["inning"] == 1]
    starters = first_inn.groupby("game_id")["pitcher_code"].apply(set).to_dict()
    pa["is_starter"] = [c in starters.get(g, set())
                        for g, c in zip(pa["game_id"], pa["pitcher_code"])]
    pa["start"] = pa["game_id"].astype(str) + "_" + pa["pitcher_code"].astype(str)

    d = pa[pa["is_starter"]].sort_values(["start", "타석번호"]).copy()
    last = d.groupby("start").tail(1).index
    d["마지막타석"] = False
    d.loc[last, "마지막타석"] = True

    # 그 등판에서 자기 평균 대비 구속. 당일 컨디션 기준이라 투수 간 차이가 지워진다.
    d["구속_편차"] = d["최고구속"] - d.groupby("start")["최고구속"].transform("mean")
    # 직전 몇 타석의 흐름이 더 눈에 띄는 신호일 수 있다.
    d["구속_최근3"] = (d.groupby("start")["구속_편차"]
                      .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    return d


def fit_hook(d: pd.DataFrame, extra: list[str]) -> sm.regression.linear_model.RegressionResults:
    X = pd.DataFrame(index=d.index)
    for c in extra:
        X[c] = d[c].astype(float)
    X["투구수_10구당"] = d["이전_누적투구수"].astype(float) / 10.0
    X["주자1"] = (d["주자수"] == 1).astype(float)
    X["주자2plus"] = (d["주자수"] >= 2).astype(float)
    X["아웃1"] = (d["아웃카운트"] == 1).astype(float)
    X["아웃2"] = (d["아웃카운트"] == 2).astype(float)
    X["타석투구수"] = d["투구수"].astype(float)
    for slot in range(2, 10):
        X[f"타순_{slot}"] = (d["batting_order"] == slot).astype(float)
    X = sm.add_constant(X)
    return sm.OLS(d["마지막타석"].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["start"]})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    d = load(a.seasons)
    # 교체가 실제로 일어나는 구간만 — 초반에는 거의 안 바꾼다.
    d = d[d["이전_누적투구수"].between(41, 110)].copy()

    # 구속-강판 관계가 U자라 직선을 그으면 양 끝이 뭉개진다.
    # 느린 쪽(하위 25%)만 떼어 중간과 비교한다. 빠른 쪽은 따로 둔다.
    q = d["구속_편차"].quantile([0.25, 0.75])
    d["구속_많이느림"] = (d["구속_편차"] <= q.loc[0.25]).astype(float)
    d["구속_많이빠름"] = (d["구속_편차"] >= q.loc[0.75]).astype(float)
    q3 = d["구속_최근3"].quantile(0.25)
    d["최근3_많이느림"] = (d["구속_최근3"] <= q3).astype(float)

    fit_v = fit_hook(d, ["구속_많이느림", "구속_많이빠름"])
    fit_r = fit_hook(d, ["최근3_많이느림"])

    def grab(fit, name):
        ci = fit.conf_int().loc[name]
        return fit.params[name], ci[0], ci[1], (ci[0] > 0) == (ci[1] > 0)

    v = grab(fit_v, "구속_많이느림")
    vf = grab(fit_v, "구속_많이빠름")
    r = grab(fit_r, "최근3_많이느림")
    runner2 = grab(fit_v, "주자2plus")
    runner1 = grab(fit_v, "주자1")
    pitches = grab(fit_v, "투구수_10구당")

    # 그림 — 구속 편차 구간별 실제 강판율(투구수 구간 안에서)
    d["구속_구간"] = pd.qcut(d["구속_편차"], 4,
                          labels=["많이 느림", "느림", "빠름", "많이 빠름"])
    d["투구수_구간"] = pd.cut(d["이전_누적투구수"], [40, 70, 90, 110],
                           labels=["41~70구", "71~90구", "91~110구"])
    tab = (d.groupby(["투구수_구간", "구속_구간"], observed=True)["마지막타석"]
             .agg(["mean", "size"]).rename(columns={"mean": "강판율", "size": "n"}))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    bands = ["41~70구", "71~90구", "91~110구"]
    width = 0.2
    x = np.arange(len(bands))
    shades = ["#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
    for k, (lab, col) in enumerate(zip(["많이 느림", "느림", "빠름", "많이 빠름"], shades)):
        vals, pos = [], []
        for i, b in enumerate(bands):
            if (b, lab) in tab.index:
                vals.append(tab.loc[(b, lab), "강판율"] * 100)
                pos.append(i + (k - 1.5) * width)
        ax.bar(pos, vals, width=width, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK, ncol=2,
              title="그날 자기 평균 대비 구속", title_fontsize=8.5)
    ax.set_title("구속이 떨어진 날 더 빨리 내려가나", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("이 타석이 마지막일 확률 (%)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    rows = [("구속 하위 25%\n(많이 느림)", v[0], v[1], v[2]),
            ("구속 상위 25%\n(많이 빠름)", vf[0], vf[1], vf[2]),
            ("최근 3타석 하위 25%", r[0], r[1], r[2]),
            ("주자 1명", runner1[0], runner1[1], runner1[2]),
            ("주자 2명 이상", runner2[0], runner2[1], runner2[2]),
            ("투구수 10구 증가", pitches[0], pitches[1], pitches[2])]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, est, lo, hi) in zip(y, rows):
        sig = (lo > 0) == (hi > 0)
        color = ORANGE if sig else MUTED
        ax.plot([lo * 100, hi * 100], [yi, yi], color=color, linewidth=2.6, solid_capstyle="round")
        ax.plot(est * 100, yi, "o", color=color, markersize=9)
        ax.text(hi * 100 + 0.4, yi, f"{est*100:+.2f}", va="center",
                color=SECONDARY_INK, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r_[0] for r_ in rows], fontsize=9)
    ax.set_title("무엇이 교체를 앞당기나 (%p)", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_xlabel("강판 확률 변화 (%p)", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("감독은 구속을 보고 교체하는가", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "hook_velocity.png")
    plt.close(fig)

    lines = [f"=== 감독은 구속을 보고 교체하는가 ({len(a.seasons)}시즌) ===",
             f"선발 타석 {len(d):,}건 / 등판 {d['start'].nunique():,}개 "
             f"(41~110구 구간, 등판 단위 군집 보정)\n"]

    lines.append("[강판 확률에 대한 효과 — 중간 50% 대비, %p 단위]")
    lines.append(f"  구속 하위 25%(많이 느림): {v[0]*100:+.3f}%p "
                 f"[{v[1]*100:+.3f}, {v[2]*100:+.3f}] {'유의함' if v[3] else '유의하지 않음'}")
    lines.append(f"  구속 상위 25%(많이 빠름): {vf[0]*100:+.3f}%p "
                 f"[{vf[1]*100:+.3f}, {vf[2]*100:+.3f}] {'유의함' if vf[3] else '유의하지 않음'}")
    lines.append(f"  최근 3타석 하위 25%     : {r[0]*100:+.3f}%p "
                 f"[{r[1]*100:+.3f}, {r[2]*100:+.3f}] {'유의함' if r[3] else '유의하지 않음'}")
    lines.append(f"  주자 1명               : {runner1[0]*100:+.3f}%p "
                 f"[{runner1[1]*100:+.3f}, {runner1[2]*100:+.3f}]")
    lines.append(f"  주자 2명 이상          : {runner2[0]*100:+.3f}%p "
                 f"[{runner2[1]*100:+.3f}, {runner2[2]*100:+.3f}]")
    lines.append(f"  투구수 10구 증가       : {pitches[0]*100:+.3f}%p "
                 f"[{pitches[1]*100:+.3f}, {pitches[2]*100:+.3f}]\n")

    lines.append("[참고] 투구수 구간 × 구속 구간별 실제 강판율")
    for (band, vel), row in tab.iterrows():
        lines.append(f"  {band} / {vel}: {row['강판율']*100:.1f}% (n={int(row['n']):,})")

    out = "\n".join(lines)
    path = RESULTS / "eda_hook_velocity_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
