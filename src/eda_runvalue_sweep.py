"""아직 피안타율로만 쓴 발견들을 득점가치로 전부 다시 잰다.

발견 29에서 교체 조언이 바뀌었다 -- 피안타율이 볼넷을 세지 않아서였다.
한 군데서 결론이 바뀐 것을 봤으니, 피안타율로 쓴 나머지도 확인해야 한다.

여섯 지표를 같은 표본에서 두 자로 나란히 잰다.
  M1 타순 1->3바퀴          (발견 06 / 12)
  M2 투구수 1~25 -> 76~100구 (발견 07 / 12 / 28)
  M3 5일 휴식               (발견 11 / 19)
  M4 시즌 누적 +100구        (발견 11 / 12)
  M5 직전 등판 +10구         (발견 11)
  M6 경기 간 익숙함          (발견 14)

시즌마다 따로 재고 역분산으로 합산하며 Cochran Q로 이질성을 본다 -- 발견 12가
쓴 방식 그대로다. 바뀌는 것이 있으면 그 지표만 리포트를 고치면 된다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

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
SEASONS = tuple(range(2017, 2027))


def load():
    pa = pd.read_parquet(DATA / "kbo_pa_runvalue.parquet")
    ok = set()
    for s in SEASONS:
        path = DATA / f"kbo_state_check_{s}.parquet"
        if path.exists():
            c = pd.read_parquet(path)
            ok |= set(c.loc[c["score_ok"] & c["starter_ok"], "game_id"])
    pa = pa[pa["game_id"].isin(ok) & pa["is_starter"]].copy()
    pa["start"] = pa["game_id"].astype(str) + "_" + pa["pitcher"].astype(str)
    pa["투수시즌"] = pa["pitcher"].astype(str) + "_" + pa["season"].astype(str)
    pa["안타"] = pa["안타"].astype(float)
    pa["타수"] = pa["타수"].astype(bool)

    # 등판 이력 — 휴식일, 직전 등판 투구수 편차, 시즌 누적 투구수
    hist = []
    for s in SEASONS:
        path = DATA / f"kbo_pitcher_appearances_{s}.parquet"
        if not path.exists():
            continue
        app = pd.read_parquet(path)
        st = app[app["is_starter"]].copy()
        st["date"] = pd.to_datetime(st["date"].astype(str), format="%Y%m%d")
        st = st.sort_values(["선수명", "date"])
        g = st.groupby("선수명")
        st["휴식일"] = (st["date"] - g["date"].shift(1)).dt.days
        st["직전_투구수"] = g["투구수"].shift(1)
        st["시즌누적투구수"] = g["투구수"].cumsum() - st["투구수"]
        st["평균투구수"] = g["투구수"].transform("mean")
        st["직전편차_10구당"] = (st["직전_투구수"] - st["평균투구수"]) / 10.0
        st["시즌누적_100구당"] = st["시즌누적투구수"] / 100.0
        st["season"] = s
        hist.append(st[["game_id", "선수명", "season", "휴식일",
                        "직전편차_10구당", "시즌누적_100구당"]])
    h = pd.concat(hist, ignore_index=True).rename(columns={"선수명": "pitcher"})
    pa = pa.merge(h, on=["game_id", "pitcher", "season"], how="left")

    # 경기 간 익숙함 — 이 시즌 이 투수를 몇 번째로 만나는가(이 타석 이전까지)
    pa = pa.sort_values(["season", "pitcher", "batter", "date"])
    pa["쌍"] = (pa["season"].astype(str) + "_" + pa["pitcher"].astype(str)
              + "_" + pa["batter"].astype(str))
    pa["누적대결"] = pa.groupby("쌍").cumcount()
    return pa


def within(d, cols, group):
    out = d[cols].astype(float).copy()
    return out - out.groupby(d[group].to_numpy()).transform("mean")


def est(d, y, x, fe=None, extra=(), cluster="start"):
    """x의 계수와 표준오차. fe가 있으면 집단 평균 차감으로 흡수한다."""
    d = d.dropna(subset=[y, x] + list(extra))
    if len(d) < 300:
        return np.nan, np.nan, 0
    cols = [y, x] + list(extra)
    if fe:
        w = within(d, cols, fe)
    else:
        w = d[cols].astype(float)
    X = w[[x] + list(extra)]
    if not fe:
        X = sm.add_constant(X)
    f = sm.OLS(w[y], X).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    return f.params[x], f.bse[x], len(d)


def pool(rows):
    """역분산 합산과 Cochran Q."""
    r = [x for x in rows if np.isfinite(x[0]) and np.isfinite(x[1]) and x[1] > 0]
    if len(r) < 2:
        return dict(est=np.nan, se=np.nan, q=np.nan, p=np.nan, i2=np.nan,
                    k=len(r), tau2=np.nan, re_est=np.nan, re_se=np.nan)
    b = np.array([x[0] for x in r])
    se = np.array([x[1] for x in r])
    w = 1 / se ** 2
    m = (w * b).sum() / w.sum()
    sem = np.sqrt(1 / w.sum())
    q = (w * (b - m) ** 2).sum()
    df = len(r) - 1
    p = 1 - stats.chi2.cdf(q, df)
    i2 = max(0.0, (q - df) / q * 100) if q > 0 else 0.0
    # 같은 투수가 여러 시즌에 걸쳐 나오므로 시즌 추정치는 완전히 독립이 아니다.
    # 고정효과만 쓰면 신뢰구간이 좁아지므로 랜덤효과(DerSimonian-Laird)도 같이 준다.
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mr = (wr * b).sum() / wr.sum()
    ser = np.sqrt(1 / wr.sum())
    return dict(est=m, se=sem, q=q, p=p, i2=i2, k=len(r),
                tau2=tau2, re_est=mr, re_se=ser)


# ── 여섯 지표의 정의 ────────────────────────────────────────
def m1(d, y):
    """타순 1->3바퀴. 3바퀴까지 간 등판만, 타순 더미 통제."""
    deep = d[d["타순회전"] >= 3]["start"].unique()
    sub = d[d["start"].isin(deep) & d["타순회전"].isin([1, 3])].copy()
    if y == "안타":
        sub = sub[sub["타수"]]
    sub["3바퀴"] = (sub["타순회전"] == 3).astype(float)
    slots = [f"타순_{s}" for s in range(2, 10)]
    for s in range(2, 10):
        sub[f"타순_{s}"] = (sub["batting_order"] == s).astype(float)
    return est(sub, y, "3바퀴", extra=slots)


def m2(d, y):
    """투구수 1~25구 -> 76~100구. 90구 이상 간 등판만."""
    reach = d[d["투수_누적투구수"] >= 90]["start"].unique()
    sub = d[d["start"].isin(reach)].copy()
    sub = sub[sub["투수_누적투구수"].between(1, 25)
              | sub["투수_누적투구수"].between(76, 100)].copy()
    if y == "안타":
        sub = sub[sub["타수"]]
    sub["후반"] = (sub["투수_누적투구수"] >= 76).astype(float)
    slots = [f"타순_{s}" for s in range(2, 10)]
    for s in range(2, 10):
        sub[f"타순_{s}"] = (sub["batting_order"] == s).astype(float)
    return est(sub, y, "후반", extra=slots)


def m3(d, y):
    """5일 휴식(짧은 휴식) 효과. 투수 고정효과."""
    sub = d[d["휴식일"].between(3, 9)].copy()
    if y == "안타":
        sub = sub[sub["타수"]]
    sub["짧은휴식"] = (sub["휴식일"] <= 5).astype(float)
    sub["진행도"] = sub["투수_누적투구수"] / 100.0
    return est(sub, y, "짧은휴식", fe="투수시즌", extra=["진행도"])


def m4(d, y):
    sub = d if y != "안타" else d[d["타수"]]
    sub = sub.copy()
    sub["진행도"] = sub["투수_누적투구수"] / 100.0
    return est(sub, y, "시즌누적_100구당", fe="투수시즌", extra=["진행도"])


def m5(d, y):
    sub = d if y != "안타" else d[d["타수"]]
    sub = sub.copy()
    sub["진행도"] = sub["투수_누적투구수"] / 100.0
    return est(sub, y, "직전편차_10구당", fe="투수시즌", extra=["진행도"])


def m6(d, y, pair_col="쌍", order_col="누적대결"):
    """경기 간 익숙함 — 발견 14가 쓴 추정량 그대로.

    쌍마다 대결 순번의 중앙값으로 초기/후기를 가르고, 양쪽에 3건 이상 있는 쌍만
    써서 (후기 평균 - 초기 평균)을 n1*n2/(n1+n2)로 가중 결합한다.
    """
    sub = d[d["타순회전"] == 1].copy()
    if y == "안타":
        sub = sub[sub["타수"]]
    cnt = sub.groupby(pair_col)[y].transform("size")
    sub = sub[cnt >= 8].copy()          # 발견 14의 MIN_PAIR_PA와 같다
    if sub.empty:
        return np.nan, np.nan, 0
    med = sub.groupby(pair_col)[order_col].transform("median")
    sub["후기"] = (sub[order_col] > med).astype(int)

    num = den = var = 0.0
    used = 0
    for _, g in sub.groupby(pair_col, sort=False):
        e = g.loc[g["후기"] == 0, y].to_numpy(dtype=float)
        l = g.loc[g["후기"] == 1, y].to_numpy(dtype=float)
        if len(e) < 3 or len(l) < 3:
            continue
        w = len(e) * len(l) / (len(e) + len(l))
        num += w * (l.mean() - e.mean())
        den += w
        var += w ** 2 * (e.var(ddof=1) / len(e) + l.var(ddof=1) / len(l))
        used += 1
    if used < 30 or den == 0:
        return np.nan, np.nan, used
    return num / den, np.sqrt(var) / den, used


def m6_slope(d, y):
    """참고 — 대결 1회당 기울기(쌍 고정효과). 같은 질문을 다른 자로 잰 것."""
    sub = d[d["타순회전"] == 1].copy()
    if y == "안타":
        sub = sub[sub["타수"]]
    cnt = sub.groupby("쌍")[y].transform("size")
    sub = sub[cnt >= 3].copy()
    return est(sub, y, "누적대결", fe="쌍")


METRICS = [
    ("M1", "타순 1→3바퀴", m1, 1000, "/1000"),
    ("M2", "투구수 1~25 → 76~100구", m2, 1000, "/1000"),
    ("M3", "5일 이하 휴식", m3, 1000, "/1000"),
    ("M4", "시즌 누적 +100구", m4, 1000, "/1000"),
    ("M5", "직전 등판 +10구", m5, 1000, "/1000"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(SEASONS))
    a = p.parse_args()

    pa = load()
    pa = pa[pa["season"].isin(a.seasons)]
    L = [f"=== 피안타율로 쓴 발견들을 득점가치로 재검증 ({len(a.seasons)}시즌) ===",
         f"선발 타석 {len(pa):,}건",
         "각 지표를 같은 표본에서 두 자로 재고, 시즌별 추정치를 역분산으로 합산했다.\n"]

    results = {}
    for code, label, fn, scale, unit in METRICS:
        L.append(f"[{code}] {label}")
        row = {}
        for y, ylab in (("안타", "피안타율"), ("득점가치", "득점가치")):
            per = []
            for s in a.seasons:
                sub = pa[pa["season"] == s]
                per.append(fn(sub, y))
            pooled = pool(per)
            row[y] = dict(pooled=pooled, per=per)
            sig = "유의함" if abs(pooled["est"]) > 1.96 * pooled["se"] else "유의하지 않음"
            L.append(f"  {ylab:6s}: {pooled['est']*scale:+.2f}{unit} "
                     f"[{(pooled['est']-1.96*pooled['se'])*scale:+.2f}, "
                     f"{(pooled['est']+1.96*pooled['se'])*scale:+.2f}] {sig} "
                     f"· I²={pooled['i2']:.0f}% (Q p={pooled['p']:.3f}, k={pooled['k']})")
            L.append(f"          랜덤효과: {pooled['re_est']*scale:+.2f}{unit} "
                     f"[{(pooled['re_est']-1.96*pooled['re_se'])*scale:+.2f}, "
                     f"{(pooled['re_est']+1.96*pooled['re_se'])*scale:+.2f}] "
                     f"· τ²={pooled['tau2']*scale**2:.2f}")
        # 두 자의 결론이 같은가
        a_sig = abs(row["안타"]["pooled"]["est"]) > 1.96 * row["안타"]["pooled"]["se"]
        r_sig = abs(row["득점가치"]["pooled"]["est"]) > 1.96 * row["득점가치"]["pooled"]["se"]
        a_pos = row["안타"]["pooled"]["est"] > 0
        r_pos = row["득점가치"]["pooled"]["est"] > 0
        if a_sig != r_sig:
            verdict = "★ 유의성이 바뀐다"
        elif a_sig and r_sig and a_pos != r_pos:
            verdict = "★ 부호가 바뀐다"
        else:
            verdict = "결론 동일"
        L.append(f"  → {verdict}")
        results[code] = (label, row, verdict)
        L.append("")

    # M6 — 쌍은 10시즌에 걸쳐 쌓이므로 시즌별로 나누지 않고 한 번에 잰다.
    L.append("[M6] 경기 간 익숙함 (발견 14와 같은 추정량 · 쌍은 10시즌 누적)")
    pa["쌍전체"] = pa["pitcher"].astype(str) + "_" + pa["batter"].astype(str)
    pa["대결순번_전체"] = pa.sort_values(["date", "game_id", "inning"]).groupby(
        "쌍전체").cumcount().reindex(pa.index)
    for vname, pair_col, order_col in (
            ("시즌 경계 없이", "쌍전체", "대결순번_전체"),
            ("같은 시즌 안에서만", "쌍전체", "누적대결")):
        L.append(f"  [{vname}]")
        for y, ylab in (("안타", "피안타율"), ("득점가치", "득점가치")):
            e, se, k = m6(pa, y, pair_col, order_col)
            if not np.isfinite(e):
                L.append(f"    {ylab}: 쌍이 모자라 추정 불가")
                continue
            sig = "유의함" if abs(e) > 1.96 * se else "유의하지 않음"
            L.append(f"    {ylab:6s}: {e*1000:+.2f}/1000 "
                     f"[{(e-1.96*se)*1000:+.2f}, {(e+1.96*se)*1000:+.2f}] {sig} "
                     f"(쌍 {k:,}개)")
    L.append("  참고 — 같은 질문을 '대결 1회당 기울기'로 재면 (시즌별 합산)")
    for y, ylab in (("안타", "피안타율"), ("득점가치", "득점가치")):
        per = [m6_slope(pa[pa["season"] == s], y) for s in a.seasons]
        pooled = pool(per)
        sig = "유의함" if abs(pooled["est"]) > 1.96 * pooled["se"] else "유의하지 않음"
        L.append(f"    {ylab:6s}: {pooled['est']*1000:+.2f}/1000 per 1회 "
                 f"[{(pooled['est']-1.96*pooled['se'])*1000:+.2f}, "
                 f"{(pooled['est']+1.96*pooled['se'])*1000:+.2f}] {sig}")
    L.append("")

    # 왜 대부분 안 바뀌는가 — 볼넷 비중이 축을 따라 얼마나 변하는지
    L.append("[진단] 피안타율이 깨지는 조건 — 볼넷 비중이 축을 따라 변하는가")
    bb = pa["종류"] == "볼넷·사구"
    deep = pa[pa["타순회전"] >= 3]["start"].unique()
    t = pa[pa["start"].isin(deep) & pa["타순회전"].isin([1, 3])]
    L.append(f"  타순 1바퀴 {bb[t.index][t['타순회전']==1].mean()*100:.2f}% → "
             f"3바퀴 {bb[t.index][t['타순회전']==3].mean()*100:.2f}%  "
             f"(차이 {(bb[t.index][t['타순회전']==3].mean()-bb[t.index][t['타순회전']==1].mean())*100:+.2f}%p)")
    reach = pa[pa["투수_누적투구수"] >= 90]["start"].unique()
    q = pa[pa["start"].isin(reach)]
    lo = q["투수_누적투구수"].between(1, 25)
    hi = q["투수_누적투구수"].between(76, 100)
    L.append(f"  투구수 1~25구 {bb[q.index][lo].mean()*100:.2f}% → "
             f"76~100구 {bb[q.index][hi].mean()*100:.2f}%  "
             f"(차이 {(bb[q.index][hi].mean()-bb[q.index][lo].mean())*100:+.2f}%p)")
    L.append("  비교: 선발 3바퀴 이상 9.06% → 구원 추격조 13.07% (차이 +4.01%p, 발견 29)")
    L.append("  → 선발 안의 축에서는 볼넷 비중이 거의 안 변해서 피안타율이 실점을 잘 대리한다.")
    L.append("    투수의 <종류>가 바뀌는 비교에서만 깨진다.")
    L.append("")

    # 비율 — 피안타율 1포인트가 몇 점인가
    L.append("[환산율] 지표마다 피안타율 1/1000이 득점가치 몇 /1000점인가")
    for code, (label, row, _) in results.items():
        ha = row["안타"]["pooled"]["est"]
        rv = row["득점가치"]["pooled"]["est"]
        if abs(ha) > 1e-9:
            L.append(f"  {code} {label:24s}: {rv/ha:+.3f}")
    L.append("  (1.0에 가까우면 피안타율이 실점을 잘 대리한 것, "
             "크면 피안타율이 놓친 것이 있다는 뜻이다)")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10.6, 5.4), dpi=150, facecolor=SURFACE)
    y = np.arange(len(METRICS))[::-1]
    for yi, (code, label, *_rest) in zip(y, METRICS):
        row = results[code][1]
        for off, (key, color, lab) in ((0.17, ("안타", MUTED, "피안타율")),
                                       (-0.17, ("득점가치", BLUE, "득점가치"))):
            pooled = row[key]["pooled"]
            # 각 지표의 피안타율 추정치 크기로 정규화해 한 그림에 올린다.
            base = abs(row["안타"]["pooled"]["est"]) or 1.0
            e, se = pooled["est"] / base, pooled["se"] / base
            ax.errorbar(e, yi + off, xerr=1.96 * se, fmt="o", color=color,
                        markersize=6.5, elinewidth=1.6, capsize=4,
                        label=lab if yi == y[0] else None)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.axvline(1, color=GRID, linewidth=1.2, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in METRICS], fontsize=9.5)
    ax.set_xlabel("각 지표의 피안타율 추정치를 1로 놓은 상대 크기",
                  color=SECONDARY_INK, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")
    ax.set_title("같은 발견을 두 자로 재면", color=INK, fontsize=13, loc="left", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / "runvalue_sweep.png")
    plt.close(fig)

    path = DATA.parent / "eda_runvalue_sweep_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
