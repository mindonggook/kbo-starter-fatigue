"""발견 01을 강판 시점 실점으로 다시 분류한다.

외부 평가의 다섯 번째 지적이다. 조기강판을 '부진형 / 소진형 / 극단형'으로 가를 때
<자책점>을 썼는데, 자책점에는 선발이 내려간 뒤 구원투수가 홈으로 들여보낸
승계 주자 점수가 포함된다. 그러면 강판 시점에는 3점이었다가 나중에 4점이 된 등판이
'부진형'으로 잡힌다 -- 감독이 보지 못한 정보로 감독의 결정을 분류한 셈이다.

상태 데이터에 타석마다의 점수가 있으므로 <그 투수가 마운드를 떠나는 순간>의 실점을
직접 구할 수 있다. 세 가지를 나란히 둔다.

  자책(전체)      원래 쓴 값. 강판 이후가 섞인다.
  실점(전체)      박스스코어. 자책에 비자책이 더해진 값.
  실점(강판 시점)  상태 데이터. 감독이 결정할 때 실제로 보고 있던 숫자.

'실점(전체) - 실점(강판 시점)'이 곧 승계 주자가 들어온 점수다.
그리고 평가가 지적한 대로 '혼합형'의 정의와 건수도 명시한다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runvalue import load_states, attach_end_score

DATA = Path(__file__).resolve().parent.parent / "data"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#7b5bd6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

LABELS = ["극단형(부상·오프너 의심)", "소진형(투구수 소모)",
          "부진형(피실점)", "혼합형(어느 쪽도 아님)"]
COLORS = {LABELS[0]: AQUA, LABELS[1]: ORANGE, LABELS[2]: BLUE, LABELS[3]: VIOLET}


def hook_runs(seasons):
    """선발이 마운드를 떠나는 순간까지 그 팀이 내준 점수."""
    d = attach_end_score(load_states(seasons))
    d = d.sort_values(["half_id", "seq"])
    nxt = d.groupby("half_id")["점수_타자팀"].shift(-1)
    d["타석후_점수"] = nxt.fillna(d["끝점수"])

    st = d[d["is_starter"]].copy()
    st["start"] = st["game_id"].astype(str) + "_" + st["pitcher"].astype(str)
    last = st.loc[st.groupby("start")["seq"].idxmax()]
    return last[["game_id", "pitcher", "season", "타석후_점수"]].rename(
        columns={"타석후_점수": "강판시점실점", "pitcher": "선수명"})


def load_starts(seasons):
    rows = []
    for s in seasons:
        path = DATA / f"kbo_pitcher_appearances_{s}.parquet"
        if not path.exists():
            continue
        app = pd.read_parquet(path)
        a = app[app["is_starter"]].copy()
        a["season"] = s
        rows.append(a)
    st = pd.concat(rows, ignore_index=True)
    for c in ("자책", "실점", "투구수", "이닝_소수"):
        st[c] = pd.to_numeric(st[c], errors="coerce")
    return st.dropna(subset=["자책", "실점", "투구수", "이닝_소수"])


def classify(early, runs_col):
    cond = [early["투구수"] < 40,
            (early[runs_col] <= 3) & (early["투구수"] >= 80),
            early[runs_col] >= 4]
    return pd.Series(np.select(cond, LABELS[:3], default=LABELS[3]), index=early.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2017, 2027)))
    a = p.parse_args()

    st = load_starts(a.seasons)
    hr = hook_runs(a.seasons)
    st = st.merge(hr, on=["game_id", "선수명", "season"], how="inner")
    st["조기강판"] = st["이닝_소수"] < 5
    early = st[st["조기강판"]].copy()

    L = [f"=== 발견 01 재분류 — 강판 시점 실점으로 ({len(a.seasons)}시즌) ===",
         f"선발 등판 {len(st):,}건 · 5이닝 미만 조기강판 {len(early):,}건 "
         f"({len(early)/len(st)*100:.1f}%)\n"]

    # ── [1] 세 가지 실점 지표의 차이 ───────────────────────────
    L.append("[1] 무엇을 '실점'으로 세느냐")
    early["승계실점"] = early["실점"] - early["강판시점실점"]
    for c, lab in (("자책", "자책(전체)"), ("실점", "실점(전체)"),
                   ("강판시점실점", "실점(강판 시점)")):
        L.append(f"  {lab:16s} 평균 {early[c].mean():.3f} · 중앙 {early[c].median():.1f}")
    n_inh = (early["승계실점"] > 0).sum()
    L.append(f"  승계 주자가 들어온 등판: {n_inh:,}건 ({n_inh/len(early)*100:.1f}%) · "
             f"그 경우 평균 {early.loc[early['승계실점']>0,'승계실점'].mean():.2f}점")
    L.append(f"  → '실점(전체) − 실점(강판 시점)'이 선발이 내려간 뒤 들어온 점수다.\n")

    # ── [2] 분류가 얼마나 바뀌나 ───────────────────────────────
    early["유형_원래"] = classify(early, "자책")
    early["유형_실점전체"] = classify(early, "실점")
    early["유형_보정"] = classify(early, "강판시점실점")
    L.append("[2] 분류 결과 비교")
    L.append("  " + "유형".ljust(24) + f"{'원래(자책)':>12s}{'보정(강판시점)':>14s}{'차이':>8s}")
    for lab in LABELS:
        o = int((early["유형_원래"] == lab).sum())
        n = int((early["유형_보정"] == lab).sum())
        L.append("  " + lab.ljust(24) + f"{o:>12,}{n:>14,}{n-o:>+8,}")
    moved = (early["유형_원래"] != early["유형_보정"]).sum()
    L.append(f"  → 유형이 바뀐 등판 {moved:,}건 ({moved/len(early)*100:.1f}%)\n")

    L.append("  어디서 어디로 옮겨갔나 (행=원래, 열=보정)")
    ct = pd.crosstab(early["유형_원래"], early["유형_보정"])
    ct = ct.reindex(index=LABELS, columns=LABELS, fill_value=0)
    L.append("    " + "".join(f"{c[:6]:>9s}" for c in LABELS))
    for idx, r in ct.iterrows():
        L.append(f"    {idx[:6]:6s}" + "".join(f"{v:>9,}" for v in r))
    # 보정은 두 가지를 동시에 바꾼다 — 자책/실점 구분과, 전체/강판시점 구분.
    # 승계 주자 효과만 떼려면 '실점(전체)'과 '실점(강판 시점)'을 맞대야 한다.
    moved_t = (early["유형_실점전체"] != early["유형_보정"]).sum()
    moved_e = (early["유형_원래"] != early["유형_실점전체"]).sum()
    L.append("  보정은 두 가지를 한꺼번에 바꾼다 — 그래서 따로 센다")
    L.append(f"    자책(전체) → 실점(전체)        : {moved_e:,}건 "
             f"({moved_e/len(early)*100:.1f}%)  [비자책을 세느냐의 차이]")
    L.append(f"    실점(전체) → 실점(강판 시점)   : {moved_t:,}건 "
             f"({moved_t/len(early)*100:.1f}%)  [평가가 지적한 승계 주자 효과]")
    sh_t = early["유형_실점전체"].value_counts(normalize=True) * 100
    sh_b = early["유형_보정"].value_counts(normalize=True) * 100
    L.append("    부진형 비중: 실점(전체) 기준 "
             f"{sh_t.get(LABELS[2], 0):.1f}% → 강판 시점 기준 {sh_b.get(LABELS[2], 0):.1f}% "
             f"({sh_b.get(LABELS[2], 0) - sh_t.get(LABELS[2], 0):+.1f}%p)")
    L.append("")

    # ── [3] 혼합형의 정의와 내용 ───────────────────────────────
    L.append("[3] 혼합형이란 무엇인가 (평가가 지적한 누락)")
    L.append("  정의: 투구수 40~79구이면서 실점 3점 이하 — 깊이 가지도, 두들겨 맞지도 않은 등판")
    mix = early[early["유형_보정"] == LABELS[3]]
    L.append(f"  {len(mix):,}건 ({len(mix)/len(early)*100:.1f}%) · "
             f"평균 투구수 {mix['투구수'].mean():.1f} · "
             f"평균 이닝 {mix['이닝_소수'].mean():.2f} · "
             f"평균 실점 {mix['강판시점실점'].mean():.2f}")
    L.append("")

    # ── [4] 결론 수치가 바뀌는가 ───────────────────────────────
    L.append("[4] 발견 01의 결론 — '조기강판의 주된 이유는 부진이다'는 유지되나")
    for col, lab in (("유형_원래", "원래"), ("유형_보정", "보정")):
        sh = early[col].value_counts(normalize=True) * 100
        L.append(f"  [{lab}] " + " / ".join(f"{k[:6]} {sh.get(k, 0):.1f}%" for k in LABELS))
    L.append("")

    # 2024만 따로 — 발견 01이 보고한 시즌
    if 2024 in a.seasons:
        e24 = early[early["season"] == 2024]
        L.append(f"[5] 2024 시즌만 (발견 01이 보고한 표본 · {len(e24):,}건)")
        for col, lab in (("유형_원래", "원래"), ("유형_보정", "보정")):
            sh = e24[col].value_counts(normalize=True) * 100
            L.append(f"  [{lab}] " + " / ".join(
                f"{k[:6]} {sh.get(k, 0):.1f}%" for k in LABELS))
        L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    y = np.arange(len(LABELS))[::-1]
    w = 0.38
    for off, (col, color, lab) in ((w / 2, ("유형_원래", MUTED, "원래 (자책 전체)")),
                                   (-w / 2, ("유형_보정", BLUE, "보정 (강판 시점 실점)"))):
        vals = [int((early[col] == k).sum()) for k in LABELS]
        ax.barh(y + off, vals, height=w, color=color, label=lab)
        for yi, v in zip(y + off, vals):
            ax.text(v + len(early) * 0.008, yi, f"{v:,}", va="center",
                    color=SECONDARY_INK, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([k.split("(")[0] for k in LABELS], fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK, loc="lower right")
    ax.set_title("분류 기준을 바꾸면", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_xlabel("조기강판 등판 수", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[1]
    sub = early.sample(min(4000, len(early)), random_state=0)
    for lab in LABELS:
        g = sub[sub["유형_보정"] == lab]
        ax.scatter(g["투구수"], g["강판시점실점"], s=18, alpha=0.55,
                   color=COLORS[lab], label=lab.split("(")[0], linewidths=0)
    ax.axvline(40, color=GRID, linewidth=1.2, linestyle="--")
    ax.axvline(80, color=GRID, linewidth=1.2, linestyle="--")
    ax.axhline(3.5, color=GRID, linewidth=1.2, linestyle="--")
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, ncol=2)
    ax.set_title("강판 시점 실점으로 본 조기강판", color=INK, fontsize=12.5,
                 loc="left", pad=10)
    ax.set_xlabel("투구수", color=SECONDARY_INK, fontsize=9.5)
    ax.set_ylabel("강판 시점 실점", color=SECONDARY_INK, fontsize=9.5)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("감독이 보고 있던 숫자로 다시 분류하다", color=INK, fontsize=14,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "hook_types_fix.png")
    plt.close(fig)

    path = DATA.parent / "eda_hook_types_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
