"""한계 3 — '주자가 있으면 교체'의 동어반복을 걷어낸다.

발견 09는 같은 투구수에서도 주자가 있으면 강판 확률이 2~3배라고 했지만,
주자가 있다는 것은 이미 출루를 허용했다는 뜻이라 일부는 동어반복이다.

그런데 주자가 '어떻게' 나갔는지는 다른 이야기다. 같은 1·2루라도

    볼넷으로 채운 1·2루  →  제구가 흔들린다는 신호 (투수 쪽 문제)
    안타로 채운 1·2루    →  공이 맞아 나간다는 신호 (대결의 결과)

감독이 둘을 다르게 읽는다면, 교체 결정은 '주자 존재'라는 동어반복을 넘어서는
무언가에 반응하는 것이다. 그래서 주자 수를 고정하고 출루 경로만 바꿔 비교한다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_facecolor(SURFACE)


def load(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        pa_path = DATA / f"kbo_pa_state_{s}.parquet"
        chk_path = DATA / f"kbo_state_check_{s}.parquet"
        if not pa_path.exists():
            continue
        pa = pd.read_parquet(pa_path)
        chk = pd.read_parquet(chk_path)
        ok = set(chk.loc[chk["score_ok"] & chk["outs_ok"] & chk["runner_ok"]
                         & chk["starter_ok"], "game_id"])
        d = pa[pa["game_id"].isin(ok) & pa["is_starter"]].copy()
        d["season"] = s
        frames.append(d)
    pa = pd.concat(frames, ignore_index=True)

    pa = pa.sort_values(["season", "game_id", "inning", "half"]).reset_index(drop=True)
    last = pa.groupby(["season", "game_id", "pitcher"]).tail(1).index
    pa["마지막타석"] = False
    pa.loc[last, "마지막타석"] = True

    # 주자는 반이닝마다 리셋되므로, 그 반이닝 안에서 '지금까지' 어떻게 출루했는지를 센다.
    grp = ["season", "game_id", "inning", "half"]
    pa["이번이닝_안타"] = pa.groupby(grp)["종류"].transform(
        lambda s: (s == "안타").cumsum().shift(1).fillna(0))
    pa["이번이닝_볼넷"] = pa.groupby(grp)["종류"].transform(
        lambda s: (s == "볼넷·사구").cumsum().shift(1).fillna(0))

    # 출루 경로가 섞이지 않은 경우만 써야 해석이 깨끗하다.
    pa["출루경로"] = np.select(
        [(pa["이번이닝_볼넷"] > 0) & (pa["이번이닝_안타"] == 0),
         (pa["이번이닝_안타"] > 0) & (pa["이번이닝_볼넷"] == 0)],
        ["볼넷으로만", "안타로만"], default="혼합")
    return pa


def rate_table(pa: pd.DataFrame) -> pd.DataFrame:
    d = pa[(pa["주자수"] >= 1) & (pa["출루경로"] != "혼합")
           & pa["투수_누적투구수"].between(41, 105)].copy()
    d["투구수_구간"] = pd.cut(d["투수_누적투구수"], [40, 60, 80, 105],
                           labels=["41~60구", "61~80구", "81~105구"])
    d["주자_구간"] = np.where(d["주자수"] >= 2, "주자 2명 이상", "주자 1명")

    g = (d.groupby(["투구수_구간", "주자_구간", "출루경로"], observed=True)["마지막타석"]
           .agg(["mean", "size"]).rename(columns={"mean": "강판율", "size": "타석"}))
    return g[g["타석"] >= 100]


def prop_diff(x1, n1, x2, n2) -> tuple[float, float, float, bool]:
    p1, p2 = x1 / n1, x2 / n2
    d = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return d, lo, hi, (lo > 0) == (hi > 0)


def fig_origin(table: pd.DataFrame) -> None:
    bands = [b for b in ["41~60구", "61~80구", "81~105구"]
             if b in table.index.get_level_values(0)]
    runner_groups = ["주자 1명", "주자 2명 이상"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150, facecolor=SURFACE)
    width = 0.34
    for ax, rg in zip(axes, runner_groups):
        x = np.arange(len(bands))
        for k, (origin, color) in enumerate((("볼넷으로만", BLUE), ("안타로만", ORANGE))):
            vals, pos = [], []
            for i, b in enumerate(bands):
                key = (b, rg, origin)
                if key in table.index:
                    vals.append(table.loc[key, "강판율"] * 100)
                    pos.append(i + (k - 0.5) * width)
            ax.bar(pos, vals, width=width, color=color, label=origin)
            for xi, v in zip(pos, vals):
                ax.text(xi, v + 0.6, f"{v:.1f}", ha="center",
                        color=SECONDARY_INK, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(bands, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="upper left")
        _style_ax(ax, f"{rg}", "그 타석 시점의 누적 투구수",
                  "이 타석이 마지막일 확률 (%)" if rg == "주자 1명" else "")
    fig.suptitle("같은 주자 수라도 볼넷으로 찼을 때와 안타로 찼을 때",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "runner_origin_hook.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int,
                   default=[2021, 2022, 2023, 2024, 2025, 2026])
    a = p.parse_args()

    pa = load(a.seasons)
    table = rate_table(pa)
    fig_origin(table)

    lines = [f"=== 주자 출루 경로별 강판 확률 ({len(a.seasons)}시즌) ===\n",
             "주자 수와 누적 투구수를 고정하고, 그 이닝에 주자가 '어떻게' 나갔는지만 바꿔 비교한다.",
             "출루 경로가 섞인 이닝은 해석이 흐려지므로 제외했다.\n"]

    for rg in ["주자 1명", "주자 2명 이상"]:
        lines.append(f"[{rg}]")
        for band in ["41~60구", "61~80구", "81~105구"]:
            k_bb, k_h = (band, rg, "볼넷으로만"), (band, rg, "안타로만")
            if k_bb not in table.index or k_h not in table.index:
                continue
            bb, h = table.loc[k_bb], table.loc[k_h]
            d, lo, hi, sig = prop_diff(bb["강판율"] * bb["타석"], bb["타석"],
                                       h["강판율"] * h["타석"], h["타석"])
            lines.append(f"  {band}: 볼넷으로만 {bb['강판율']*100:.1f}% (n={int(bb['타석']):,}) "
                         f"vs 안타로만 {h['강판율']*100:.1f}% (n={int(h['타석']):,}) "
                         f"→ {d*100:+.1f}%p [{lo*100:+.1f}, {hi*100:+.1f}] "
                         f"{'유의함' if sig else '유의하지 않음'}")
        lines.append("")

    # 전체를 합쳐 한 번 더 — 투구수·주자수를 층으로 묶어 가중 결합
    d_all = pa[(pa["주자수"] >= 1) & (pa["출루경로"] != "혼합")
               & pa["투수_누적투구수"].between(41, 105)].copy()
    d_all["투구수_구간"] = pd.cut(d_all["투수_누적투구수"], [40, 60, 80, 105])
    num = den = var = 0.0
    for _, g in d_all.groupby(["투구수_구간", "주자수"], observed=True):
        bb = g[g["출루경로"] == "볼넷으로만"]["마지막타석"]
        h = g[g["출루경로"] == "안타로만"]["마지막타석"]
        if len(bb) < 30 or len(h) < 30:
            continue
        p1, p2 = bb.mean(), h.mean()
        n1, n2 = len(bb), len(h)
        w = (n1 * n2) / (n1 + n2)
        num += w * (p2 - p1)
        den += w
        var += w ** 2 * (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = num / den
    se = np.sqrt(var) / den
    lines.append("[층화 결합 — 투구수 구간 × 주자 수를 묶어 전체 비교]")
    lines.append(f"  안타로만 − 볼넷으로만: {diff*100:+.1f}%p "
                 f"[{(diff-1.96*se)*100:+.1f}, {(diff+1.96*se)*100:+.1f}] "
                 f"{'유의함' if (diff-1.96*se > 0) == (diff+1.96*se > 0) else '유의하지 않음'}")

    out = "\n".join(lines)
    path = RESULTS / "eda_runner_origin_summary.txt"
    path.write_text(out, encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
