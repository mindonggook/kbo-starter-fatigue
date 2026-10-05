"""득점기대값과 선형 가중치를 우리 데이터에서 직접 만든다.

이 프로젝트의 모든 결론이 피안타율 위에 서 있다. 그런데 교체 판단은 실점으로
해야 하고, 피안타율은 삼진과 인플레이 아웃을 똑같이 '아웃'으로 묶어버린다.
발견 28에서 '맞히기 채널이 70%'라고 했을 때 바로 이 문제가 걸렸다 --
삼진이 인플레이 아웃으로 바뀌는 것은 피안타율을 올리지만 실점은 거의 안 올린다.

MLB 가중치를 빌려 오지 않는다. 10시즌 타석 상태 데이터가 있으니 KBO 자체의
득점기대값을 계산할 수 있다.

    상태 = (아웃카운트, 주자수)   -- 12개
    RE(상태) = 그 상태에서 이닝이 끝날 때까지 들어온 평균 득점
    타석 득점가치 = (이 타석에 들어온 점수) + RE(다음 상태) - RE(현재 상태)

주자 수만 쓰는 것은 한계다(1루 주자와 3루 주자가 같게 잡힌다). 상태 추적이
루별 위치까지 남기지 않았기 때문이고, 24상태 대신 12상태로 재는 셈이다.
타석 종류별 평균을 뽑는 데에는 영향이 작다 -- 종류마다 상태 분포가 비슷하므로.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
SEASONS = tuple(range(2017, 2027))


def _half_key(d):
    return (d["game_id"].astype(str) + "_" + d["half"].astype(str) + "_"
            + d["inning"].astype(str))


def load_states(seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = DATA / f"kbo_pa_state_{s}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = s
            frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["아웃카운트", "주자수", "점수_타자팀", "종류"])
    d = d[d["아웃카운트"].between(0, 2) & d["주자수"].between(0, 3)].copy()
    d["half_id"] = _half_key(d)
    d["팀반이닝"] = (d["game_id"].astype(str) + "_" + d["batting_team"].astype(str))
    # 반이닝 안의 시간 순서는 파싱 순서가 유일한 근거다.
    # 투구수로 정렬하면 이닝 중간에 바뀐 구원투수의 타석이 앞으로 튀어 나온다.
    d["seq"] = np.arange(len(d))
    return d


def attach_end_score(d: pd.DataFrame) -> pd.DataFrame:
    """반이닝이 끝났을 때 그 팀의 점수를 붙인다.

    점수 칸은 '타석 시작 시점'의 누적 점수다. 그래서 반이닝의 끝 점수는
    같은 팀이 <다음에> 공격하러 나왔을 때의 시작 점수와 같다.
    마지막 반이닝은 다음이 없으므로 버린다 -- 끝내기·미완성 이닝도 함께 빠진다.
    """
    d = d.sort_values(["팀반이닝", "inning", "seq"])
    half = (d.groupby(["팀반이닝", "inning"], as_index=False)
            .agg(시작점수=("점수_타자팀", "first")))
    half = half.sort_values(["팀반이닝", "inning"])
    half["끝점수"] = half.groupby("팀반이닝")["시작점수"].shift(-1)
    d = d.merge(half[["팀반이닝", "inning", "끝점수"]], on=["팀반이닝", "inning"], how="left")
    return d[d["끝점수"].notna()].copy()


def run_expectancy(d: pd.DataFrame) -> pd.Series:
    """상태별 잔여 득점 기대값."""
    rest = d["끝점수"] - d["점수_타자팀"]
    return rest.groupby([d["아웃카운트"], d["주자수"]]).mean()


def pa_run_value(d: pd.DataFrame, re: pd.Series) -> pd.DataFrame:
    """타석마다 RE24식 득점가치를 계산한다."""
    d = d.sort_values(["half_id", "seq"]).copy()
    g = d.groupby("half_id")
    # 다음 타석의 상태 -- 반이닝이 끝나면 아웃 3개(RE=0)로 본다.
    d["다음아웃"] = g["아웃카운트"].shift(-1)
    d["다음주자"] = g["주자수"].shift(-1)
    d["다음점수"] = g["점수_타자팀"].shift(-1)
    end = d["다음아웃"].isna()
    d.loc[end, "다음점수"] = d.loc[end, "끝점수"]

    idx = pd.MultiIndex.from_arrays([d["아웃카운트"], d["주자수"]])
    d["RE_전"] = re.reindex(idx).to_numpy()
    nxt = pd.MultiIndex.from_arrays([d["다음아웃"].fillna(0), d["다음주자"].fillna(0)])
    d["RE_후"] = np.where(end, 0.0, re.reindex(nxt).to_numpy())

    d["득점"] = (d["다음점수"] - d["점수_타자팀"]).clip(lower=0)
    d["득점가치"] = d["득점"] + d["RE_후"] - d["RE_전"]
    return d


def linear_weights(d: pd.DataFrame) -> pd.DataFrame:
    """타석 종류별 평균 득점가치 = KBO 선형 가중치."""
    w = d.groupby("종류").agg(가중치=("득점가치", "mean"), n=("득점가치", "size"),
                            sd=("득점가치", "std"))
    w["se"] = w["sd"] / np.sqrt(w["n"])
    return w.sort_values("가중치", ascending=False)


def build(seasons=SEASONS):
    d = attach_end_score(load_states(seasons))
    re = run_expectancy(d)
    d = pa_run_value(d, re)
    w = linear_weights(d)
    re.rename("RE").to_frame().to_parquet(DATA / "run_expectancy.parquet")
    w.to_parquet(DATA / "linear_weights.parquet")
    d[["game_id", "season", "inning", "half", "pitcher", "is_starter",
       "batting_order", "batter", "종류", "타수", "안타", "타순회전",
       "투수_누적투구수", "아웃카운트", "주자수", "점수차_투수팀기준",
       "득점가치", "date"]].to_parquet(DATA / "kbo_pa_runvalue.parquet", index=False)
    return re, w, d


if __name__ == "__main__":
    re, w, d = build()
    L = ["=== KBO 득점기대값과 선형 가중치 (2017~2026) ===",
         f"타석 {len(d):,}건 (마지막 반이닝 제외)\n",
         "[1] 득점기대값 RE(아웃, 주자수)"]
    L.append("  아웃  " + "".join(f"{f'주자 {r}':>10s}" for r in range(4)))
    for o in range(3):
        row = "  " + f"{o}   "
        for r in range(4):
            v = re.get((o, r), np.nan)
            row += f"{v:>10.3f}"
        L.append(row)
    L.append("")
    L.append("[2] 선형 가중치 (타석 종류별 평균 득점가치)")
    for idx, r in w.iterrows():
        L.append(f"  {idx:9s}: {r['가중치']:+.4f} 점 (±{r['se']*1.96:.4f}, n={int(r['n']):,})")
    L.append("")
    L.append("[3] 검산")
    L.append(f"  전체 타석 평균 득점가치: {d['득점가치'].mean():+.6f} (0에 가까워야 한다)")
    L.append(f"  타석당 평균 득점: {d['득점'].mean():.4f}")
    L.append(f"  삼진과 인플레이 아웃의 차이: "
             f"{w.loc['삼진','가중치'] - w.loc['인플레이 아웃','가중치']:+.4f} 점")
    L.append(f"  안타와 삼진의 차이      : "
             f"{w.loc['안타','가중치'] - w.loc['삼진','가중치']:+.4f} 점")
    (RESULTS / "runvalue_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("summary -> runvalue_summary.txt")
