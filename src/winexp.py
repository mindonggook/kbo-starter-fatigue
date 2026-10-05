"""승리기대값(WE)과 WPA를 우리 데이터에서 직접 만든다.

득점가치는 '몇 점을 더 주느냐'를 답하지만, 교체 판단은 '이길 확률이 얼마나
바뀌느냐'로 해야 한다. 7점 차에서의 0.36점과 1점 차에서의 0.36점은 전혀 다르다.
1차 평가의 ④번 지적이 바로 이것이었고, 접전만 따로 보는 것으로 대신했었다.

득점기대값과 같은 방식이다 -- 상태를 정하고, 그 상태에서 실제로 어떻게 끝났는지
세면 된다.

    상태 = (이닝, 초/말, 홈 기준 점수차, 아웃, 주자 수)
    WE(상태) = 그 상태에서 홈팀이 이긴 비율
    타석 WPA = WE(다음 상태) - WE(현재 상태),  투수팀 관점으로 부호를 맞춘다

칸을 그냥 세면 극단 상태가 비어 버리므로(9회말 7점차 2사 만루 같은), 부드럽게
잇도록 LightGBM으로 추정한다. 트리는 단조성을 보장하지 않으므로 검산을 붙인다.

최종 승패는 박스스코어의 실점 합으로 구한다 -- 상태 데이터의 마지막 반이닝은
끝점수를 모르기 때문이다.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from runvalue import load_states, attach_end_score

DATA = Path(__file__).resolve().parent.parent / "data"
SEASONS = tuple(range(2017, 2027))
FEATS = ["inning", "말", "점수차_홈", "아웃카운트", "주자수"]


def final_scores(seasons, d):
    """경기별 최종 승패.

    상태 데이터의 끝점수로 구하면 <그 팀의 마지막 공격 이닝>이 빠진다 --
    끝내기 득점이 사라져 9회말 라벨이 통째로 오염됐다. 그래서 총점은
    박스스코어에서 가져온다(투수 실점 합 = 상대 득점).

    마지막 타석의 점수차와 부호가 82%만 맞는 것은 오류가 아니다 --
    점수차는 '그 타석 <전>'의 값이라 끝내기와 막판 득점에서 당연히 뒤집힌다.
    모형이 맞는지는 아래 [1-b]에서 칸별 실제 비율과 직접 대조한다.
    """
    rows = []
    for s_ in seasons:
        p_ = DATA / f"kbo_pitcher_appearances_{s_}.parquet"
        if not p_.exists():
            continue
        a = pd.read_parquet(p_)
        a["실점"] = pd.to_numeric(a["실점"], errors="coerce").fillna(0)
        rows.append(a.groupby(["game_id", "team_side"], as_index=False)["실점"].sum())
    g = pd.concat(rows, ignore_index=True)
    w = g.pivot(index="game_id", columns="team_side", values="실점")
    if not {"home", "away"} <= set(w.columns):
        raise SystemExit(f"team_side 값이 예상과 다르다: {list(w.columns)}")
    out = pd.DataFrame({"홈득점": w["away"], "원정득점": w["home"]})
    out["홈승"] = (out["홈득점"] > out["원정득점"]).astype(float)
    return out[out["홈득점"] != out["원정득점"]]


def build(seasons=SEASONS):
    d = attach_end_score(load_states(seasons))
    fin = final_scores(seasons, d)
    d = d.join(fin, on="game_id", how="inner")

    d["말"] = (d["half"] == "말").astype(float)
    # 점수_타자팀은 공격 중인 팀의 점수다. 홈 기준 점수차로 바꾼다.
    home_score = np.where(d["말"] == 1, d["점수_타자팀"], d["점수_투수팀"])
    away_score = np.where(d["말"] == 1, d["점수_투수팀"], d["점수_타자팀"])
    d["점수차_홈"] = home_score - away_score
    d["inning"] = d["inning"].clip(upper=10)

    model = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.05, num_leaves=63,
                               min_child_samples=200, random_state=20261005, verbose=-1)
    model.fit(d[FEATS].astype(float), d["홈승"])
    d["WE"] = model.predict_proba(d[FEATS].astype(float))[:, 1]

    # 다음 상태 — 같은 반이닝의 다음 타석, 없으면 다음 반이닝의 첫 타석
    d = d.sort_values(["game_id", "seq"])
    g = d.groupby("game_id")
    nxt = g["WE"].shift(-1)
    # 경기 마지막 타석의 다음 상태는 결과 그 자체다.
    d["WE_후"] = nxt.fillna(d["홈승"])
    d["ΔWE_홈"] = d["WE_후"] - d["WE"]
    # 투수팀 관점 — 초에는 홈이 던지고, 말에는 원정이 던진다.
    d["WPA"] = np.where(d["말"] == 1, -d["ΔWE_홈"], d["ΔWE_홈"])
    return d, model


def we_grid(model):
    """검산용 격자 — 상식과 맞는지 눈으로 본다."""
    rows = []
    for inn, half, diff, outs, run in [
            (1, 0, 0, 0, 0), (5, 0, 0, 0, 0), (7, 1, 0, 0, 0),
            (9, 1, 0, 0, 0), (9, 1, 0, 0, 3), (9, 1, -1, 2, 0),
            (9, 1, 1, 2, 0), (9, 1, 3, 0, 0), (9, 0, -3, 0, 0)]:
        x = pd.DataFrame([[inn, half, diff, outs, run]], columns=FEATS).astype(float)
        rows.append((inn, "말" if half else "초", diff, outs, run,
                     float(model.predict_proba(x)[0, 1])))
    return pd.DataFrame(rows, columns=["이닝", "초말", "점수차", "아웃", "주자", "홈 승률"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", type=int, default=list(SEASONS))
    a = p.parse_args()
    d, model = build(a.seasons)

    d[["game_id", "season", "inning", "half", "pitcher", "is_starter",
       "batting_order", "batter", "종류", "타수", "안타", "타순회전",
       "투수_누적투구수", "아웃카운트", "주자수", "점수차_투수팀기준",
       "점수차_홈", "WE", "WPA", "date"]].to_parquet(
        DATA / "kbo_pa_wpa.parquet", index=False)

    L = ["=== 승리기대값과 WPA (10시즌) ===",
         f"타석 {len(d):,}건 · 경기 {d['game_id'].nunique():,}개 (무승부 제외)",
         f"홈 승률 {d['홈승'].mean()*100:.1f}%\n"]

    L.append("[1] 검산 — 모형 추정치 vs 그 칸의 실제 비율 (실제로 존재하는 상태만)")
    for inn, h, lo, hi, lab in [(1, 0, 0, 0, "1회초 동점"), (3, 0, 0, 0, "3회초 동점"),
                                (5, 0, 0, 0, "5회초 동점"), (5, 0, -2, -2, "5회초 홈 2점 뒤짐"),
                                (5, 0, 2, 2, "5회초 홈 2점 앞섬"),
                                (7, 1, 0, 0, "7회말 동점"), (8, 0, -1, -1, "8회초 홈 1점 뒤짐"),
                                (8, 0, 1, 1, "8회초 홈 1점 앞섬"),
                                (9, 0, -3, -3, "9회초 홈 3점 뒤짐"),
                                (9, 0, 3, 3, "9회초 홈 3점 앞섬")]:
        m = d[(d["inning"] == inn) & (d["말"] == h)
              & d["점수차_홈"].between(lo, hi) & (d["아웃카운트"] == 0)]
        if len(m) > 60:
            L.append(f"  {lab:18s} 모형 {m['WE'].mean()*100:5.1f}% · "
                     f"실제 {m['홈승'].mean()*100:5.1f}% (n={len(m):,})")
    L.append("")

    L.append("[2] 검산 — WPA도 망원경처럼 접히는가")
    L.append(f"  전체 평균 WPA: {d['WPA'].mean():+.6f} (0에 가까워야 한다)")
    per_game = d.groupby("game_id")["ΔWE_홈"].sum()
    L.append(f"  경기별 ΔWE 합의 평균: {per_game.mean():+.4f} "
             f"(홈 승률 {d.groupby('game_id')['홈승'].first().mean():.4f} − 시작 승률만큼 나와야 한다)")
    L.append("")

    L.append("[3] 득점가치와 WPA는 어디서 갈리나")
    d["|점수차|"] = d["점수차_투수팀기준"].abs()
    d["점수차_구간"] = pd.cut(d["|점수차|"], [-1, 0, 2, 4, 20],
                          labels=["동점", "1~2점", "3~4점", "5점 이상"])
    rv = pd.read_parquet(DATA / "kbo_pa_runvalue.parquet",
                         columns=["game_id", "inning", "half", "batting_order",
                                  "batter", "득점가치"])
    m = d.merge(rv, on=["game_id", "inning", "half", "batting_order", "batter"],
                how="inner")
    L.append("  타석 하나가 승부에 미치는 영향 (같은 1점이라도 상황마다 다르다)")
    for idx, g in m.groupby("점수차_구간", observed=True):
        corr = g["WPA"].corr(-g["득점가치"])
        slope = np.polyfit(-g["득점가치"], g["WPA"], 1)[0] if len(g) > 100 else np.nan
        L.append(f"    {idx}: 1점당 승률 {slope*100:5.2f}%p · "
                 f"상관 {corr:.3f} (n={len(g):,})")
    L.append("")

    L.append("[4] 5~8회만 — 교체 판단이 벌어지는 구간")
    mid = m[m["inning"].between(5, 8)]
    for idx, g in mid.groupby("점수차_구간", observed=True):
        slope = np.polyfit(-g["득점가치"], g["WPA"], 1)[0] if len(g) > 100 else np.nan
        L.append(f"    {idx}: 1점당 승률 {slope*100:5.2f}%p (n={len(g):,})")
    L.append("")

    path = DATA.parent / "winexp_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
