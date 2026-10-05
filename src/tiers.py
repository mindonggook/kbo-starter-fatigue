"""불펜 보직 등급 — 결과 누수를 없앤 정의.

외부 평가에서 지적받았다. 같은 시즌의 세이브·홀드로 등급을 매기면
'그 시즌에 잘 막은 투수'가 필승조로 분류된 뒤, 같은 시즌 성적으로
'필승조가 잘 막는다'를 확인하는 꼴이다. 타깃 누수다.

두 가지 정의를 모두 만들어 나란히 비교할 수 있게 한다.

  same  같은 시즌 세이브+홀드 (원래 정의 · 누수 있음)
  prev  직전 시즌 세이브+홀드 (누수 없음)

prev는 직전 시즌 기록이 있어야 하므로 첫 시즌과 신인이 빠진다. 그 손실이
얼마인지도 같이 보고한다. 선수는 팀을 옮기므로 이름으로만 잇는다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TIERS = ["추격조", "중간", "필승조"]


def _success_by_season(seasons):
    """시즌·선수별 세이브+홀드 수."""
    rows = []
    for s in seasons:
        path = DATA / f"kbo_pitcher_appearances_{s}.parquet"
        if not path.exists():
            continue
        box = pd.read_parquet(path)
        rel = box[~box["is_starter"]].copy()
        rel["결과"] = rel["결과"].fillna("")
        rel["성공"] = rel["결과"].str.contains("세|홀드", regex=True).astype(int)
        g = rel.groupby(["team", "선수명"], as_index=False)["성공"].sum()
        g["등판수"] = rel.groupby(["team", "선수명"]).size().to_numpy()
        g["season"] = s
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def _grade(n):
    return np.select([n >= 15, n >= 5], ["필승조", "중간"], default="추격조")


def build_pit(seasons, window=365, min_days=60):
    """시점 기준 누적 등급 — 그 등판 <직전까지> 쌓인 세이브+홀드로만 매긴다.

    직전 시즌 정의는 신인과 첫 시즌을 통째로 버린다(33.2%). 대신 등판 날짜마다
    최근 365일 창의 누적 성공 수를 세면 누수도 없고 표본도 지킨다.
    금융 백테스트에서 '그날 공개된 자료만 쓰는 것'(point-in-time)과 같은 생각이다.

    반환 단위가 다르다 -- (season, team, pitcher)가 아니라 <등판 한 건>마다의 등급이다.
    """
    need = sorted(set(seasons) | {min(seasons) - 1})
    rows = []
    for s in need:
        path = DATA / f"kbo_pitcher_appearances_{s}.parquet"
        if not path.exists():
            continue
        box = pd.read_parquet(path)
        rel = box[~box["is_starter"]].copy()
        rel["결과"] = rel["결과"].fillna("")
        rel["성공"] = rel["결과"].str.contains("세|홀드", regex=True).astype(int)
        rel["date"] = pd.to_datetime(rel["date"].astype(str), format="%Y%m%d")
        rel["season"] = s
        rows.append(rel[["game_id", "team", "선수명", "date", "성공", "season"]])
    app = pd.concat(rows, ignore_index=True).sort_values(["선수명", "date"])

    out = []
    for name, g in app.groupby("선수명", sort=False):
        d = g["date"].to_numpy()
        cum = np.concatenate([[0], g["성공"].to_numpy().cumsum()])
        lo = np.searchsorted(d, d - np.timedelta64(window, "D"), side="left")
        # 자기 자신은 빼야 한다 -- 그 등판의 결과를 쓰면 누수다.
        prior = cum[np.arange(len(d))] - cum[lo]
        first = d[0]
        enough = (d - first) / np.timedelta64(1, "D") >= min_days
        out.append(g.assign(창누적=prior, 자격=enough))
    t = pd.concat(out, ignore_index=True)
    t["등급"] = np.where(t["자격"], _grade(t["창누적"]), None)
    t = t[t["season"].isin(seasons)]
    return t.rename(columns={"team": "pitcher_team", "선수명": "pitcher"})[
        ["game_id", "season", "pitcher_team", "pitcher", "등급", "창누적"]]


def build(seasons, mode="prev"):
    """(season, team, pitcher) -> 등급.

    mode="same"이면 그 시즌 기록, "prev"면 직전 시즌 기록으로 매긴다.
    """
    need = sorted(set(seasons) | {min(seasons) - 1}) if mode == "prev" else sorted(seasons)
    raw = _success_by_season(need)

    if mode == "same":
        t = raw.copy()
        t["등급"] = _grade(t["성공"])
        return t.rename(columns={"선수명": "pitcher", "team": "pitcher_team"})[
            ["season", "pitcher_team", "pitcher", "등급"]]

    # 직전 시즌은 소속팀이 다를 수 있으므로 이름으로만 합산한다.
    prev = raw.groupby(["season", "선수명"], as_index=False)["성공"].sum()
    prev["season"] = prev["season"] + 1          # 다음 시즌에 적용할 등급
    prev["등급"] = _grade(prev["성공"])
    prev = prev.rename(columns={"선수명": "pitcher", "성공": "직전성공"})

    # 그 시즌 실제로 등판한 (팀, 선수)에 붙인다.
    cur = raw[raw["season"].isin(seasons)][["season", "team", "선수명"]].drop_duplicates()
    cur = cur.rename(columns={"team": "pitcher_team", "선수명": "pitcher"})
    out = cur.merge(prev[["season", "pitcher", "등급", "직전성공"]],
                    on=["season", "pitcher"], how="left")
    return out


def coverage(seasons):
    """prev 정의로 등급을 못 매기는 비율 — 첫 시즌과 신인이 빠진다."""
    t = build(seasons, mode="prev")
    miss = t["등급"].isna()
    by = t.groupby("season")["등급"].apply(lambda s: s.isna().mean() * 100)
    return dict(전체결측=miss.mean() * 100, 시즌별=by)


if __name__ == "__main__":
    seasons = list(range(2018, 2027))
    same = build(seasons, "same")
    prev = build(seasons, "prev")
    cov = coverage(seasons)

    L = ["=== 불펜 등급 정의 비교 (누수 있음 vs 없음) ===",
         f"대상 시즌 {seasons[0]}~{seasons[-1]} (prev 정의는 직전 시즌이 필요해 2017은 제외)\n"]
    L.append("[1] 등급 분포 (투수×시즌 수)")
    for name, t in (("같은 시즌 (원래)", same), ("직전 시즌 (보정)", prev)):
        vc = t["등급"].value_counts(dropna=False)
        L.append(f"  {name}: " + " / ".join(
            f"{'등급없음' if pd.isna(k) else k} {v:,}" for k, v in vc.items()))
    L.append(f"\n[2] 직전 시즌 정의로 등급을 못 매기는 비율: {cov['전체결측']:.1f}%")
    for s, v in cov["시즌별"].items():
        L.append(f"    {s}: {v:.1f}%")

    # 두 정의가 얼마나 일치하나
    m = same.merge(prev, on=["season", "pitcher_team", "pitcher"],
                   suffixes=("_same", "_prev")).dropna(subset=["등급_prev"])
    agree = (m["등급_same"] == m["등급_prev"]).mean() * 100
    L.append(f"\n[3] 두 정의의 일치율: {agree:.1f}% (n={len(m):,})")
    ct = pd.crosstab(m["등급_prev"], m["등급_same"])
    L.append("    행=직전 시즌 기준, 열=같은 시즌 기준")
    L.append("    " + "".join(f"{c:>10s}" for c in ct.columns))
    for idx, r in ct.iterrows():
        L.append(f"    {idx:6s}" + "".join(f"{v:>10,}" for v in r))

    (DATA.parent / "tiers_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("summary -> tiers_summary.txt")
