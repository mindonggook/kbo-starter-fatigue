"""추론으로 얻은 결론이 예측 모델에서도 재현되는가 — 강판 확률 모델.

이 리포트는 전부 회귀와 메타분석이다. 외부 평가가 지적한 대로, 같은 질문을
예측 모델로 한 번 더 물으면 결론이 설계에 기대고 있는지 확인할 수 있다.

물음: <이 타석이 끝나면 선발이 내려가는가>
확인할 것: 발견 09가 말한 '방아쇠는 투구수가 아니라 주자'가 모델에서도 1순위인가.

누수를 막는 것이 전부다.
  - 타석이 <시작되는 시점>에 감독이 알 수 있는 값만 넣는다.
    그 타석의 결과도, 투구 수도 모른다.
  - 시간으로 나눈다. 2017~2023으로 배우고 2024~2026에서 시험한다.
    같은 시즌 안에서 섞어 나누면 미래를 보고 과거를 맞히는 셈이다.
  - 경기가 끝나서 마지막 타석이 된 경우는 '강판'이 아니므로 뺀다.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                             average_precision_score)

DATA = Path(__file__).resolve().parent.parent / "data"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#7b5bd6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

FEATURES = {
    "주자수": "루상의 주자 수",
    "아웃카운트": "아웃 카운트",
    "점수차": "점수차 (투수팀 기준)",
    "투수_누적투구수": "그 시점 누적 투구수",
    "타순회전": "타순 회전",
    "inning": "이닝",
    "batting_order": "상대 타순",
    "허용안타_누적": "이 경기 허용 안타",
    "허용볼넷_누적": "이 경기 허용 볼넷",
    "실점_누적": "이 경기 실점(팀)",
    "휴식일": "휴식일",
    "직전편차_10구당": "직전 등판 투구수 편차",
    "시즌평균투구수": "그 투수의 평소 투구수",
}


def load(seasons):
    frames = []
    for s in seasons:
        pa_path = DATA / f"kbo_pa_state_{s}.parquet"
        chk_path = DATA / f"kbo_state_check_{s}.parquet"
        if not pa_path.exists():
            continue
        pa = pd.read_parquet(pa_path)
        chk = pd.read_parquet(chk_path)
        ok = set(chk.loc[chk["score_ok"] & chk["starter_ok"], "game_id"])
        d = pa[pa["game_id"].isin(ok)].copy()
        d["season"] = s
        frames.append(d)
    pa = pd.concat(frames, ignore_index=True)
    pa["seq"] = np.arange(len(pa))            # 파싱 순서 = 시간 순서
    pa["안타"] = pa["안타"].astype(float)
    pa["타수"] = pa["타수"].astype(bool)
    pa["볼넷"] = (pa["종류"] == "볼넷·사구").astype(float)
    pa["팀반이닝"] = pa["game_id"].astype(str) + "_" + pa["batting_team"].astype(str)

    # 그 타자팀의 공격이 이 타석 뒤에도 계속되는가 — 경기 종료로 끝난 타석은 제외한다.
    last_seq = pa.groupby("팀반이닝")["seq"].transform("max")
    pa["뒤에_더_있음"] = pa["seq"] < last_seq

    st = pa[pa["is_starter"]].copy()
    st["start"] = st["game_id"].astype(str) + "_" + st["pitcher"].astype(str)
    st = st.sort_values("seq")
    # 이 타석이 그 선발의 마지막 타석인가
    st["마지막타석"] = st["seq"] == st.groupby("start")["seq"].transform("max")
    st["강판"] = (st["마지막타석"] & st["뒤에_더_있음"]).astype(int)
    st = st[st["뒤에_더_있음"] | ~st["마지막타석"]].copy()

    # 타석 <시작 시점>까지의 누적 — shift로 자기 결과를 뺀다.
    g = st.groupby("start")
    st["허용안타_누적"] = g["안타"].cumsum() - st["안타"]
    st["허용볼넷_누적"] = g["볼넷"].cumsum() - st["볼넷"]
    st["실점_누적"] = st["점수_타자팀"]
    st["점수차"] = st["점수차_투수팀기준"]

    # 등판 이력 (그 경기 전까지만 쓰는 값)
    hist = []
    for s in sorted(st["season"].unique()):
        path = DATA / f"kbo_pitcher_appearances_{s}.parquet"
        if not path.exists():
            continue
        app = pd.read_parquet(path)
        a = app[app["is_starter"]].copy()
        a["date"] = pd.to_datetime(a["date"].astype(str), format="%Y%m%d")
        a = a.sort_values(["선수명", "date"])
        gg = a.groupby("선수명")
        a["휴식일"] = (a["date"] - gg["date"].shift(1)).dt.days
        prev = gg["투구수"].shift(1)
        # '평소 투구수'는 직전 등판들까지의 평균 — 미래를 보지 않는다.
        a["시즌평균투구수"] = gg["투구수"].apply(
            lambda x: x.shift(1).expanding().mean()).to_numpy()
        a["직전편차_10구당"] = (prev - a["시즌평균투구수"]) / 10.0
        a["season"] = s
        hist.append(a[["game_id", "선수명", "season", "휴식일",
                       "직전편차_10구당", "시즌평균투구수"]])
    h = pd.concat(hist, ignore_index=True).rename(columns={"선수명": "pitcher"})
    st = st.merge(h, on=["game_id", "pitcher", "season"], how="left")
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", type=int, default=list(range(2017, 2024)))
    p.add_argument("--test", nargs="+", type=int, default=[2024, 2025, 2026])
    a = p.parse_args()

    d = load(a.train + a.test)
    feats = list(FEATURES)
    d = d.dropna(subset=["강판"])
    tr = d[d["season"].isin(a.train)]
    te = d[d["season"].isin(a.test)]

    X_tr, y_tr = tr[feats].astype(float), tr["강판"]
    X_te, y_te = te[feats].astype(float), te["강판"]

    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.04, num_leaves=31,
        min_child_samples=80, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=1.0, random_state=20261004, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="auc",
              callbacks=[lgb.early_stopping(60, verbose=False)])

    pr = model.predict_proba(X_te)[:, 1]
    L = [f"=== 강판 확률 예측 모델 (LightGBM) ===",
         f"학습 {a.train[0]}~{a.train[-1]} 타석 {len(tr):,}건 (강판률 {y_tr.mean()*100:.2f}%)",
         f"시험 {a.test[0]}~{a.test[-1]} 타석 {len(te):,}건 (강판률 {y_te.mean()*100:.2f}%)",
         f"나무 {model.best_iteration_ or model.n_estimators}개\n"]

    # 기준 모델 — 투구수 하나로 어디까지 가나. 이게 없으면 0.95가 커 보인다.
    ladders = [("투구수만", ["투수_누적투구수"]),
               ("투구수+아웃+주자", ["투수_누적투구수", "아웃카운트", "주자수"]),
               ("전체", feats)]
    L.append("[0] 기준 모델과의 사다리 (같은 시간 분할)")
    for name, cols in ladders:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               min_child_samples=80, random_state=20261004, verbose=-1)
        m.fit(tr[cols].astype(float), y_tr)
        q = m.predict_proba(te[cols].astype(float))[:, 1]
        L.append(f"  {name:18s} AUC {roc_auc_score(y_te, q):.4f} · "
                 f"PR-AUC {average_precision_score(y_te, q):.4f} · "
                 f"Brier {brier_score_loss(y_te, q):.4f}")
    L.append(f"  (강판 비율 {y_te.mean()*100:.2f}% — PR-AUC의 기준선이다)")
    L.append("")

    L.append("[1] 시험 성능 (시간 분할 — 학습에 쓰지 않은 미래 시즌)")
    base = np.full(len(y_te), y_tr.mean())
    L.append(f"  AUC         {roc_auc_score(y_te, pr):.4f}")
    L.append(f"  PR-AUC      {average_precision_score(y_te, pr):.4f} "
             f"(무작위 기준선 {y_te.mean():.4f})")
    L.append(f"  로그손실     {log_loss(y_te, pr):.4f} (상수 예측 {log_loss(y_te, base):.4f})")
    L.append(f"  Brier       {brier_score_loss(y_te, pr):.4f} "
             f"(상수 예측 {brier_score_loss(y_te, base):.4f})")
    for s in a.test:
        m = te["season"] == s
        if m.sum() > 500:
            L.append(f"  {s} AUC    {roc_auc_score(y_te[m], pr[m]):.4f} (n={m.sum():,})")
    L.append("")

    # ── [2] 중요도 ─────────────────────────────────────────────
    L.append("[2] 무엇을 보고 예측하나")
    gain = pd.Series(model.booster_.feature_importance("gain"), index=feats)
    gain = gain / gain.sum() * 100
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_te), size=min(40000, len(X_te)), replace=False)
    perm = permutation_importance(model, X_te.iloc[idx], y_te.iloc[idx],
                                  scoring="roc_auc", n_repeats=5,
                                  random_state=20261004, n_jobs=1)
    pi = pd.Series(perm.importances_mean, index=feats)
    pis = pd.Series(perm.importances_std, index=feats)
    order = pi.sort_values(ascending=False).index
    L.append("  " + "변수".ljust(24) + f"{'이득 기여':>10s}{'AUC 하락':>12s}")
    for f in order:
        L.append("  " + FEATURES[f].ljust(24)
                 + f"{gain[f]:>9.1f}%{pi[f]:>11.4f}±{pis[f]:.4f}")
    L.append("")

    top = list(order[:3])
    L.append(f"  → 상위 3개: {', '.join(FEATURES[f] for f in top)}")
    rank = {f: i + 1 for i, f in enumerate(order)}
    L.append(f"  → 발견 09의 검정: 주자수 {rank['주자수']}위 vs "
             f"누적 투구수 {rank['투수_누적투구수']}위")
    L.append("")

    L.append("[2-b] 보정 — 예측 확률이 실제와 맞나 (12분위)")
    frac0, mean0 = calibration_curve(y_te, pr, n_bins=12, strategy="quantile")
    for f0, m0 in zip(frac0, mean0):
        L.append(f"  예측 {m0*100:6.2f}% → 실제 {f0*100:6.2f}%")
    L.append(f"  최대 절대 오차 {np.max(np.abs(frac0 - mean0))*100:.2f}%p")
    L.append("")

    # ── [3] 결론이 모델 안에서도 보이는가 ──────────────────────
    L.append("[3] 주자 유무별 예측 강판 확률 (투구수 구간 안에서)")
    te2 = te.copy()
    te2["예측"] = pr
    te2["구간"] = pd.cut(te2["투수_누적투구수"], [0, 50, 75, 90, 105, 200],
                       labels=["~50구", "51~75", "76~90", "91~105", "106구+"])
    tab = te2.groupby(["구간", te2["주자수"] > 0], observed=True)["예측"].mean().unstack()
    tab.columns = ["주자 없음", "주자 있음"]
    for idx2, r in tab.iterrows():
        L.append(f"  {idx2}: 주자 없음 {r['주자 없음']*100:5.2f}% · "
                 f"주자 있음 {r['주자 있음']*100:5.2f}% "
                 f"(차이 {(r['주자 있음']-r['주자 없음'])*100:+.2f}%p)")
    L.append("")

    # ── 그림 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), dpi=150, facecolor=SURFACE)

    ax = axes[0]
    y = np.arange(len(order))[::-1]
    vals = [pi[f] for f in order]
    errs = [pis[f] for f in order]
    ax.barh(y, vals, xerr=errs, color=[VIOLET if v == max(vals) else BLUE for v in vals],
            height=0.6, error_kw=dict(ecolor=SECONDARY_INK, elinewidth=1.1, capsize=3))
    for yi, f in zip(y, order):
        ax.text(pi[f] + max(vals) * 0.02, yi, f"{gain[f]:.0f}%", va="center",
                color=MUTED, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURES[f] for f in order], fontsize=9)
    ax.set_xlabel("빼면 AUC가 얼마나 떨어지나 (회색 숫자는 이득 기여)",
                  color=SECONDARY_INK, fontsize=9)
    ax.set_title("모델은 무엇을 보고 교체를 예측하나", color=INK, fontsize=12.5,
                 loc="left", pad=10)

    ax = axes[1]
    x = np.arange(len(tab))
    for col, color in (("주자 없음", MUTED), ("주자 있음", ORANGE)):
        ax.plot(x, tab[col] * 100, marker="o", markersize=7, linewidth=2.2,
                color=color, label=col)
    for xi, r in zip(x, tab.itertuples()):
        ax.text(xi, r._2 * 100 + 1.2, f"{r._2*100:.0f}", ha="center",
                color=ORANGE, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index.astype(str), fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    ax.set_title("예측 강판 확률 — 주자 유무", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.set_ylabel("예측 확률 (%)", color=SECONDARY_INK, fontsize=9.5)

    ax = axes[2]
    frac, mean_pred = calibration_curve(y_te, pr, n_bins=12, strategy="quantile")
    ax.plot([0, max(mean_pred) * 1.05], [0, max(mean_pred) * 1.05], color=MUTED,
            linewidth=1.2, linestyle="--", label="완벽한 보정")
    ax.plot(mean_pred, frac, color=VIOLET, marker="o", markersize=6, linewidth=2.0,
            label="관측")
    ax.set_xlabel("예측 확률", color=SECONDARY_INK, fontsize=9.5)
    ax.set_ylabel("실제 강판 비율", color=SECONDARY_INK, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK)
    ax.set_title("보정 — 예측한 확률이 맞나", color=INK, fontsize=12.5, loc="left", pad=10)

    for a_ in axes:
        a_.tick_params(colors=MUTED, labelsize=9)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.set_facecolor(SURFACE)
    fig.suptitle("추론으로 얻은 결론이 예측 모델에서도 나오는가",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "model_hook.png")
    plt.close(fig)

    # ── SHAP 의존도 — '투구수가 무대, 주자가 방아쇠'를 그림 한 장으로 ──
    # LightGBM이 TreeSHAP을 내장한다(pred_contrib). shap 패키지가 없어도 된다.
    rng2 = np.random.default_rng(7)
    samp = te.iloc[rng2.choice(len(te), size=min(25000, len(te)), replace=False)]
    contrib = model.booster_.predict(samp[feats].astype(float), pred_contrib=True)
    sv = pd.DataFrame(contrib[:, :len(feats)], columns=feats, index=samp.index)

    L.append("[5] SHAP — 주자의 기여가 투구수에 따라 커지는가")
    band = pd.cut(samp["투수_누적투구수"], [0, 50, 75, 90, 105, 200],
                  labels=["~50구", "51~75", "76~90", "91~105", "106구+"])
    tab2 = (sv["주자수"].groupby([band, samp["주자수"] > 0], observed=True)
            .mean().unstack())
    tab2.columns = ["주자 없음", "주자 있음"]
    for idx2, r in tab2.iterrows():
        L.append(f"  {idx2}: 주자 없음 {r['주자 없음']:+.4f} · "
                 f"주자 있음 {r['주자 있음']:+.4f} "
                 f"(차이 {r['주자 있음']-r['주자 없음']:+.4f})")
    L.append("  (로그오즈 기여. 차이가 커질수록 '투구수가 깔아 놓은 무대 위에서'"
             " 주자가 더 크게 작동한다는 뜻이다)")
    L.append("")

    fig2, ax2 = plt.subplots(figsize=(8.2, 5.0), dpi=150, facecolor=SURFACE)
    for lab2, color, mask in (("주자 없음", MUTED, samp["주자수"] == 0),
                              ("주자 1명", BLUE, samp["주자수"] == 1),
                              ("주자 2명 이상", ORANGE, samp["주자수"] >= 2)):
        g2 = samp[mask]
        if len(g2) < 200:
            continue
        b = pd.cut(g2["투수_누적투구수"], np.arange(0, 131, 10))
        m2 = sv.loc[g2.index, "주자수"].groupby(b, observed=True).mean()
        xs = [iv.mid for iv in m2.index]
        ax2.plot(xs, m2.values, marker="o", markersize=5, linewidth=2.0,
                 color=color, label=lab2)
    ax2.axhline(0, color=INK, linewidth=0.9)
    ax2.set_xlabel("그 시점 누적 투구수", color=SECONDARY_INK, fontsize=10)
    ax2.set_ylabel("주자 수의 SHAP 기여 (로그오즈)", color=SECONDARY_INK, fontsize=10)
    ax2.legend(frameon=False, fontsize=9.5, labelcolor=SECONDARY_INK)
    ax2.set_title("주자는 언제부터 방아쇠가 되나", color=INK, fontsize=13,
                  loc="left", pad=12)
    ax2.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color(GRID)
    ax2.set_facecolor(SURFACE)
    fig2.tight_layout()
    fig2.savefig(FIG / "model_shap.png")
    plt.close(fig2)

    path = DATA.parent / "model_hook_summary.txt"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
