"""대표값을 한 곳에 모은다 — 리포트가 숫자를 타이핑하지 않게.

외부 평가가 세 차례 연속으로 같은 종류의 문제를 지적했다. 첫 화면과 결론부가
부록의 수정을 따라가지 못해, 같은 지표가 섹션마다 다른 값으로 나왔다.
원인은 단순하다 -- 숫자를 본문에 직접 써 넣었기 때문이다.
상수를 여러 파일에 하드코딩해 둔 것과 같다.

그래서 흐름을 바꾼다.

    분석 스크립트 → *_summary.txt → (여기) key_numbers.json → build_report.py → 리포트

리포트 본문에는 {{N_tto_avg}} 같은 자리표시자만 둔다. 분석을 다시 돌리면
숫자가 저절로 따라온다. 손으로 고칠 곳이 없으니 어긋날 수도 없다.

요약 파일의 형식이 바뀌면 조용히 틀린 값이 들어가는 것이 가장 위험하므로,
찾지 못한 항목은 예외를 던지고 멈춘다.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "key_numbers.json"   # 리포트 빌드가 읽으므로 루트에 둔다


def read(name):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"요약 파일이 없다: {name} — 해당 분석을 먼저 돌려야 한다")
    return p.read_text(encoding="utf-8")


def grab(text, pattern, where):
    """하나만 뽑는다. 못 찾으면 멈춘다 — 조용히 틀린 값이 들어가는 것보다 낫다."""
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"[{where}] 형식이 바뀌었다: {pattern}")
    return m.groups()


def build():
    n = {}

    # ── 타순 회전 (발견 06 · 34) ────────────────────────────────
    sv = read("eda_survivorship_summary.txt")
    blocks = sv.split("[피안타율]")[1].split("[득점가치]")
    for key, blk in (("avg", blocks[0]), ("rv", blocks[1])):
        a, lo, hi = grab(blk, r"C 투수×시즌 고정효과\s*:\s*([+-][\d.]+)/1000 "
                              r"\[([+-][\d.]+), ([+-][\d.]+)\]", "생존편향 C")
        (A,) = grab(blk, r"A 생존 등판만 \(지금까지\)\s*:\s*([+-][\d.]+)/1000", "A")
        (B,) = grab(blk, r"B 제한 없음\s*:\s*([+-][\d.]+)/1000", "B")
        n[f"tto_{key}"] = float(a)
        n[f"tto_{key}_lo"] = min(float(A), float(B))
        n[f"tto_{key}_hi"] = max(float(A), float(B))

    # ── 교체 이득 · 실점 (발견 38) ──────────────────────────────
    tf = read("eda_tier_fix2_summary.txt")
    pit = tf.split("[시점 기준 누적")[1].split("[접전")[0]
    for tier in ("추격조", "중간", "필승조"):
        v, lo, hi = grab(pit, rf"구원 {tier}\s+이득 ([+-][\d.]+)점/9타자 "
                              rf"\[([+-][\d.]+), ([+-][\d.]+)\]", f"발견38 {tier}")
        n[f"relief_rv_{tier}"] = float(v)
        n[f"relief_rv_{tier}_lo"] = float(lo)
        n[f"relief_rv_{tier}_hi"] = float(hi)

    # ── 교체 이득 · 승률 환산 (발견 39) ─────────────────────────
    tw = read("eda_tier_wpa2_summary.txt")
    close = tw.split("[접전 (2점 이내)]")[1]
    for tier in ("추격조", "중간", "필승조"):
        v, lo, hi = grab(close, rf"{tier}\s+[+-][\d.]+\*?\s*\[[^\]]+\]\s*"
                                rf"([+-][\d.]+)\*?\s*\[([+-][\d.]+),([+-][\d.]+)\]",
                         f"발견39 {tier}")
        n[f"relief_wp_{tier}"] = float(v)
        n[f"relief_wp_{tier}_lo"] = float(lo)
        n[f"relief_wp_{tier}_hi"] = float(hi)

    # ── 1점당 승률 (발견 39) ────────────────────────────────────
    we = read("winexp_summary.txt")
    mid = we.split("[4] 5~8회만")[1]
    for lab, key in (("동점", "tie"), ("5점 이상", "blowout")):
        (v,) = grab(mid, rf"{lab}: 1점당 승률\s*([\d.]+)%p", f"환율 {lab}")
        n[f"rate_{key}"] = float(v)

    # ── 구속 (발견 24) ──────────────────────────────────────────
    pt = read("eda_pitchtype_summary.txt")
    (v,) = grab(pt, r"직구만\s*:\s*([+-][\d.]+)km/h", "발견24 직구")
    n["velo_fastball"] = float(v)

    # ── 매개분 (발견 23 · 33 · 35) ──────────────────────────────
    md = read("eda_mediation3_summary.txt")
    fixed = md.split("흔들림 (구종 고정)")[1]
    v, lo, hi = grab(fixed, r"설명하는 몫: ([+-][\d.]+)% \[부트스트랩 "
                            r"([+-][\d.]+), ([+-][\d.]+)\]", "매개분")
    n["mediation_share"] = float(v)
    n["mediation_lo"] = float(lo)
    n["mediation_hi"] = float(hi)

    # ── 전환 효율 (발견 28) ─────────────────────────────────────
    cv = read("eda_conversion_summary.txt")
    (v,) = grab(cv, r"분해: 맞히기 채널 [+-][\d.]+/1000 \(([\d.]+)%\)", "전환효율")
    n["contact_channel"] = float(v)

    # ── 모델 (발견 36) ──────────────────────────────────────────
    mh = read("model_hook_summary.txt")
    # 사다리([0])에도 같은 이름의 숫자가 있으므로 시험 성능 절로 범위를 좁힌다.
    perf = mh.split("[1] 시험 성능")[1].split("[2]")[0]
    n["model_auc"] = float(grab(perf, r"AUC\s+([\d.]+)", "모델 AUC")[0])
    n["model_prauc"] = float(grab(perf, r"PR-AUC\s+([\d.]+)", "모델 PR-AUC")[0])
    n["model_auc_base"] = float(grab(mh, r"투구수만\s+AUC ([\d.]+)", "기준 AUC")[0])

    # ── 파생값 (9타자 환산) ─────────────────────────────────────
    n["tto_runs_9"] = round(n["tto_rv"] / 1000 * 9, 3)
    n["tto_runs_9_lo"] = round(n["tto_rv_lo"] / 1000 * 9, 3)
    n["tto_runs_9_hi"] = round(n["tto_rv_hi"] / 1000 * 9, 3)

    return n


def _avg(v):
    """피안타율은 야구 관례대로 앞의 0을 떼고 +.018 꼴로 쓴다."""
    return f"{v / 1000:+.3f}".replace("0.", ".")


def _avg_bare(v):
    return f"{v / 1000:.4f}".lstrip("0")


FORMATS = {
    "tto_avg": _avg,
    "tto_avg_lo": _avg_bare,
    "tto_avg_hi": _avg_bare,
}


def fmt(key, v):
    if key in FORMATS:
        return FORMATS[key](v)
    if key.startswith(("tto_runs", "relief_rv")):
        return f"{v:+.3f}"
    if key.startswith("relief_wp") or key.startswith("rate_"):
        return f"{v:+.2f}" if key.startswith("relief_wp") else f"{v:.2f}"
    if key.startswith("model_"):
        return f"{v:.4f}"
    if key in ("mediation_share", "mediation_lo", "mediation_hi", "contact_channel"):
        return f"{v:.1f}"
    if key == "velo_fastball":
        return f"{v:+.2f}"
    return f"{v:+.1f}" if v > 0 else f"{v:.1f}"


if __name__ == "__main__":
    nums = build()
    payload = {k: {"value": v, "text": fmt(k, v)} for k, v in sorted(nums.items())}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    lines = [f"대표값 {len(payload)}개 -> {OUT.name}"]
    for k, d in payload.items():
        lines.append(f"  {{{{N_{k}}}}}".ljust(32) + f"{d['text']:>10s}")
    (RESULTS / "key_numbers_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"key_numbers.json ({len(payload)}개)")
