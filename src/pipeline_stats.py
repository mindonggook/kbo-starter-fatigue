"""파이프라인 현황을 집계한다 — 리포트의 데이터 엔지니어링 섹션에 쓸 실제 수치."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def mb(files):
    return sum(f.stat().st_size for f in files) / 1024 ** 2


def main():
    L = ["=== 파일 종류별 ==="]
    groups = {}
    for f in DATA.rglob("*"):
        if f.is_file():
            groups.setdefault(f.suffix or "(없음)", []).append(f)
    allf = [f for v in groups.values() for f in v]
    for k, v in sorted(groups.items(), key=lambda x: -mb(x[1])):
        L.append(f"  {k:10s} {len(v):>6,}개  {mb(v):>9.1f} MB")
    L.append(f"  {'합계':10s} {len(allf):>6,}개  {mb(allf):>9.1f} MB")

    L.append("\n=== 주요 데이터셋 ===")
    for pat in ("kbo_pitcher_appearances_*.parquet", "kbo_pa_state_*.parquet",
                "kbo_pitches_*.parquet", "naver_full_*.parquet"):
        files = sorted(DATA.glob(pat))
        if not files:
            continue
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        L.append(f"  {pat:34s} {len(files):>2}개 · {rows:>11,} 행 · {mb(files):>7.1f} MB")
    for name in ("kbo_pa_runvalue.parquet", "linear_weights.parquet",
                 "run_expectancy.parquet"):
        p = DATA / name
        if p.exists():
            L.append(f"  {name:34s} {'':>2}   · "
                     f"{pq.ParquetFile(p).metadata.num_rows:>11,} 행 · {mb([p]):>7.1f} MB")

    L.append("\n=== 캐시 (원본 보존) ===")
    tot_files = tot_mb = 0
    for d in sorted(DATA.glob("*cache*")):
        if d.is_dir():
            files = list(d.glob("*"))
            tot_files += len(files)
            tot_mb += mb(files)
            L.append(f"  {d.name:30s} {len(files):>6,}개 · {mb(files):>8.1f} MB")
    L.append(f"  {'합계':30s} {tot_files:>6,}개 · {tot_mb:>8.1f} MB")

    L.append("\n=== 코드 ===")
    src = sorted((ROOT / "src").glob("*.py"))
    lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in src)
    L.append(f"  파이썬 파일 {len(src)}개 · {lines:,}줄")

    L.append("\n=== 시즌 범위 ===")
    for pat, lab in (("kbo_pa_state_*.parquet", "KBO 공식"),
                     ("naver_full_*.parquet", "네이버 추적")):
        ys = sorted(int(f.stem.split("_")[-1]) for f in DATA.glob(pat))
        if ys:
            L.append(f"  {lab}: {ys[0]}~{ys[-1]} ({len(ys)}시즌)")

    (ROOT / "pipeline_stats.txt").write_text("\n".join(L), encoding="utf-8")
    print("summary -> pipeline_stats.txt")


if __name__ == "__main__":
    main()
