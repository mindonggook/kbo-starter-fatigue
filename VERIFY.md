# 사전 등록 검증 방법

2026 시즌 종료(10월 9일) 전에 예측값을 고정했다는 것을 확인하는 방법.

## 1. 태그가 언제 올라갔는지

저장소의 **Tags** 탭에서 `prereg-2026`을 연다.
GitHub 서버가 기록한 시각이고, 저장소 주인이 고칠 수 없다.

## 2. 내용이 그대로인지

```bash
git clone https://github.com/mindonggook/kbo-starter-fatigue.git
cd kbo-starter-fatigue
git checkout prereg-2026
git cat-file blob prereg-2026:prereg.txt | sha256sum
```

아래 값과 같아야 한다.

| 파일 | SHA-256 (저장소에 저장된 원본) |
|---|---|
| `prereg.txt` | `51eb12d314da51e947a9cb6e2f728a50fd0c5ec0138725aaab9b1fda25ea4e17` |
| `prereg_designC.txt` | `58ce46346a4518104a1d01bd2d0b7545eb74849a925d4bd2de23c02ec94fb58f` |
| `prereg_manifest.json` | `b258906510579aff73eb218fc58d033dcba09c63ad4731bdffd67186fbcef3d7` |
| `src/prereg.py` | `aa7cd4a1420ecb2c24800c08355687ad4be7f97a02330d473596dd5f0ed4668b` |
| `src/eda_runvalue_sweep.py` | `8f3434f31bb707462bfac44ce0e687135bab2526471afbbc703a24c3c278003b` |
| `src/eda_survivorship.py` | `475b0441ff08d442f0ebbfd1bd5ef20dfa2ac1309487a7fa9465d9ef300167ee` |

> **줄바꿈 주의.** 윈도우에서 작업했으므로 디스크의 파일은 CRLF,
> 저장소에 저장된 원본은 LF다(`core.autocrlf=true`). 위 해시는 **저장소 원본 기준**이며
> `git cat-file blob`으로 꺼내야 같은 값이 나온다. `git show`는 줄바꿈을 되돌려 주므로 다르다.

## 3. 무엇을 예측했는지

`prereg.txt` — 2017~2025 아홉 시즌만으로 계산한 2026 예측구간 9개.
`prereg_designC.txt` — 대표 설계(투수×시즌 고정효과)의 예측구간.

**합격 기준**: 9개 중 8개 이상이 구간 안이면 재현, 7개 이하면 재현 실패.
방향 예측 5개는 따로 세어 4개 이상이면 통과.

## 4. 검정 방법

```bash
python src/eda_runvalue_sweep.py --seasons 2026
```

**스크립트를 고치지 않는다.** 고치면 사전 등록이 아니다.
