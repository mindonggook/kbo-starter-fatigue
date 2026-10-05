"""리포트 HTML의 그림 자리표시자를 PNG의 data URI로 바꿔 완성본을 만든다.

Artifact로 게시하면 로컬 파일 경로는 못 읽으므로 그림을 전부 인라인해야 한다.
"""
import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

IMAGES = {
    "{{IMG_TYPES}}": "2024_early_hook_types.png",
    "{{IMG_SCATTER}}": "2024_early_hook_scatter.png",
    "{{IMG_CARRY_NAIVE}}": "2024_prev_pitch_carryover.png",
    "{{IMG_CARRY_WITHIN}}": "2024_within_prev_pitch.png",
    "{{IMG_REST}}": "2024_within_rest_days.png",
    "{{IMG_SEASON}}": "2024_within_season.png",
    "{{IMG_BF}}": "2024_bf_threshold.png",
    "{{IMG_TTO_STD}}": "2024_tto_standardized.png",
    "{{IMG_PITCH_SURV}}": "2024_pitch_bucket_survivor.png",
    "{{IMG_INSTABILITY}}": "2024_collinear_instability.png",
    "{{IMG_HOOK_CTX}}": "2024_hook_context.png",
    "{{IMG_HOOK_RUNNER}}": "2024_hook_rate_runners.png",
    "{{IMG_FATIGUE_PA}}": "2024_fatigue_pa_effects.png",
    "{{IMG_BULLPEN}}": "2024_bullpen_effect.png",
    "{{IMG_DROPOUT}}": "dropout_bias.png",
    "{{IMG_FOREST}}": "pooled_forest_multi.png",
    "{{IMG_ORIGIN}}": "runner_origin_hook.png",
    "{{IMG_PITCHLEVEL}}": "pitch_level_two_axes.png",
    "{{IMG_FAMILIAR}}": "familiarity_between_games.png",
    "{{IMG_HOOKVALUE}}": "hook_value.png",
    "{{IMG_TIER}}": "bullpen_tier.png",
    "{{IMG_VELO}}": "velocity_decline.png",
    "{{IMG_MED}}": "mediation_velocity.png",
    "{{IMG_RESTVELO}}": "rest_velocity.png",
    "{{IMG_HOOKVELO}}": "hook_velocity.png",
    "{{IMG_RELIEFVELO}}": "relief_velocity.png",
    "{{IMG_RELCMD}}": "release_command.png",
    "{{IMG_MED2}}": "mediation_release.png",
    "{{IMG_PITCHTYPE}}": "pitchtype_recheck.png",
    "{{IMG_MEDPT}}": "mediation_pitchtype.png",
    "{{IMG_REPERTOIRE}}": "repertoire_tto.png",
    "{{IMG_EXPOSURE}}": "exposure_learning.png",
    "{{IMG_HANDOFF}}": "handoff_novelty.png",
    "{{IMG_CONVERSION}}": "conversion.png",
    "{{IMG_RUNVALUE}}": "runvalue.png",
    "{{IMG_BPCOST}}": "bullpen_cost.png",
    "{{IMG_SWEEP}}": "runvalue_sweep.png",
    "{{IMG_TIERFIX}}": "tier_fix.png",
    "{{IMG_RELFIX}}": "release_fix.png",
    "{{IMG_SURV}}": "survivorship.png",
    "{{IMG_MODEL}}": "model_hook.png",
    "{{IMG_HOOKTYPE}}": "hook_types_fix.png",
    "{{IMG_TIERCI}}": "tier_ci.png",
    "{{IMG_TIERWPA}}": "tier_wpa.png",
    "{{IMG_SHAP}}": "model_shap.png",
}


def build(template: Path, out: Path) -> None:
    html = template.read_text(encoding="utf-8")
    for token, filename in IMAGES.items():
        raw = (FIG / filename).read_bytes()
        uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        html = html.replace(token, uri)

    # IMAGES에 없는 자리표시자까지 잡아야 한다 — 등록을 빠뜨리면 그림이 조용히 깨진다.
    leftover = sorted(set(re.findall(r"\{\{IMG_[A-Z0-9_]+\}\}", html)))
    if leftover:
        raise SystemExit(f"치환되지 않은 자리표시자: {leftover}")

    out.write_text(html, encoding="utf-8")
    print(f"{out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
