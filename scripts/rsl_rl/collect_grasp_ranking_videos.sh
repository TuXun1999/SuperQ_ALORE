#!/usr/bin/env bash
# Collect one comparable set of grasp-ranking videos for four target-pose sets.
#
# Usage:
#   ./scripts/rsl_rl/collect_grasp_ranking_videos.sh chair1 --headless
#
# Run this from an Isaac Lab-enabled shell. Additional arguments are forwarded
# to grasp_ranking_eval.py (for example: --device cuda:0).

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: collect_grasp_ranking_videos.sh <object-category> [extra evaluator arguments]

Runs grasp_ranking_eval.py four times with one environment per run. Each run
uses a distinct seed, yielding one reproducible target-pose set that is shared
by all evaluation modes. Bucket categories use two grasp poses; all other
categories use three. Videos are written to ./logs/videos.

Environment overrides:
  PYTHON_BIN       Python executable to use (default: python)
  BASE_SEED        Seed for target-pose set 1 (default: 3401)
  VIDEO_LENGTH     Frames per exported video (default: 750)
  VIDEO_FPS        Video frame rate (default: 30)
  ROLLOUT_STEPS    Evaluation horizon per mode (default: 750)
  NUM_TARGET_SETS  Number of target-pose sets (default: 4)
  VIDEO_DIR        Output directory (default: ./logs/videos)

Example:
  VIDEO_LENGTH=300 ./scripts/rsl_rl/collect_grasp_ranking_videos.sh chair1 --headless
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

object_category="$1"
shift
extra_args=("$@")

case "${object_category,,}" in
    bucket*)
        num_grasp_poses=2
        ;;
    *)
        num_grasp_poses=3
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
video_dir="${VIDEO_DIR:-${project_root}/logs/videos}"
if [[ "${video_dir}" != /* ]]; then
    video_dir="${project_root}/${video_dir}"
fi
python_bin="${PYTHON_BIN:-python}"
base_seed="${BASE_SEED:-3401}"
video_length="${VIDEO_LENGTH:-750}"
video_fps="${VIDEO_FPS:-30}"
rollout_steps="${ROLLOUT_STEPS:-750}"
num_target_sets="${NUM_TARGET_SETS:-4}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Error: Python executable '${python_bin}' was not found. Activate the Isaac Lab environment or set PYTHON_BIN." >&2
    exit 1
fi

if ! [[ "${base_seed}" =~ ^[0-9]+$ && "${video_length}" =~ ^[1-9][0-9]*$ && "${video_fps}" =~ ^[1-9][0-9]*$ && "${rollout_steps}" =~ ^[1-9][0-9]*$ && "${num_target_sets}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: BASE_SEED must be non-negative and VIDEO_LENGTH, VIDEO_FPS, ROLLOUT_STEPS, and NUM_TARGET_SETS must be positive integers." >&2
    exit 1
fi

mkdir -p "${video_dir}"
cd "${project_root}"

for ((target_set_id = 1; target_set_id <= num_target_sets; target_set_id++)); do
    run_seed=$((base_seed + target_set_id - 1))
    echo "[INFO] Target-pose set ${target_set_id}/${num_target_sets}: object=${object_category}, seed=${run_seed}, grasp_poses=${num_grasp_poses}"

    "${python_bin}" scripts/rsl_rl/grasp_ranking_eval.py \
        "${extra_args[@]}" \
        --task Grasp-Ranking-EVAL \
        --object_name "${object_category}" \
        --target_set_id "${target_set_id}" \
        --seed "${run_seed}" \
        --num_envs 1 \
        --num_grasp_poses "${num_grasp_poses}" \
        --rollout_steps "${rollout_steps}" \
        --video \
        --video_length "${video_length}" \
        --video_fps "${video_fps}" \
        --video_folder "${video_dir}"
done

echo "[INFO] Completed ${num_target_sets} target-pose sets. Videos are in ${video_dir}."
