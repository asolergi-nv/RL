#!/bin/bash
#SBATCH -p batch
#SBATCH --account=nemotron_sw_post
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 01:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J confirm-hang
#
# Confirm the fix for the two non-colocated hangs found in job 6321283.
#
# THE BUG. VllmGeneration.update_weights_from_collective picks the worker method from
# config -- "..._async" when vllm_cfg.async_engine is true, the plain name otherwise --
# then forwards refit_timeout_s to whichever it picked. 8cd8e565a added that kwarg to the
# async worker and to the forwarding site but not to the sync one, so with
# async_engine=False the generation actor raised TypeError at the Ray boundary, never
# joined the NCCL broadcast, and the training side blocked in ray.get forever.
#
# That is exactly the observed split: the async sibling of each hanging test passed.
#
# THE CHECK. Run the two that hung and the two that passed. The two async ones are the
# controls: they passed before the fix, so if they now fail the fix broke the working
# path, and "the hangs are gone" would mean nothing on its own.
#
# Branch only -- main is already known to pass all four (job 6321283), so a B side would
# just spend GPU time re-confirming that.
#
# Expected: 4 passed, no HUNG. Anything else and the diagnosis is incomplete.
#
#   sbatch submit-confirm-hang.sh          # ~10 min
#
set -uo pipefail

BRANCH="${BRANCH:-feat/sc-resiliency-03-elastic-recovery}"
NEMO_RL="${NEMO_RL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/RL}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/images/nemo-rl-nightly-gym.sqsh}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PER_TEST_TIMEOUT_S="${PER_TEST_TIMEOUT_S:-480}"
HARD_KILL_S=$(( PER_TEST_TIMEOUT_S + 120 ))
FILE="tests/unit/models/generation/test_vllm_generation.py"

# id order is the reverse of the decorator order, so the LAST element is async_engine
# for the refit test. The two that hung are the async_engine=False ones.
TESTS=(
    "test_vllm_refit_non_colocated_update_weights[dtensor-1-False]|HUNG before the fix"
    "test_vllm_generation_with_hf_training_non_colocated[False-True-bfloat16-False]|HUNG before the fix"
    "test_vllm_refit_non_colocated_update_weights[dtensor-1-True]|control, passed before"
    "test_vllm_generation_with_hf_training_non_colocated[True-False-bfloat16-False]|control, passed before"
)

BASE=$(dirname "$NEMO_RL")
RUN_DIR="$BASE/slurm/confirm-${SLURM_JOB_ID:-manual}"
CACHE="$BASE/ci-cache"

HF_TOKEN="$(sed -n 's/^HF_TOKEN="\(.*\)"$/\1/p' "$BASE/submit-ci.sh" | head -1)"
[[ -z "$HF_TOKEN" ]] && { echo "FATAL: no HF_TOKEN in $BASE/submit-ci.sh"; exit 1; }

export HF_HOME="$CACHE/hf"
export HF_TOKEN
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-600}"
mkdir -p "$RUN_DIR" "$HF_HOME" "$CACHE/uv"

MOUNTS="$NEMO_RL:$NEMO_RL,$CACHE:$CACHE,$RUN_DIR:$RUN_DIR"
CNAME="nemorl-confirm-${SLURM_JOB_ID:-manual}"
CONTAINER_ENV=(
    "HF_HOME=$HF_HOME" "HF_TOKEN=$HF_TOKEN" "UV_LOCK_TIMEOUT=$UV_LOCK_TIMEOUT"
    "UV_PYTHON_INSTALL_DIR=/root/.local/share/uv/python" "HOME=/root" "PYTHONUNBUFFERED=1"
)
_PREP='
for _d in /opt/nemo_rl_venv/bin /root/.local/bin; do
    [ -d "$_d" ] && PATH="$_d:$PATH"
done
export PATH
for _c in python3 uv; do
    command -v "$_c" >/dev/null || { echo "FATAL: $_c not on PATH"; exit 127; }
done
'
in_container() {
    if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
        env "${CONTAINER_ENV[@]}" bash -c "$_PREP$1"; return
    fi
    srun --ntasks=1 --ntasks-per-node=1 --nodes=1 --gpus-per-node="$GPUS_PER_NODE" \
         --container-image="$CONTAINER" --container-name="$CNAME" \
         --container-mounts="$MOUNTS" --no-container-mount-home \
         --container-workdir="$NEMO_RL" \
         --export=ALL,"$(IFS=,; echo "${CONTAINER_ENV[*]}")" bash -c "$_PREP$1"
}
banner() { echo; echo "=============== $* ==============="; echo; }

ORIGINAL_REF="$(git -C "$NEMO_RL" rev-parse --abbrev-ref HEAD 2>/dev/null)"
trap 'banner "restoring to $ORIGINAL_REF"; git -C "$NEMO_RL" checkout -q "$ORIGINAL_REF" 2>&1|tail -2; git -C "$NEMO_RL" submodule update --init --recursive 2>&1|tail -2' EXIT

banner "pre-flight"
git -C "$NEMO_RL" fetch origin 2>&1 | tail -2
git -C "$NEMO_RL" checkout -q -B "${BRANCH#origin/}" "origin/${BRANCH#origin/}" 2>&1 | tail -2
git -C "$NEMO_RL" submodule update --init --recursive 2>&1 | tail -3
git -C "$NEMO_RL" log -1 --format='  HEAD: %h %s'
git -C "$NEMO_RL" submodule status --recursive | grep '^+' | sed 's/^/  WARNING pin mismatch: /'

# The fix must actually be in the checkout. Cheap, and it is the whole premise: without
# it every result below is about the old code.
#
# Assert the kwarg is PRESENT rather than that the old text is absent. The first version
# of this guard tested `def update_weights_from_collective($` for the old signature and
# had it exactly backwards -- the fixed signature is the one that wraps after the paren,
# so it matched the fix and missed the bug. Matching on what must be true beats matching
# on what must be gone.
if ! grep -A3 'def update_weights_from_collective(' \
        "$NEMO_RL/nemo_rl/models/generation/vllm/vllm_worker.py" \
        | grep -q 'refit_timeout_s'; then
    echo "  FATAL: the sync worker does not take refit_timeout_s -- the fix is not in this"
    echo "         checkout, so every result below would be about the old code."
    exit 1
fi
echo "  fix present: sync worker takes refit_timeout_s"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | sed 's/^/  gpu: /'

RESULTS="$RUN_DIR/RESULTS.txt"
: > "$RESULTS"
idx=0; fails=0
for entry in "${TESTS[@]}"; do
    test="${entry%%|*}"; note="${entry##*|}"
    idx=$((idx + 1))
    banner "[$idx/${#TESTS[@]}] $test  ($note)"
    xml="$RUN_DIR/t${idx}.xml"; log="$RUN_DIR/t${idx}.log"
    # thread method + an outer hard kill: in job 6321283 pytest-timeout fired on one of
    # these tests and silently did not on the other, and only the KILL bounded it.
    in_container "cd $NEMO_RL && timeout --signal=KILL ${HARD_KILL_S} \
        uv run --no-sync python -m pytest '$FILE::$test' \
        --maxfail=0 --timeout=${PER_TEST_TIMEOUT_S} --timeout-method=thread \
        -p no:cacheprovider --junitxml=$xml -q 2>&1" > "$log" 2>&1
    rc=$?
    if [[ -s "$xml" ]]; then
        v=$(in_container "python3 -c \"
import xml.etree.ElementTree as ET
r=ET.parse('$xml').getroot()
v='passed'
for c in r.iter('testcase'):
    for ch in c:
        if ch.tag in ('failure','error'): v=ch.tag
        elif ch.tag=='skipped': v='skipped'
print(v)\"" 2>/dev/null | tr -d '\r\n ')
    elif [[ $rc -eq 137 ]]; then v="HUNG(hard-killed at ${HARD_KILL_S}s)"
    else v="no-junit(rc=$rc)"
    fi
    [[ "$v" != "passed" ]] && fails=$((fails + 1))
    printf '%-70s %-14s %s\n' "$test" "$v" "$note" >> "$RESULTS"
    echo "  -> $v"
    grep -E '^[0-9]+ (passed|failed)|passed,|failed,' "$log" | tail -1 | sed 's/^/  /'
done

banner "RESULTS"
cat "$RESULTS"
echo
if (( fails == 0 )); then
    echo "  VERDICT: CONFIRMED -- both hangs are gone and neither control regressed."
else
    echo "  VERDICT: NOT CONFIRMED -- $fails of ${#TESTS[@]} did not pass. Read $RUN_DIR/tN.log."
    echo "           A control failing means the fix broke the path that already worked."
fi
echo
echo "  logs: $RUN_DIR"
