#!/bin/bash
#SBATCH -p batch
#SBATCH --account=nemotron_sw_post
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J ab-vllm
#
# A/B: of the tests that fail in test_vllm_generation.py, which fail on the Part 3
# branch ONLY (ours) and which fail on main too (not ours)?
#
# ---------------------------------------------------------------------------
# v3, after job 6298887 timed out having produced nothing. Three things went wrong
# and each is fixed here, because each will happen again otherwise:
#
#   1. ONE TEST HUNG FOR 3h45m AND --timeout=1200 NEVER FIRED. The v2 script used
#      --timeout-method=signal on the theory that the hang sits in ray.get and is
#      therefore Python-level and interruptible. It is not: 17 tests completed in the
#      first 14 minutes, the last worker output was at 16:10, and the job was killed
#      at 19:55 still inside the same test. So: thread method, which kills the process,
#      AND a coreutils `timeout` around the whole invocation as a backstop. Never trust
#      pytest-timeout alone again -- it has now been observed to miss this exact hang.
#
#   2. RESULTS LIVED ONLY IN ONE JUNIT FILE WRITTEN AT THE END. pytest was killed before
#      writing it, so all 17 completed results were lost. Now every test gets its own
#      pytest invocation and its own junit file, so a hang costs that test and nothing
#      else. Partial results are the normal case here, not the exception.
#
#   3. ALL OF A, THEN ALL OF B. Running out of time therefore left stage A only, which
#      answers nothing -- the comparison needs pairs. Now each test is run on BOTH refs
#      back to back, so whatever finishes yields complete, usable pairs.
#
# Also narrowed: v2 ran the whole 49-test file on both refs. Only the failing tests need
# adjudicating, and on 4 GPUs the rest actually run rather than skipping, which is where
# the four hours went.
# ---------------------------------------------------------------------------
#
#   sbatch submit-ab-vllm.sh
#
# Overridable: BRANCH_A, BRANCH_B, NEMO_RL, CONTAINER, GPUS_PER_NODE,
#              PER_TEST_TIMEOUT_S, RUN_FUNCTIONAL.
#
# Deliberately NOT `set -e`: a failing test is the RESULT here, not an error.
set -uo pipefail

BRANCH_A="${BRANCH_A:-feat/sc-resiliency-03-elastic-recovery}"
# Empty means "the main commit this branch actually contains", resolved after the fetch.
#
# NOT origin/main, deliberately. main has moved on past the merge-base -- five commits
# at the time of writing, two of them touching SingleController. Diffing A against the
# current tip mixes two variables, our delta and upstream's newer work, so a test that
# passes on B and fails on A could be upstream fixing something rather than us breaking
# it, and the script would report that as OUR regression. Against the merge-base the
# only difference is our delta, which is the question being asked.
#
# Set BRANCH_B=origin/main to ask the other question instead: will this branch still
# behave once it is synced forward again.
BRANCH_B="${BRANCH_B:-}"
NEMO_RL="${NEMO_RL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/RL}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/images/nemo-rl-nightly-gym.sqsh}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
# pytest-timeout gets this; the hard kill gets this + 120s of grace to let it try first.
PER_TEST_TIMEOUT_S="${PER_TEST_TIMEOUT_S:-480}"
HARD_KILL_S=$(( PER_TEST_TIMEOUT_S + 120 ))
RUN_FUNCTIONAL="${RUN_FUNCTIONAL:-0}"
FILE="tests/unit/models/generation/test_vllm_generation.py"

# Stop starting new work with this much walltime left, so the comparison and summary
# always get written. v2 died mid-test and wrote no verdict at all.
RESERVE_S="${RESERVE_S:-900}"
START_EPOCH=$(date +%s)
WALL_S="${WALL_S:-14400}"

# The tests under question, plus the two refit ones as controls: those failed with
# TypeError before 1a2bdd5a9 and should now pass on A and fail on B, which is a
# positive result that proves the harness can actually tell the refs apart.
TESTS=(
    "test_vllm_generation_with_hf_training_colocated[True-False-fp8-False]"
    "test_vllm_generation_with_hf_training_colocated[False-True-fp8-False]"
    "test_vllm_generation_with_hf_training_non_colocated[True-False-bfloat16-False]"
    "test_vllm_generation_with_hf_training_non_colocated[False-True-bfloat16-False]"
    "test_vllm_generation_with_hf_training_non_colocated[True-False-fp8-False]"
    "test_vllm_generation_with_hf_training_non_colocated[False-True-fp8-False]"
    "test_vllm_policy_tensor_parallel"
    "test_vllm_weight_update_and_prefix_cache_reset[fp8-1]"
    "test_vllm_refit_non_colocated_update_weights[dtensor-1-True]"
    "test_vllm_refit_non_colocated_update_weights[dtensor-1-False]"
)

BASE=$(dirname "$NEMO_RL")
RUN_DIR="$BASE/slurm/ab-${SLURM_JOB_ID:-manual}"
CACHE="$BASE/ci-cache"

HF_TOKEN="$(sed -n 's/^HF_TOKEN="\(.*\)"$/\1/p' "$BASE/submit-ci.sh" | head -1)"
if [[ -z "$HF_TOKEN" ]]; then
    echo "FATAL: could not read HF_TOKEN from $BASE/submit-ci.sh"; exit 1
fi

export HF_HOME="$CACHE/hf"
export HF_TOKEN
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-600}"
mkdir -p "$RUN_DIR" "$HF_HOME" "$CACHE/uv"

MOUNTS="$NEMO_RL:$NEMO_RL,$CACHE:$CACHE,$RUN_DIR:$RUN_DIR"
CNAME="nemorl-ab-${SLURM_JOB_ID:-manual}"

CONTAINER_ENV=(
    "HF_HOME=$HF_HOME"
    "HF_TOKEN=$HF_TOKEN"
    "UV_LOCK_TIMEOUT=$UV_LOCK_TIMEOUT"
    "UV_PYTHON_INSTALL_DIR=/root/.local/share/uv/python"
    "HOME=/root"
    # Ray forwards actor output to the driver, whose stdout here is a redirected file and
    # therefore block-buffered (job 6262310).
    "PYTHONUNBUFFERED=1"
)

_PREP='
for _d in /opt/nemo_rl_venv/bin /root/.local/bin; do
    [ -d "$_d" ] && PATH="$_d:$PATH"
done
export PATH
for _c in python3 uv; do
    command -v "$_c" >/dev/null || { echo "FATAL: $_c not on PATH in container"; exit 127; }
done
'

in_container() {
    if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
        env "${CONTAINER_ENV[@]}" bash -c "$_PREP$1"
        return
    fi
    srun --ntasks=1 --ntasks-per-node=1 --nodes=1 \
         --gpus-per-node="$GPUS_PER_NODE" \
         --container-image="$CONTAINER" \
         --container-name="$CNAME" \
         --container-mounts="$MOUNTS" \
         --no-container-mount-home \
         --container-workdir="$NEMO_RL" \
         --export=ALL,"$(IFS=,; echo "${CONTAINER_ENV[*]}")" \
         bash -c "$_PREP$1"
}

banner() { echo; echo "=============== $* ==============="; echo; }
budget_left() { echo $(( WALL_S - ( $(date +%s) - START_EPOCH ) )); }

ORIGINAL_REF="$(git -C "$NEMO_RL" rev-parse --abbrev-ref HEAD 2>/dev/null)"
[[ "$ORIGINAL_REF" == "HEAD" ]] && ORIGINAL_REF="$(git -C "$NEMO_RL" rev-parse HEAD)"
restore() {
    banner "restoring checkout to $ORIGINAL_REF"
    git -C "$NEMO_RL" checkout -q "$ORIGINAL_REF" 2>&1 | tail -3
    git -C "$NEMO_RL" submodule update --init --recursive 2>&1 | tail -3
    git -C "$NEMO_RL" log -1 --format='  now at %h %s'
}
trap restore EXIT

# ---------------------------------------------------------------------------
banner "pre-flight"
git -C "$NEMO_RL" fetch origin 2>&1 | tail -3

if [[ -z "$BRANCH_B" ]]; then
    BRANCH_B="$(git -C "$NEMO_RL" merge-base "$BRANCH_A" origin/main 2>/dev/null)"
    if [[ -z "$BRANCH_B" ]]; then
        echo "FATAL: could not resolve merge-base($BRANCH_A, origin/main)"; exit 1
    fi
    echo "  B resolved to the merge-base: $(git -C "$NEMO_RL" rev-parse --short "$BRANCH_B")"
    ahead=$(git -C "$NEMO_RL" rev-list --count "$BRANCH_B"..origin/main)
    echo "  (origin/main is $ahead commit(s) further on; excluded on purpose, see the header)"
fi
# A is only meaningful if it really does contain B, otherwise the two differ by more
# than our delta and every verdict below is suspect.
git -C "$NEMO_RL" merge-base --is-ancestor "$BRANCH_B" "$BRANCH_A" 2>/dev/null \
    || echo "  WARNING: $BRANCH_A does not contain $BRANCH_B -- verdicts will be confounded"

nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | sed 's/^/  gpu: /'
echo "  run dir:  $RUN_DIR"
echo "  budget:   ${WALL_S}s, reserving ${RESERVE_S}s for the summary"
echo "  per test: pytest-timeout ${PER_TEST_TIMEOUT_S}s, hard kill ${HARD_KILL_S}s"
echo "  tests:    ${#TESTS[@]} x 2 refs"

checkout() {
    local ref="$1" bare="${1#origin/}"
    if git -C "$NEMO_RL" rev-parse --verify -q "origin/$bare" >/dev/null; then
        git -C "$NEMO_RL" checkout -q -B "$bare" "origin/$bare" 2>&1 | tail -2
    else
        git -C "$NEMO_RL" checkout -q "$ref" 2>&1 | tail -2
    fi
    git -C "$NEMO_RL" submodule update --init --recursive 2>&1 | tail -3
    # A '+' means a submodule sits off its pinned commit and PYTHONPATH puts that
    # worktree first -- the run would test the wrong Megatron-Bridge and report a pass.
    git -C "$NEMO_RL" submodule status --recursive | grep '^+' | sed 's/^/  WARNING pin mismatch: /'
}

# One test, one ref, one junit file. Returns the verdict on stdout.
run_one() {
    local label="$1" test="$2" idx="$3"
    local xml="$RUN_DIR/${label}-${idx}.xml"
    local log="$RUN_DIR/${label}-${idx}.log"
    # --timeout-method=thread, not signal: signal did not interrupt this hang (6298887).
    # `timeout --signal=KILL` outside it because pytest-timeout has now been seen to miss.
    in_container "cd $NEMO_RL && timeout --signal=KILL ${HARD_KILL_S} \
        uv run --no-sync python -m pytest '$FILE::$test' \
        --maxfail=0 --timeout=${PER_TEST_TIMEOUT_S} --timeout-method=thread \
        -p no:cacheprovider --junitxml=$xml -q 2>&1" > "$log" 2>&1
    local rc=$?
    if [[ -s "$xml" ]]; then
        python3 - "$xml" <<'PY'
import sys, xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
except Exception:
    print("unparseable"); sys.exit()
v = "passed"
for case in root.iter("testcase"):
    for child in case:
        if child.tag in ("failure", "error"):
            v = child.tag
        elif child.tag == "skipped":
            v = "skipped"
print(v)
PY
    elif [[ $rc -eq 137 ]]; then
        echo "HUNG(killed at ${HARD_KILL_S}s)"
    else
        echo "no-junit(rc=$rc)"
    fi
}

# ---------------------------------------------------------------------------
# Per test: A then B, so partial results are still complete PAIRS.
# ---------------------------------------------------------------------------
RESULTS="$RUN_DIR/RESULTS.tsv"
printf 'test\tA(%s)\tB(%s)\tverdict\n' "$BRANCH_A" "$BRANCH_B" > "$RESULTS"

idx=0
for test in "${TESTS[@]}"; do
    idx=$((idx + 1))
    left=$(budget_left)
    if (( left < RESERVE_S + 2 * HARD_KILL_S )); then
        printf '%s\tNOT-RUN\tNOT-RUN\tout of budget\n' "$test" >> "$RESULTS"
        echo "  [$idx/${#TESTS[@]}] SKIPPED, ${left}s left -- not enough for a pair"
        continue
    fi
    banner "[$idx/${#TESTS[@]}] $test   (${left}s budget left)"

    checkout "$BRANCH_A" >/dev/null 2>&1
    a=$(run_one A "$test" "$idx")
    echo "  A ($BRANCH_A): $a"

    checkout "$BRANCH_B" >/dev/null 2>&1
    b=$(run_one B "$test" "$idx")
    echo "  B ($BRANCH_B): $b"

    # Classify on "is this outcome bad", not by enumerating outcome pairs. The pair-
    # matching version of this silently mis-filed a real regression in job 6321283:
    # A came back "no-junit(rc=1)" (pytest-timeout killed the process before it could
    # write junit), which matched none of the REGRESSION arms, fell through to the
    # catch-all, and was reported as "pre-existing" against a B that passed in 61s.
    # A catch-all that names a specific conclusion is a bug waiting to happen -- anything
    # unanticipated lands in it wearing the wrong label.
    bad() {
        case "$1" in
            failure|error|HUNG*|no-junit*|unparseable) return 0 ;;
            *) return 1 ;;
        esac
    }
    if [[ "$a" == passed && "$b" == passed ]]; then      verdict="both pass"
    elif [[ "$a" == skipped && "$b" == skipped ]]; then   verdict="skipped both (hardware gate)"
    elif bad "$a" && [[ "$b" == passed ]]; then           verdict="REGRESSION (ours)"
    elif [[ "$a" == passed ]] && bad "$b"; then           verdict="fixed by the branch"
    elif bad "$a" && bad "$b"; then                       verdict="fails on both -- pre-existing"
    else                                                  verdict="INCONCLUSIVE ($a vs $b) -- read the logs"
    fi
    printf '%s\t%s\t%s\t%s\n' "$test" "$a" "$b" "$verdict" >> "$RESULTS"
    echo "  => $verdict"
    # Written after every pair, so a kill still leaves everything decided so far.
    column -t -s $'\t' "$RESULTS" > "$RUN_DIR/RESULTS.txt" 2>/dev/null || cp "$RESULTS" "$RUN_DIR/RESULTS.txt"
done

# ---------------------------------------------------------------------------
if [[ "$RUN_FUNCTIONAL" == "1" ]] && (( $(budget_left) > 3600 )); then
    banner "functional: shard recovery on $BRANCH_A"
    checkout "$BRANCH_A"
    for t in recovery recovery-reshard; do
        (( $(budget_left) < 1800 )) && { echo "  out of budget, skipping $t"; break; }
        echo "--- $t ---"
        case "$t" in
            recovery)         cmd="uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh" ;;
            recovery-reshard) cmd="env REFIT_TRANSPORT=nccl_reshard uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh" ;;
        esac
        in_container "cd $NEMO_RL && $cmd 2>&1" > "$RUN_DIR/func-$t.log" 2>&1
        echo "  rc=$?"
        grep -qE "\[recovery\] SKIP" "$RUN_DIR/func-$t.log" && echo "  SKIPPED -- not a pass"
        grep -E "\[recovery\] (PASS|FAIL)|refit membership absent=" "$RUN_DIR/func-$t.log" | tail -5 | sed 's/^/  /'
    done
fi

# ---------------------------------------------------------------------------
banner "RESULTS"
column -t -s $'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"
echo
regressions=$(grep -c 'REGRESSION' "$RESULTS")
notrun=$(grep -c 'NOT-RUN' "$RESULTS")
echo "  regressions: $regressions"
echo "  not run:     $notrun"
if (( regressions > 0 )); then
    echo "  VERDICT: $regressions REGRESSION(S) -- do not merge until explained"
elif (( notrun > 0 )); then
    echo "  VERDICT: no regressions among the tests that ran, but $notrun never ran"
else
    echo "  VERDICT: CLEAN -- every test adjudicated, nothing regressed"
fi
echo
echo "  everything under: $RUN_DIR"
echo "    RESULTS.txt     the table above"
echo "    A-N.log/B-N.log per-test output, N is the row order above"
