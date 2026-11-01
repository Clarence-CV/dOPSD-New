#!/bin/bash
# Run LOCALLY (WSL) after `ssh -fN ls6` (or `ssh -fN vista` with TACC_HOST=vista). Syncs this working tree to
# $WORK/inf385t/dOPSD-sd on that system, shows the queue limits, and submits tacc/llada2_pipeline.slurm from there.
# Knobs are passed through to the job. Vista: TACC_HOST=vista SBATCH_ARGS="-p gh -t 30:00:00" (1 GH200 per node).
#   SMOKE=1 STAGES="baseline states profile readout joint jointeval" SBATCH_ARGS="-p gpu-a100-small -t 00:45:00" \
#       bash tacc/sync_and_submit_llada2.sh
#   STAGES=baseline SBATCH_ARGS="-t 03:00:00" bash tacc/sync_and_submit_llada2.sh
#   DRY=1 bash tacc/sync_and_submit_llada2.sh        # sync + qlimits only
set -euo pipefail
cd "$(dirname "$0")/.."
H="${TACC_HOST:-ls6}"
ssh -O check "$H" >/dev/null 2>&1 || { echo "no $H master connection; run: ssh -fN $H"; exit 1; }
REMOTE_WORK=$(ssh "$H" 'echo $WORK')
DEST="$REMOTE_WORK/inf385t/dOPSD-sd"
rsync -az --delete --exclude '.git/' --exclude 'outputs/' --exclude 'logs/' --exclude '__pycache__/' \
      --exclude 'checkpoints/' --exclude 'wandb/' ./ "$H:$DEST/"
git rev-parse HEAD > /tmp/sd_rev && git status --short > /tmp/sd_status
scp -q /tmp/sd_rev "$H:$DEST/.code_revision"; scp -q /tmp/sd_status "$H:$DEST/.code_status"
echo "synced to $DEST at $(cat /tmp/sd_rev)$( [[ -s /tmp/sd_status ]] && echo ' (+uncommitted changes)')"
ssh "$H" "qlimits 2>/dev/null | grep -E 'Queue|gpu-a100' || true"
PASS=""
for v in RUN STAGES SMOKE EVAL_SPLIT EVAL_SUBSPLIT N_EVAL READOUT_STEPS JOINT_STEPS READOUT_SELECT CONFIG STATE_SETS VAL_N CONF_N TRAIN_N DEV_N FINAL_TEST JOINT_EVAL_STEPS TESTS MOECHECK_STATES MOECHECK_NGEN MOE_IMPL; do
    [[ -n "${!v:-}" ]] && PASS+="$v='${!v}' "
done
if [[ "${DRY:-0}" == "1" ]]; then echo "DRY=1: not submitting"; exit 0; fi
ssh "$H" "cd $DEST && mkdir -p logs && env $PASS sbatch --parsable ${SBATCH_ARGS:-} tacc/llada2_pipeline.slurm" | tee /tmp/llada2_jobid
echo "submitted job $(cat /tmp/llada2_jobid); monitor: ssh "$H" 'squeue -u \$USER'"
