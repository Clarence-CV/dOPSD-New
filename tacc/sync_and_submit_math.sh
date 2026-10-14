#!/bin/bash
# Run LOCALLY (WSL) after `ssh -fN ls6`. Syncs this working tree to $WORK/inf385t/dOPSD-sd on Lonestar6
# (a separate directory, so the THUdOPSD checkout in $WORK/inf385t/dOPSD is untouched), checks the queue
# limits, and submits tacc/selective_draft_math.slurm from there.
#   bash tacc/sync_and_submit_math.sh              # sync + qlimits + sbatch
#   DRY=1 bash tacc/sync_and_submit_math.sh        # sync + qlimits only
# Knobs are passed through to the job: RUN, N_TRAIN, N_ANALYSIS, N_EVAL, READOUT_STEPS, GUIDE_STEPS, ...
set -euo pipefail
cd "$(dirname "$0")/.."
ssh -O check ls6 >/dev/null 2>&1 || { echo "no ls6 master connection; run: ssh -fN ls6"; exit 1; }
REMOTE_WORK=$(ssh ls6 'echo $WORK')
DEST="$REMOTE_WORK/inf385t/dOPSD-sd"
rsync -az --delete --exclude '.git/' --exclude 'outputs/' --exclude 'logs/' --exclude '__pycache__/' \
      --exclude 'checkpoints/' --exclude 'wandb/' ./ "ls6:$DEST/"
git rev-parse HEAD > /tmp/sd_rev && git status --short > /tmp/sd_status
scp -q /tmp/sd_rev "ls6:$DEST/.code_revision"; scp -q /tmp/sd_status "ls6:$DEST/.code_status"
echo "synced to $DEST at $(cat /tmp/sd_rev)"
ssh ls6 "qlimits 2>/dev/null | grep -E 'Queue|gpu-a100-small' || true"
PASS=""
for v in RUN N_TRAIN N_ANALYSIS N_EVAL N_SMOKE READOUT_STEPS GUIDE_STEPS COLLECT_BS EVAL_THRESHOLDS DIAG_STATES; do
    [[ -n "${!v:-}" ]] && PASS+="$v='${!v}' "
done
if [[ "${DRY:-0}" == "1" ]]; then echo "DRY=1: not submitting"; exit 0; fi
ssh ls6 "cd $DEST && mkdir -p logs && env $PASS sbatch --parsable tacc/selective_draft_math.slurm" | tee /tmp/sd_jobid
echo "submitted job $(cat /tmp/sd_jobid); monitor: ssh ls6 'squeue -u \$USER'"
