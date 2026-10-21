#!/bin/bash
# Queue position of a Slurm job on Lonestar6. Run locally (after `ssh -fN ls6`) or on a login node.
#   bash tacc/qstatus.sh 3519705
# Prints: state, wait so far, priority, pending jobs/nodes ahead in the same partition, the scheduler's start
# estimate, and a rough estimate = when the k-th running node frees up (k = nodes ahead + own nodes), from
# running jobs' remaining time limits. Rough means: ignores backfill, early finishes, and higher-priority
# jobs submitted later; jobs usually end before their limit, so the real start is often earlier.
JOB="${1:?usage: qstatus.sh JOBID}"
read -r -d '' REMOTE <<'EOF'
J=$1
st=$(squeue -j $J -h -o "%T|%P|%Q|%D|%V|%l" 2>/dev/null)
if [[ -z "$st" ]]; then sacct -j $J -X --format=JobIDRaw,State,Submit,Start,End,Elapsed,ExitCode; exit; fi
IFS='|' read -r state part prio nodes submit limit <<< "$st"
echo "job $J  $state  partition=$part  nodes=$nodes  limit=$limit  submitted=$submit  now=$(date +%FT%T)"
if [[ "$state" != "PENDING" ]]; then squeue -j $J -o "%.10i %.8T %.10M %.10L %R"; exit; fi
echo "waited: $(( ($(date +%s) - $(date -d "$submit" +%s)) / 3600 )) h"
echo "priority: $prio"
squeue -p $part -h -t PD -o "%Q %D" | awk -v m=$prio '{t++; if ($1>m) {n++; k+=$2}} END {printf "pending in %s: %d total, %d ahead of us (%d nodes)\n", "'$part'", t, n, k}'
echo "scheduler estimate: $(squeue -j $J -h --start -o %S)"
ahead=$(squeue -p $part -h -t PD -o "%Q %D" | awk -v m=$prio '$1>m {k+=$2} END {print k+0}')
need=$(( ahead + nodes ))
# remaining time of each running node, soonest first; the need-th one is the rough start
squeue -p $part -h -t R -o "%D %L" | awk '{
  s=$2; d=0; if (index(s,"-")) {split(s,a,"-"); d=a[1]; s=a[2]}
  n=split(s,b,":"); sec=(n==3?b[1]*3600+b[2]*60+b[3]:(n==2?b[1]*60+b[2]:b[1])) + d*86400
  for (i=0;i<$1;i++) print sec}' | sort -n | awk -v k=$need '
  NR==k {printf "rough estimate: ~%.1f h from now (when the %d-th running node frees up)\n", $1/3600, k; f=1}
  END {if (!f) printf "rough estimate: unknown (only %d running nodes, need %d)\n", NR, k}'
EOF
if [[ "$(hostname)" == login*.ls6* ]]; then bash -c "$REMOTE" _ "$JOB"
else ssh ls6 "bash -s -- $JOB" <<< "$REMOTE" 2>&1 | grep -v "Retrieving notices"; fi
