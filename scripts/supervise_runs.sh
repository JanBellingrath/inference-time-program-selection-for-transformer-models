#!/usr/bin/env bash
# =============================================================================
# Generic sequential job supervisor.
#
# Runs a list of jobs one after another, in the order given. Each job is a
# single shell command (typically a wrapper script). For each job:
#   * a per-job log file is written under --log_dir/<job_name>.log
#   * the master log records start/end/exit-code/elapsed-time
#   * on non-zero exit, the supervisor sleeps RETRY_SLEEP seconds and retries
#     the job up to MAX_ATTEMPTS times
#   * after MAX_ATTEMPTS the supervisor either continues to the next job
#     (default) or stops (with --stop_on_failure)
#
# Job manifest format (--jobs <file>):
#   - one job per line
#   - blank lines and lines starting with '#' are ignored
#   - syntax: NAME = COMMAND...
#       NAME may not contain whitespace or '='; COMMAND is everything after
#       the first '='. NAME is used for the per-job log filename.
#   - if a line has no '=', it is treated as both NAME (basename of first
#     token) and COMMAND
#
# Example manifest (scripts/runs_assign_dense_jobs.txt):
#   csqa_nonft_assign_increment = bash scripts/mine_csqa_nonft_assign_increment.sh
#   hellaswag_assign_increment  = bash scripts/mine_hellaswag_assign_increment.sh
#
# Launch (foreground):
#   bash scripts/supervise_runs.sh --jobs scripts/runs_assign_dense_jobs.txt \
#        --log_dir runs/assign_dense_$(date +%Y%m%d_%H%M%S)
#
# Launch (detached, recommended for long jobs):
#   nohup bash scripts/supervise_runs.sh \
#        --jobs scripts/runs_assign_dense_jobs.txt \
#        --log_dir runs/assign_dense_$(date +%Y%m%d_%H%M%S) \
#        > /tmp/supervise_runs_nohup.txt 2>&1 &
# =============================================================================
set -u

JOBS_FILE=""
LOG_DIR=""
MAX_ATTEMPTS=3
RETRY_SLEEP=60
STOP_ON_FAILURE=0

print_help() {
  sed -n '2,40p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)            JOBS_FILE="$2"; shift 2 ;;
    --log_dir)         LOG_DIR="$2"; shift 2 ;;
    --max_attempts)    MAX_ATTEMPTS="$2"; shift 2 ;;
    --retry_sleep)     RETRY_SLEEP="$2"; shift 2 ;;
    --stop_on_failure) STOP_ON_FAILURE=1; shift ;;
    -h|--help)         print_help; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; print_help; exit 2 ;;
  esac
done

if [[ -z "$JOBS_FILE" ]]; then
  echo "FATAL: --jobs <file> is required" >&2
  print_help
  exit 2
fi
if [[ ! -f "$JOBS_FILE" ]]; then
  echo "FATAL: jobs file not found: $JOBS_FILE" >&2
  exit 2
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/supervise_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/supervisor.log"
SUMMARY_TSV="$LOG_DIR/jobs_summary.tsv"

mlog() {
  local ts
  ts="$(date -Iseconds)"
  echo "[$ts] $*" | tee -a "$MASTER_LOG"
}

mlog "=== supervise_runs ==="
mlog "JOBS_FILE=$JOBS_FILE"
mlog "LOG_DIR=$LOG_DIR"
mlog "MAX_ATTEMPTS=$MAX_ATTEMPTS  RETRY_SLEEP=${RETRY_SLEEP}s  STOP_ON_FAILURE=$STOP_ON_FAILURE"

printf "job_name\tstatus\tattempts\texit_code\telapsed_sec\tlog_path\n" > "$SUMMARY_TSV"

# Parse jobs file into two parallel arrays: names[] and commands[].
declare -a names=()
declare -a commands=()
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue
  if [[ "$line" == *"="* ]]; then
    name="${line%%=*}"
    cmd="${line#*=}"
    name="${name%"${name##*[![:space:]]}"}"
    name="${name#"${name%%[![:space:]]*}"}"
    cmd="${cmd#"${cmd%%[![:space:]]*}"}"
  else
    cmd="$line"
    first_tok="${line%% *}"
    name="$(basename "$first_tok" .sh)"
  fi
  if [[ -z "$name" || -z "$cmd" ]]; then
    mlog "WARN: skipping malformed line: $raw_line"
    continue
  fi
  names+=("$name")
  commands+=("$cmd")
done < "$JOBS_FILE"

n_jobs=${#names[@]}
if [[ "$n_jobs" -eq 0 ]]; then
  mlog "FATAL: no jobs parsed from $JOBS_FILE"
  exit 2
fi
mlog "Parsed $n_jobs job(s)."

global_status=0
for ((i = 0; i < n_jobs; i++)); do
  name="${names[$i]}"
  cmd="${commands[$i]}"
  job_log="$LOG_DIR/${name}.log"
  mlog "--- Job $((i+1))/$n_jobs: name='$name' ---"
  mlog "    cmd: $cmd"
  mlog "    log: $job_log"

  attempt=0
  job_status="failed"
  exit_code=1
  job_t0=$(date +%s)
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))
    mlog ">>> $name attempt $attempt/$MAX_ATTEMPTS starting"
    a_t0=$(date +%s)
    {
      echo "[$(date -Iseconds)] === $name attempt $attempt ==="
      echo "[$(date -Iseconds)] cmd: $cmd"
    } >> "$job_log"
    bash -c "$cmd" >> "$job_log" 2>&1
    exit_code=$?
    a_t1=$(date +%s)
    mlog ">>> $name attempt $attempt finished exit=$exit_code in $((a_t1 - a_t0))s"
    if [[ "$exit_code" -eq 0 ]]; then
      job_status="ok"
      break
    fi
    if (( attempt < MAX_ATTEMPTS )); then
      mlog ">>> $name FAILED (exit $exit_code). Sleeping ${RETRY_SLEEP}s before retry."
      sleep "$RETRY_SLEEP"
    fi
  done
  job_t1=$(date +%s)
  job_elapsed=$((job_t1 - job_t0))

  printf "%s\t%s\t%d\t%d\t%d\t%s\n" \
    "$name" "$job_status" "$attempt" "$exit_code" "$job_elapsed" "$job_log" \
    >> "$SUMMARY_TSV"

  if [[ "$job_status" == "ok" ]]; then
    mlog "*** Job '$name' OK after $attempt attempt(s), elapsed ${job_elapsed}s"
  else
    mlog "*** Job '$name' FAILED after $attempt attempt(s), exit $exit_code, elapsed ${job_elapsed}s"
    global_status=1
    if [[ "$STOP_ON_FAILURE" -eq 1 ]]; then
      mlog "STOP_ON_FAILURE=1; skipping remaining jobs."
      break
    fi
  fi
done

mlog "=== supervise_runs done (global_status=$global_status) ==="
mlog "Summary: $SUMMARY_TSV"
exit "$global_status"
