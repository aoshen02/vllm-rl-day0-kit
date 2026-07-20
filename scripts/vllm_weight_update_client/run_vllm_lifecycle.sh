#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "usage: $0 URL [--abort-inflight] [--offload] -- PUBLISHER..." >&2
}
[[ $# -ge 3 ]] || { usage; exit 2; }
base_url=${1%/}
shift
abort_inflight=false
offload=false
while [[ ${1:-} != -- ]]; do
  case ${1:-} in
    --abort-inflight) abort_inflight=true ;;
    --offload) offload=true ;;
    *) usage; exit 2 ;;
  esac
  shift
done
shift
[[ $# -gt 0 ]] || { usage; exit 2; }

post() {
  echo "+ POST /$1" >&2
  curl --fail --show-error --silent --max-time "${VLLM_CONTROL_TIMEOUT:-600}" \
    -X POST "$base_url/$1" "${@:2}"
  echo
}

wait_until_ready() {
  local deadline=$((SECONDS + ${VLLM_READY_TIMEOUT:-600}))
  local sleeping paused poll_timeout=${VLLM_READY_POLL_TIMEOUT:-5}
  while (( SECONDS < deadline )); do
    sleeping=$(curl --fail --show-error --silent --max-time "$poll_timeout" \
      "$base_url/is_sleeping" || true)
    paused=$(curl --fail --show-error --silent --max-time "$poll_timeout" \
      "$base_url/is_paused" || true)
    if [[ $sleeping =~ \"is_sleeping\"[[:space:]]*:[[:space:]]*false && \
          $paused =~ \"is_paused\"[[:space:]]*:[[:space:]]*false ]]; then
      echo "+ READY /is_sleeping=$sleeping /is_paused=$paused" >&2
      return 0
    fi
    sleep 1
  done
  echo "service did not become ready after lifecycle transition" >&2
  return 1
}

reset_cache() {
  local reply
  reply=$(post "reset_prefix_cache?reset_running_requests=false")
  echo "$reply"
  [[ $reply =~ \"success\"[[:space:]]*:[[:space:]]*true ]]
}

$abort_inflight && post abort_requests -H content-type:application/json -d '{}'
if $offload; then
  reset_cache
  sleep_level=${VLLM_SLEEP_LEVEL:-2}
  sleep_mode=${VLLM_SLEEP_MODE:-abort}
  post "sleep?level=${sleep_level}&mode=${sleep_mode}"
  post "wake_up?tags=weights"
fi
post "pause?mode=keep&clear_cache=false" -H content-type:application/json -d '{}'
reset_cache
rc=0; "$@" || rc=$?
((rc == 0)) || { echo "publisher failed; service remains paused" >&2; exit "$rc"; }
post resume -H content-type:application/json -d '{}'
if $offload; then
  post "wake_up?tags=kv_cache"
  wait_until_ready
fi
