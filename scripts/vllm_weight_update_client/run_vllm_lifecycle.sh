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

reset_cache() {
  local reply
  reply=$(post "reset_prefix_cache?reset_running_requests=false")
  echo "$reply"
  [[ $reply =~ \"success\"[[:space:]]*:[[:space:]]*true ]]
}

$abort_inflight && post abort_requests -H content-type:application/json -d '{}'
if $offload; then
  reset_cache
  post "sleep?level=2&mode=abort"
  post "wake_up?tags=weights"
fi
post "pause?mode=keep&clear_cache=false" -H content-type:application/json -d '{}'
reset_cache
rc=0; "$@" || rc=$?
((rc == 0)) || { echo "publisher failed; service remains paused" >&2; exit "$rc"; }
post resume -H content-type:application/json -d '{}'
if $offload; then
  post "wake_up?tags=kv_cache"
fi
