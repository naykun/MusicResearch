#!/bin/bash
export OMP_NUM_THREADS=1
set -euo pipefail
children=()

_term() {
    echo -e "\n\033[0;31mCaught SIGTERM signal.\033[0m"
    for i in "${children[@]}"
    do
        kill -9 "$i" 2> /dev/null || true
    done
}

while read cmd; do
    $cmd &
    children+=($!)
done

trap _term SIGINT

wait "${children[0]}"
