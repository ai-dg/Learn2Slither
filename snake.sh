#!/usr/bin/env bash

set -e

echo "Learn2Slither - Snake with Reinforcement Learning"

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install it."
    exit 1
fi

COMPONENTS=(
    numpy
    tensorflow
    pygame
    keras
    tabulate
)

FILE="./nbr_pkg.txt"
required_count=${#COMPONENTS[@]}

if [ -f "$FILE" ]; then
    export NUMBER_OF_PKG=$(cat "$FILE")
else
    NUMBER_OF_PKG=0
fi

if [ "${NUMBER_OF_PKG:-0}" -ne "$required_count" ]; then
    number_of_pkg=0
    for component in "${COMPONENTS[@]}"; do
        if ! python3 -c "import $component" &> /dev/null; then
            echo "Installing $component..."
            python3 -m pip install "$component" >/dev/null
        else
            echo "$component already installed"
        fi
        ((++number_of_pkg))
    done
    echo "Python packages verified/installed: $NUMBER_OF_PKG"
    echo "$number_of_pkg" > "$FILE"
else
    echo "All Python packages already verified ($NUMBER_OF_PKG total). Skipping check."
fi

exec python3 ./snake.py "$@"


