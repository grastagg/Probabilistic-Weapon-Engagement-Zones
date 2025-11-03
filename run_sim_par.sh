#!/bin/bash

SEEDS=(24 12 37 1 48 16 39)

commandIndex=0
commandList=()
for SEED in {102..500}; do
  # for SEED in "${SEEDS[@]}"; do
  commandList+=("nohup python3 -u learning_dubins_ez.py $SEED 1 > logs/$SEED.log 2>&1")
  commandIndex=$((commandIndex + 1))
done
echo "${commandList[@]}"

numCommands=${#commandList[@]}

numThreads=3
currentCommand=0
for cmd in "${commandList[@]}"; do
  # wait until fewer than numThreads jobs are running
  while (($(jobs -rp | wc -l) >= numThreads)); do
    wait -n # bash ≥ 5; else: sleep 0.2
  done
  echo "Running command: $cmd"
  bash -lc "$cmd" &
done
wait
