#!/bin/bash

SEEDS=(24 12 37 1 48 16 39)

commandIndex=0
commandList=()
for SEED in {0..0}; do
  # for SEED in "${SEEDS[@]}"; do
  commandList+=("nohup python3 -m GEOMETRIC_BEZ.monte_carlo_runner $SEED 0 0 > logs/$SEED.log 2>&1")
  commandList+=("nohup python3 -m GEOMETRIC_BEZ.monte_carlo_runner $SEED 1 0 > logs/$SEED.log 2>&1")
  commandList+=("nohup python3 -m GEOMETRIC_BEZ.monte_carlo_runner $SEED 0 1 > logs/$SEED.log 2>&1")
  commandList+=("nohup python3 -m GEOMETRIC_BEZ.monte_carlo_runner $SEED 1 1 > logs/$SEED.log 2>&1")
  # commandList+=("python3 -m GEOMETRIC_BEZ.sacraficial_agent_planner $SEED")
  commandIndex=$((commandIndex + 1))
done
echo "${commandList[@]}"

numCommands=${#commandList[@]}

numThreads=5
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
