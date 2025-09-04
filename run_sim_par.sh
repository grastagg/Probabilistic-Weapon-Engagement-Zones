#!/bin/bash

# for SEED in 383 331 132 338 273 409 439 40 58 192 25 415 68 315 413 9 268 261 31 65 184 80 240 32 18 359 43 351 175 98 205 345 204 293 181 38 353 306 379 370 374 406 224 187 336 30 385 366 382 347 72 436 262 438 27 340 226 419 100 456 60 344 301 0 449 395 355; do
# for SEED in 132; do
# for SEED in 276 38 248 193 239 63 216 27 225; do
# for SEED in {362..500}; do
SEEDS=(3 18 42 114 120 123 125 129 222 223 272 282 307 309 331 339 348 352 363 364 384 400 406 409 416 420 430 437 439 440 441 452 470 487 491 499)

commandIndex=0
commandList=()
for SEED in {0..500}; do
  # for SEED in "${SEEDS[@]}"; do
  commandList+=("nohup python3 -u learning_dubins_ez.py $SEED 1 > logs/$SEED.log 2>&1")
  commandIndex=$((commandIndex + 1))
done
echo "${commandList[@]}"

numCommands=${#commandList[@]}

numThreads=2
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
