#!/bin/bash

# for SEED in 383 331 132 338 273 409 439 40 58 192 25 415 68 315 413 9 268 261 31 65 184 80 240 32 18 359 43 351 175 98 205 345 204 293 181 38 353 306 379 370 374 406 224 187 336 30 385 366 382 347 72 436 262 438 27 340 226 419 100 456 60 344 301 0 449 395 355; do
# for SEED in 132; do
# for SEED in 276 38 248 193 239 63 216 27 225; do
# for SEED in {362..500}; do
# 48 49
SEEDS=(91 338 273 257 488 147 258 280 292 140 102 49 388 64 165 358 88 415 68 315 76 243 333 368 84 177 31 460 373 184 254 119 433 483 500 381 7 367 94 28 222 151 390 118 112 109 391 497 172 495 217 441 183 402 473 330 179 99 38 410 353 89 314 335 384 244 427 363 370 467 352 310 287 360 48 249 484 320 188 382 365 347 494 325 197 21 262 150 438 154 349 455 340 456 115 62 322 260 355 45)

commandIndex=0
commandList=()
for SEED in {100..500}; do
  # for SEED in "${SEEDS[@]}"; do
  commandList+=("nohup python3 -u learning_dubins_ez.py $SEED 1 > logs/$SEED.log 2>&1")
  commandIndex=$((commandIndex + 1))
done
echo "${commandList[@]}"

numCommands=${#commandList[@]}

numThreads=4
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
