#!/bin/bash

# for SEED in {119..119}; do
for SEED in {1..1}; do
  # for SEED in 72 303 463; do
  echo "Running seed $SEED"
  python learning_dubins_ez.py $SEED
done
