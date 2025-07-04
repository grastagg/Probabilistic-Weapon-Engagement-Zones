#!/bin/bash

for SEED in {48..500}; do
  # for SEED in 1; do
  echo "Running seed $SEED"
  python learning_dubins_ez.py $SEED
done
