#!/bin/bash

for SEED in {31..31}; do
  echo "Running seed $SEED"
  python learning_dubins_ez.py $SEED
done
