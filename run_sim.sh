#!/bin/bash

# for SEED in {2..500}; do
for SEED in 229; do
  python learning_dubins_ez.py $SEED
done
