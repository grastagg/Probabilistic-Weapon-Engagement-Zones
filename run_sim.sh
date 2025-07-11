#!/bin/bash

for SEED in 40 64 68 333 433 381 348 106 46 330 384 336 93; do
  python learning_dubins_ez.py $SEED
done
