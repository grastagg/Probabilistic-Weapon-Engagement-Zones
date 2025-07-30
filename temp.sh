#!/bin/bash

# Loop through numbers 0 to 300
for i in {0..500}; do
  # Check if the file exists
  if [ -f "${i}_results.json" ]; then
    :
  else
    echo "${i}_results.json is MISSING"
  fi
done
