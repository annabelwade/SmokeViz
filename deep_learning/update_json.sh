#!/bin/bash

for i in {1..5}; do
    cp exp${i}.json exp${i}.1.json
    sed -i 's/"lr": 1e-2/"lr": 1e-3/g' exp${i}.1.json
done
