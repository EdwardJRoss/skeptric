#!/bin/bash

awk '/^```{.yaml filename="environment.yml"}/{in_code=1} in_code {print $0} (in_code && /^``` *$/) {exit 0}' index.qmd | grep -v '```' > environment.yml


awk '/^```{.sh}/{in_code=1} in_code {print $0} (in_code && /^``` *$/) {exit 0}' index.qmd | grep -v '```' > setup-environment.sh


awk '/^```{.python filename="test_gpu.py"}/{in_code=1} in_code {print $0} (in_code && /^``` *$/) {exit 0}' index.qmd | grep -v '```' > test_gpu.py

awk '/^```{.python filename="test_train.py"}/{in_code=1} in_code {print $0} (in_code && /^``` *$/) {exit 0}' index.qmd | grep -v '```' > test_train.py
