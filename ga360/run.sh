#!/bin/bash
set -exuo pipefail
kaggle competitions download -c ga-customer-revenue-prediction -f train_v2.csv
kaggle competitions download -c ga-customer-revenue-prediction -f test_v2.csv

python ga360.py

Rscript --no-save --no-restore -e 'renv::restore()'
