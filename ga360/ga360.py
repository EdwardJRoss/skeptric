#!/usr/bin/env python
import json
import csv
from io import TextIOWrapper
from zipfile import ZipFile
import sys

# Allow really long lines
csv.field_size_limit(sys.maxsize)

fout = open('ga360.csv', 'w')
csvout = csv.writer(fout)

csvout.writerow(['user', 'start_time', 'visit_time', 'pages', 'revenue'])

for path in ['train_v2.csv.zip', 'test_v2.csv.zip']:
    with ZipFile(path) as z:
        with z.open(path[:-4], mode="r") as f:
            csvin = csv.reader(TextIOWrapper(f, 'utf-8'))
            header = next(csvin)

            total_idx = header.index('totals')
            time_idx = header.index('visitStartTime')
            user_idx = header.index('fullVisitorId')

            for row in csvin:
                totals = json.loads(row[total_idx])
                csvout.writerow([
                    row[user_idx],
                    row[time_idx],
                    totals.get('timeOnSite', '0'),
                    totals.get('pageviews', '0'),
                    totals.get('totalTransactionRevenue', '0')
                    ])
