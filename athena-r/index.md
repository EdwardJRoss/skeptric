---
date: 2019-08-02 10:39:06+10:00
description: ''
draft: true
featured_image: ''
title: Connecting to Amazon Athena in R from Ubuntu with ODBC
---

Alternatives: JDBC, RPresto

ODBC Driver: https://docs.aws.amazon.com/athena/latest/ug/connect-with-odbc.html
Docs: https://s3.amazonaws.com/athena-downloads/drivers/ODBC/Simba+Athena+ODBC+Install+and+Configuration+Guide.pdf


sudo apt-get install alien
sudo alien -i *.rpm


* How to find S3OutputLocation, AwsRegion
* Options for Authentication

```
dp <- DBI::dbConnect(
                odbc::odbc(),
                driver='/opt/simba/athenaodbc/lib/64/libathenaodbc_sb64.so',
                Schema="jobs",
                AwsRegion = "ap-southeast-2",
                AuthenticationType = "Default Credentials",
                S3OutputLocation = 's3://aws-athena-query-results-529344562275-ap-southeast-2/',
                StringColumnLength = 25000
            )
```
