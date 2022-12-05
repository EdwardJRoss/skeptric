---
categories:
- pandas
- aws
- python
date: '2021-05-28T19:36:34+10:00'
image: /images/s3fs.png
title: Writing Pandas Dataframes to S3
---

Writing a Pandas (or Dask) dataframe to Amazon S3, or Google Cloud Storage, all you need to do is pass an S3 or GCS path to a serialisation function, e.g.

```python
# df is a pandas dataframe
df.to_csv(f's3://{bucket}/{key}')
```

Under the hood Pandas uses [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) which lets you work easily with remote filesystems, and abstracts over [s3fs](https://s3fs.readthedocs.io/) for Amazon S3 and [gcfs](https://gcsfs.readthedocs.io) for Google Cloud Storage (and other backends such as (S)FTP, SSH or HDFS).
In particular s3fs is very handy for doing simple file operations in S3 because boto is often quite [subtly complex](https://stackoverflow.com/questions/54314563/how-to-get-more-than-1000-objects-from-s3-by-using-list-objects-v2) to use.

Sometimes managing access credentials can be difficult, [s3fs uses botocore credentials](https://s3fs.readthedocs.io/en/latest/#credentials), trying first environment variables, then configuration files, then IAM metadata.
But you can also specify an AWS Profile manually, and you can pass this (and [other arguments](https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem)) through pandas using the `storage_options` keyword argument:

```python
# df is a pandas dataframe
df.to_parquet(f's3://{bucket}/{key}', storage_options={'profile': aws_profile}))
```

One useful alternative is to create AWS Athena tables over the dataframes, so you can access them with SQL.
The fastest way to do this is with [AWS Data Wrangler](https://aws-data-wrangler.readthedocs.io/en/stable/tutorials/006%20-%20Amazon%20Athena.html), although [PyAthena](https://github.com/laughingman7743/PyAthena) is also a good option.