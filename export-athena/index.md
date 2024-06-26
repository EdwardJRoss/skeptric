---
date: 2020-03-16 21:26:57+11:00
image: /images/athena.png
title: Exporting data to Python with Amazon Athena
---

One necessary hurdle in doing data analysis or machine learning is loading the data.
In many businesses larger datasets live in databases, in an object store (like Amazon S3) or the Hadoop File System.
For some use cases you can do the work where the data lives using SQL or Spark, but sometimes it's more convenient to load it into a language like Python (or R) with a wider range of tools.

[Presto](https://prestodb.io/), and Amazon's managed version [Athena](https://aws.amazon.com/athena/) are very powerful tools for preparing and exporting data.
They can query data accross data files directly in S3 (and HDFS for Presto) and many common databases via [Presto connectors](https://prestodb.io/docs/current/connector.html) or [Athena's federated queries](https://aws.amazon.com/blogs/big-data/query-any-data-source-with-amazon-athenas-new-federated-query/).
They've got a very powerful query language and can process large volumes of data quickly in memory accross a cluster of commodity machines.
For this reason many tech companies like Facebook, Uber and Netflix use Presto/Athena as a core way to access their data platform.

The most effective workflow I've found for exporting data from Athena or Presto into Python is:

* Writing SQL to filter and transform the data into what you want to load into Python
* Wrapping the SQL into a [Create Table As Statement (CTAS)](https://docs.aws.amazon.com/athena/latest/ug/create-table-as.html) to export the data to S3 as Avro, Parquet or JSON lines files.
* Reading the data into memory using fastavro, pyarrow or Python's JSON library; optionally using Pandas.

This is very robust and for large data files is a very quick way to export the data.
I will focus on Athena but most of it will apply to Presto using [presto-python-client](https://github.com/prestodb/presto-python-client) with some minor changes to DDLs and authentication.

There is another way, directly reading the output of an Athena query as a CSV from S3, but there are some limitations.

I have a [sample implementation](https://gist.github.com/EdwardJRoss/66561eb91049d9838db71403bd07c950) showing how to query avro with `query_avro` and using the CSV trick with `query`.

Note that since this article was originally written Athena has added an [unload](https://docs.aws.amazon.com/athena/latest/ug/unload.html) command for exporting a query result as a file type, and [AWS Data Wrangler](https://github.com/awslabs/aws-data-wrangler) now has convenient wrappers for quickly exporting data from Athena by using a CTAS or unload query in the background.

# Athena Fast Export workflow

[PyAthena](https://github.com/laughingman7743/PyAthena) is a good library for accessing Amazon Athena, and works seamlessly once you've [configured the credentials](https://github.com/laughingman7743/PyAthena#credentials).
However the `fetch` method of the default database cursor is very slow for large datasets (from around 10MB up).
Instead it's much faster to export the data to S3 and then download it into python directly.
I am focus on Athena for this example, but the same method applies to Presto using ) with a few small changes to the queries.

The final method looks like this:

    def download_table(cursor, outfolder, query, format='AVRO'):
    """Use PyAthena cursor to download query to outfolder in format

    Note that all columns in query must be named for this to work
    Multiple files may be created in outfolder.
    """
        table = temp_table_name()
        try:
            create_table_as(cursor, table, query, format)
            s3_locations = table_file_location(cursor, table)
            download_s3(s3_locations, outpath)
        finally:
            drop_table(cursor, table)

The input query in a [CTAS](https://docs.aws.amazon.com/athena/latest/ug/create-table-as.html) to change the output format.

    def create_table_as(cursor, table, query, format='AVRO'):
        cursor.execute(f"CREATE TABLE {table} WITH (format = '{format}') as {query}")

The location of the output tables can be obtained with "\$path":

    def table_file_location(cursor, table):
       cursor.execute(f'SELECT DISTINCT "$path" FROM {table}')
       return [row[0] for row in cursor.fetchall()]

The output objects can be downloaded from S3 using boto3 (depending on your configuration you may need to pass additional connection arguments):

    from pathlib import Path
    from boto3.session import Session
    def download_s3(s3_paths, outpath):
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)
        client = Session().client('s3')
        for s3_path in s3_paths:
            bucket_name, bucket_key = get_bucket_key(s3_path)
            filename = bucket_key.split('/')[-1]
            client.download_file(bucket_name, bucket_key, str(outpath / filename))

Finally the table can be dropped; we use `IF EXISTS` so the function completes even if something goes wrong.

    def drop_table(cursor, table):
        cursor.execute(f'DROP TABLE {table} IF EXISTS')
        # Optionally remove underlying S3 files here

The individual files can then be read in with `fastavro` for Avro, `pyarrow` for Parquet or `json` for JSON.

Note that because it can be spread accross files, any sorting from the query may be lost unless you merge sort the input.

The full details (streaming instead of downloading) are available in the [sample implementation](https://gist.github.com/EdwardJRoss/66561eb91049d9838db71403bd07c950).

## Optimisations

There's a lot that could be done to make this faster or more convenient:

* The queries could be executed without blocking using the [AsynchronousCursor](https://github.com/laughingman7743/PyAthena#asynchronouscursor)
* S3 files could be downloaded in parallel, which may be faster
* The files don't need to be directly downloaded when parsing a S3 path to Pandas or using [s3fs](https://github.com/dask/s3fs) (this is usually slower)
* The files could be concatenated together into a single outfile

## Choosing an export format

| Format | Python | Pandas | Datatypes | Storage Type | CLI Tool |
| ------ | -------|------  | --------- | ------------ | -------  |
| [Avro](https://avro.apache.org/)   |  fastavro        |      With [pandavro](https://github.com/ynqa/pandavro) |  All |  Row       |  [avro-tools](https://www.michael-noll.com/blog/2013/03/17/reading-and-writing-avro-files-from-the-command-line/) |
| [JSON lines](http://jsonlines.org/)   |  json (builtin) | [`read_json(..., lines=True)`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html)  | Not binary, decimal, date, timestamp |   Row       | [jq](https://stedolan.github.io/jq/) or cat |
| [Parquet](https://parquet.apache.org/)   |  pyarrow        |      [`read_parquet(..., engine='pyarrow')`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html)  | All* |   Column       |  [parquet-tools](https://github.com/apache/parquet-mr/tree/master/parquet-tools) |
| [ORC](https://orc.apache.org/)   |  N/A        |      No  |  All |  Column       |  [orc-tools](https://orc.apache.org/docs/java-tools.html) |
| TEXTFILE |  N/A        |      No        | String |   Row       | cat -vt  |

Avro can represent almost all Athena/Presto datatypes (except Map) and has excellent support through fastavro.
The only major drawback is that it doesn't have native pandas support, but is very easy to convert.

JSON format is also a good choice as it can represent nested structures and all the basic types (strings, integers, double precision floats, boolean and nulls).
It won't preserve the types of some of the more complex datatypes like timestamps, and can't handle binary data.

Parquet can represent preserve all the datatypes, and as a column store is efficient for both Presto/Athena and Pandas.
Unfortunately pyarrow can't handle lists of structs as raised in [ARROW-1644](https://issues.apache.org/jira/browse/ARROW-1644) (though it's currently being worked on!)
Until this happens you can't read and write arbitrary data from Python (don't use `fastparquet`, it considers *silently* replacing nested structures with nulls a [feature]((https://github.com/dask/fastparquet/issues/443)), but is fine for simpler data structures (you can usually [unnest](https://prestodb.io/docs/current/sql/select.html#unnest) and destructure the data if you need to).
ORC is even less well supported in Python.


## The problem with TEXTFILE

TEXTFILE is a text delimited format, *similar* to a CSV.

As per the [documentation](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-StorageFormatsStorageFormatsRowFormat,StorageFormat,andSerDe) the rows are separated by new lines `\n`, and the fields are delimited by a separator, by default the [Start of Heading character](https://en.wikipedia.org/wiki/C0_and_C1_control_codes#SOH) `\001` (and strangely not the [Record Separator](https://en.wikipedia.org/wiki/C0_and_C1_control_codes#RS)).
The record separator could be specified to be a ',' (using properties in Presto or the [`field_delimiter`](https://docs.aws.amazon.com/athena/latest/ug/create-table-as.html) in Athena), and in many cases this will read or write a CSV.

There's a mechanism for escaping characters (so a newline in a field can be written `\n`, and a backslash as `\\`) and a special character for NULLs (`\N`), but there's no method for escaping (or quoting) the field separator!

So for example the following query in Athena:

    create table sandbox.test_textfile with (format='TEXTFILE', delimited=',') as select ',' as a, ',,' as b

leads to an output file (which you can find with `select distinct "$path" from sandbox.test_textfile`)

    ,,,,\n

It's impossible to tell if it's meant to represent (",", ",,") or (",,", ",").
If I try to select back from that table the rows are reported to be the empty string!

This explains why the default separator is `\001`, because it's unlikely to occur in a field.
But if it ever does it will cause hours of headaches to understand why the data is corrupted.

Moreover this type of format with backslash escapes and special null delimiters is uncommon and unless you're using the Java Hadoop libraries you'll probably have to write your own parser.
It's a pity they don't support [RFC-4180 CSVs](https://tools.ietf.org/html/rfc4180), but admittedly they have no way of dealing with missing values (nulls) or data types which makes them more limited.



# One weird S3 CSV trick

Athena will output the result of every query as a CSV on S3.
Interestingly this is a proper fully quoted CSV (unlike TEXTFILE).
It turns out to be much quicker to read this CSV directly than to iterate over the rows, and this is implemented in [Pyathena Pandas Cursor](https://github.com/laughingman7743/PyAthena#pandascursor) - although there's nothing Pandas specific about it!

Unlike using Avro complex fields (like arrays and structs) will come through as strings (which can mostly be JSON parsed), and binary will come as a series of hex digits.
NULLs will be represented by an unquoted field, which can't be distinguished from an empty string by Pythons csv reader or by Pandas `read_csv`; you could roll your own, but a worse-is-better solution would be to treat empty fields as NULL.

While this is convenient and has advantages like preserving sorting and working for any valid query, it is uncompressed and so transfer may be much slower.

See the function `query` in the [sample implementation](https://gist.github.com/EdwardJRoss/66561eb91049d9838db71403bd07c950) for details.
