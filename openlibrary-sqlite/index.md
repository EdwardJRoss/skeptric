---
categories:
- nlp
- ner
- hnbooks
- python
date: '2022-07-02T21:35:01+10:00'
image: /images/open_library_sqlite.png
title: Importing Open Library into SQLite
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline), and want to link the books to records from [Open Library](/open-library).
The Open Library data dumps are several gigabytes of compressed TSV, so are too big to fit in memory on a standard machine.
The [Libraries Hacked repository](https://github.com/LibrariesHacked/openlibrary-search) imports it into a PostgreSQL database, but I don't want to set up a Postgres database for an analysis job if I can avoid it.
Instead I want to use an SQLite database to store the data.

The SQLite CLI has an [import statement](https://www.sqlite.org/cli.html#importing_files_as_csv_or_other_formats), but when I tried to run it with `.mode tabs` it kept complaining about unescaped quotes and wrong number of fields.
Instead I wrote a little Python script to iterate over all the data and yield tuples of the 5 columns.


```python
from pathlib import Path
import gzip

ol_dump_date = '2022-06-06'
data_path = Path('../data/01_raw')

def ol_path(segment):
    return data_path / f'ol_dump_{segment}_{ol_dump_date}.txt.gz'

def ol_data(segment):
    with gzip.open(ol_path(segment), 'rt') as f:
        for line in f:
            yield tuple(line.split('\t', 5))
```


It turns out it's a bit slower inserting one row at a time, so we'll make [minibatches](/python-minibatching) from the sequence to write.
If the batch size is too large it uses a lot of memory, if it's very small then it's slower due to the overhead of parsing the statement.
In practice 1000 seems like a good tradeoff.

```python
def minibatch(seq, size):
    items = []
    for x in seq:
        items.append(x)
        if len(items) >= size:
            yield items
            items = []
    if items:
        yield items
```

We can then create the table and insert the rows a minibatch at a time.
Wrapping this in a transaction makes it slightly faster.
It would be reasonable to put a `PRIMARY KEY` constraint on the `key` column; but in this use case it won't make much difference if the incoming data is valid.

```python
from tqdm.auto import tqdm

def create_segment(cur, segment, batch_size=10_000):
    cur.execute(f'CREATE TABLE {segment} (type TEXT, key TEXT, revision INT, last_modified TEXT, json TEXT);')
    
    with con:
        for batch in minibatch(tqdm(ol_data(segment)), batch_size):
            con.executemany('INSERT INTO authors VALUES (?,?,?,?,?)', batch)
```

Finally we actually create all the tables.
We set some `PRAGMA` to make bulk inserts faster (as per [this blog](http://blog.quibb.org/2010/08/fast-bulk-inserts-into-sqlite/) and [this Stackoverflow](https://stackoverflow.com/questions/1711631/improve-insert-per-second-performance-of-sqlite)), and then create each segment.
This completes in about 25 minutes on my laptop, leaving a 62GB SQLite file (since we now have all that uncompressed JSON).

```python
con = sqlite3.connect('openlibrary.sqlite')

con.execute("PRAGMA synchronous=OFF")
con.execute("PRAGMA count_changes=OFF")
con.execute("PRAGMA journal_mode=MEMORY")
con.execute("PRAGMA temp_store=MEMORY")

for segment in ['authors', 'works', 'editions']:
    create_segment(con, segment)
```

This is just a start; we'll need to do some more work to make this usable for looking up entries such as extracting names and adding indexes.

With SQLite's [json support](https://www.sqlite.org/json1.html) we can now query the json column.
For example to get all the authors of the book `Regression and Other Stories` we can run:


```sql
SELECT works.key,
       authors.key,
       json_extract(authors.json, '$.name') as author_name
FROM works
JOIN json_each((
                SELECT json
                FROM works as w
                WHERE w.key = works.key
              ), '$.authors') as work_authors
JOIN authors ON authors.key = json_extract(work_authors.value,
                                           '$.author.key')
WHERE json_extract(works.json, '$.title') = 'Regression and Other Stories';
```

This is pretty slow; it takes around 4 minutes to return the 3 authors.
To make this more usable we're going to need to create indexes, and likely change the tables.