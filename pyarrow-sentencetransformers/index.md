---
categories:
- python
- nlp
date: '2022-07-15T20:41:35+10:00'
image: /images/sentence_transformers_pyarrow.png
title: Training SentenceTransformers Using Memory Mapping with PyArrow
---

[SentenceTransformers](https://www.sbert.net/) provides a convenient interface for training linguistic embeddings using Transformers, which can be used for example with approximate nearest neighbours for search.
However it's not obvious how to train with datasets larger than memory; for example I'm not sure how to use it with Huggingface datasets.
In fact it's not too difficult using the flexibility of PyTorch's datasets and PyArrow's memory mapping mechanism to train it out of core and avoid an Out of Memory error.

First you will need to rewrite the data into an arrow file on disk to perform memory mapping.
If it's Parquet, ORC, or CSV files it can be read in as a [PyArrow Dataset](https://arrow.apache.org/docs/python/dataset.html) (even from a remote [filesystem](https://arrow.apache.org/docs/python/filesystems.html) such as S3 or HDFS), and then written to a local file.
By using the `to_batches` we can do this without ever having to store the whole dataset in memory; we just need enough room to store it on disk.

```python
import pyarrow as pa
from tqdm.auto import tqdm

def write_arrow(dataset: pa.dataset.Dataset,
                dest_path: str,
                filter=None,
                show_progress=True,
                **kwargs):
    batches = dataset.to_batches(filter=filter, **kwargs)
    with pa.OSFile(dest_path, 'wb') as sink:
        with tqdm(total=dataset.count_rows(filter=filter) if show_progress else None,
                  disable=not show_progress) as pbar:
            # Get the first batch to read the schema
            batch = next(batches)
            with pa.ipc.new_file(sink, schema=batch.schema) as writer:
                writer.write(batch)
                pbar.update(len(batch))
                for batch in batches:
                    writer.write(batch)
                    pbar.update(len(batch))
```

Once the data has been written like this we can memory map it, following the [PyArrow Cookbook](https://arrow.apache.org/cookbook/py/io.html#memory-mapping-arrow-arrays-from-disk).
This completes almost instantly because it doesn't actually load the data.

```python
with pa.memory_map(train_arrow_path, 'r') as source:
    tbl_train = pa.ipc.open_file(source).read_all()
```

SentenceTransformers expects the data as a Dataset of `InputExample`. 
A Dataset is just any Python class that has a `len` and `getitem` methods, so we can fetch the data from the PyArrow Table by index, and convert it into an `InputExample`.

```python
from sentence_transformers import SentenceTransformer, InputExample

class InputDataset:
    def __init__(self, left, right, outcome):
        self.left = left
        self.right = right
        self.outcome = outcome
        
    def __len__(self):
        return len(self.outcome)
    
    def __getitem__(self, idx):
        return InputExample(texts=[self.left[idx].as_py(),
                            self.right[idx].as_py()],
                            label=float(self.outcome[idx].as_py()))
                            
train_examples = InputDataset(
             tbl_train.column('sentence1'),
             tbl_train.column('sentence2'),
             tbl_train.column('label'),
             )
             
```

We can then pass this into a DataLoader and train a SentenceTransformers model [as per the SentenceTransformers documentation](https://www.sbert.net/docs/training/overview.html).
It's worth comparing the performance with loading a subset into memory; in my case it made no difference because the GPU was the bottleneck.

```python
from torch.utils.data import DataLoader, RandomSampler

train_dataloader = DataLoader(train_examples,
                              batch_size=batch_size,
                              sampler=RandomSampler(train_examples,
                                                    num_samples=num_samples))
                              
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=3,
          warmup_steps=0.1*num_samples,
          use_amp=True)
```

Evaluation is a bit more prickly.
The built-in SentenceTransformers [evaluation module](https://www.sbert.net/docs/package_reference/evaluation.html) takes the data in as lists to only ever encode the data once, and writes the output to a CSV.
If you've got more data than fits in memory you may want to write your own incremental evaluation routine.
If there is a lot of time spent recomputing embeddings it could be worth caching the results, for example using [diskcache](https://grantjenks.com/docs/diskcache/).

As I start to use SentenceTransformers more I'm finding some limitations of it's inbuilt training loop (e.g. it doesn't support gradient accumulation and so for large models I have to use really small batches which leads to poorer training with the standard parameters).
But this is a good way to get started with really large datasets, and abstracts well to other cases, all thanks to the simplicity and flexibility of PyTorch's concept of a Dataset.