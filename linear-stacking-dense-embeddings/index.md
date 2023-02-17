---
date: 2023-02-18 08:00:00+11:00
image: siamese_network.png
title: Linear Stacking Cosine Embeddings
---

Dense embeddings of entities measured by cosine similarity are a widespread useful tool in machine learning.
In natural language processing they come up through procedures such as [matrix factorization](https://arxiv.org/abs/1003.1141), [word2vec](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf), [GloVe](https://nlp.stanford.edu/projects/glove/), or [from transformer models](https://aclanthology.org/2020.acl-main.431/) where they represent similarity of words by types of co-occurance.
These also occur at a sentence or document level through [SentenceTransformers](https://www.sbert.net/).
These techniques are [related to implicit matrix factorization](https://proceedings.neurips.cc/paper/2014/file/feab05aa91085b7a8012516bc3533958-Paper.pdf) which is a common techniques in [implicit recommendations](https://implicit.readthedocs.io/en/latest/) where it is known as collaborative filtering.
Embeddings also [occur in graph settings](https://arxiv.org/abs/1709.05584) which subsumes local context word methods (a graph where words that occur together are connected) and collaborative filtering (a graph where users are connected to items).
They are also very useful for information retrieval because they are amenable to Approximate Nearest Neighbours methods (through [Locality-sensitive-hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing); specifically [random projections](https://en.wikipedia.org/wiki/Random_projection)) and are used in IR systems such as [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) and [ColBERT](https://arxiv.org/abs/2004.12832) and in many industrial settings such as [Facebook search](https://arxiv.org/pdf/2006.11632.pdf).

In many cases we can have multiple embeddings, either representing different features (such as a description and a location) or representing different aspects of a feature (such as GloVe embeddings and embeddings from a Truncated SVD of PMI matrix of local co-occurance).
There's a simple way to combine them via a weighted combination to obtain a new cosine embedding, which can again be used for efficient search.
The underlying mathematics is in Section 6.2 of [Embedding-based Retrieval in Facebook Search](https://arxiv.org/pdf/2006.11632.pdf).
If we have training dataset of `X_train` and `y_train` we can learn the optimal feature weights through Linear Regression in Python.

We can represent each embedding as a Pandas DataFrame (or numpy array) containing the vector representation of each row.
Then the models can be stored as a dictionary from their name to the embeddings.
We can calculate the cosine similarities with `X_train` for each model, and then optimise it to maximise similarity to `y_train`.
A simple way to do this is with Linear Regression.


```python
from sklearn.linear import LinearRegression
import numpy as np
import pandas as pd

def get_stack_weights(models: dict[str, pd.DataFrame],
                      X_train, y_train) -> dict[str, float]:
    X = np.array([cosine_predictions(X_train, v) for v in models.values()]).T

    lr = LinearRegression()
    lr.fit(X, y_train)
    return {model: coef for model, coef in zip(models, lr.coef_)}
```

The cosine predictions are the dot product once we've normalised them.

```python
def row_normalise(df):
    """l2 normalise each row of df"""
    row_norms = np.sqrt((np.array(df)**2).sum(axis=1))
    return df / row_norms[:, None]

def cosine_predictions(X1, X2):
    """Return cosine distances of pairs in rel_df using vsm_df"""
    return (row_normalise(X1) * row_normalise(X2)).sum(axis=1)
```

Then we can combine the models using these weights by using a weighted concatenation.
We would want to discard any models with a low weight, set here with a threshold.

```python
def combine_models(models: dict[str, pd.DataFrame],
                    weights: dict[str, float],
                    threshold=0.05) -> pd.DataFrame:
    """Combine vector models using weights

    Any model with a weight < threshold * max(weight) is discarded.
    """
    max_weight = max(weights.values())
    assert max_weight > 0, "At least one weight should be positive"

    vocab = sorted(get_common_vocab(models.values()))

    result = []

    for model_name in models:
        weight = weights[model_name] / max_weight

        if weight > threshold:
            # Weight by sqrt(weight) since it appears on both sides of cosine product
            # This is for a symmetric model
            result.append(np.sqrt(weight) * row_normalise(models[model_name].loc[vocab]))
    return pd.concat(result, axis=1)
```

This is just the start of this technique, some natural extensions are:

* penalised regression if there are a lot of models
* [feature weighted linear stacking](https://arxiv.org/abs/0911.0460)
* learning embeddings end-to-end [as in TaoBao Search](https://arxiv.org/pdf/2106.09297.pdf)

But this is a surprisingly effective baseline.
