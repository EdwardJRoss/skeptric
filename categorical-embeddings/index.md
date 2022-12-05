---
categories:
- data
date: '2021-09-10T20:00:00+10:00'
image: /images/embedding.png
title: Building Categorical Embeddings
---

High cardinality categorical data are tricky for machine learning models to deal with.
A linear model tries to estimate a different coefficient for every category, treating them as totally independent.
There are tools like hierarchical models that can encode some cross-correlations but (to my knowledge) they don't scale well to large datasets.
A tree model will try to estimate different coefficients on groups of categories (based on the order they are sorted in), but for many categories there is no canonical order (like locations on a map).
[Embeddings](/embeddings) are a useful way to make efficient use of this information.

The idea is to turn the category into a vector representing the data using other information about it.
This lets you pool information between categories that are similar in some sense.
In neural networks this is how categorical data is typically represented (often initialised with random vectors) and then the embeddings are fit via back-propagation.
However if you can separately generate some embeddings you can effectively use them with linear or logistic regression, or have them as a more effective initialisation in a neural network model.

# Creating embeddings

The idea of embeddings is to represent a categorical feature by some vector that captures that information to reuse in another setting.
The information used can create different embeddings, and potentially complementary embeddings could be combined.

One example is [behavioural embeddings](/embed-behaviour), where you take user behaviour to represent the items.
For example items that are purchased by the same people, or in the same basket, are similar.
One approach I've successfully used is to create the item-user co-purchase matrix, and then reduce it by a Truncated Singular Value Decomposition.
Then for each category the embedding is the *average* over the items for that category.

Another similar kind of co-occurrence matrix is a term-frequency matrix.
If there are terms or features that are associated with each item in the category you can build a category-term frequency matrix, and transform it for example with TF-IDF.

You can also use embeddings based on rich data associated with the items.
For example if there are textual descriptions you can create embeddings from a language model, or if there are images you can create embeddings from an image model.
This lets you reuse other existing models that may be fine-tuned to specific applications.
To get it back to a category level you can average over all the items in the category.
If they're normalised to be on the unit sphere you can rebuild that normalisation by normalising the average (since the centre of items on a sphere is [the projection of their Euclidean centre](/centroid-spherical-polygon); this lets you find items that are close to categories using an appropriate metric.

For an open categorical variable there needs to be a way to impute an embedding for unseen categories.
One option is to calculate the mean of all the other category embeddings, perhaps weighted by frequency.
Another approach would be to explicitly keep an "other" category for categories with few items, and build a specific embedding for those.

# Evaluating Embeddings

We can think about embeddings as enabling *pooling* information between similar categories.
Consequently they will have the most advantage where the information is sparse and there are some categories that are more similar than others.
I've found binary classification problems to be a fruitful testing ground, where each data point only contains a single bit of information, and so pooling can be very useful.

You can evaluate this using [binary cross entropy](/binary-rms) (also called logistic loss, or log loss), which is a measure of how likely the data is given the model.
However where there's sufficient data you can compare the percentage predicted for the category with the actual percentage of positive cases in the category (just keep in mind the [standard deviation of the binomial](/bernoulli-binomial) is $$\sqrt{\frac{p(1-p)}{n}$$, which bounds how accurately you can evaluate the percentage from the data).
In any case it's useful to compare uplift compared with the [constant model](/constant-models) of predicting the overall average probability (i.e. number of positive cases divided by total number of cases) for every category, and a One Hot Encoding of the categorical variable (with appropriate treatment of uncommon categories, and appropriate use of any available hierarchy information).

For running these evaluations I've found Logistic Regression with a L2 regularisation (i.e. logistic ridge regression) works quite well.
In scikit learn this is the default for [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and you just need to tune the regularisation parameter C (you can do this automatically with cross-validation using [`LogisticRegressionCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)), and it can handle large sparse term-frequency matrices using [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
Regularisation allows use of large dimensional embedding matrices without over-fitting.


# Open questions

I've found these methods quite effective to actually solve real information problems on a single categorical variable.
However this is just the beginning; how do we combine multiple embeddings for one variable, or multiple variables?

For multiple embeddings for one variable you could, in theory, just concatenate the embeddings together.
However I have found this doesn't always work better with regularised regression and I'm not exactly sure why.
One potential issue is different scales of coefficients, which could be addressed by weighting or standardisation.
Other options would be to ensemble the separate logistic regression models, or to try different types of models.

There are even more options to combine multiple variables.
They can just be added to the embeddings as extra variables in the regression, with appropriate preprocessing, and the model re-fit.
However if there are interactions you would need to multiply each embedding vector with the other vectors, which could quickly get quite large, and at this point it may be worth considering another model.
Similarly one could combine embeddings of other categorical variables, with the same caveats about interactions.
Neural networks could be a very strong candidate for these problems as they can build complex interactions between the variables, and even fine tune the embeddings themselves.
Taking this to the extreme the embedding tasks could be combined in a single multi-task setting, which could potentially mine the relevant information more effectively (but is *much* more complex).

Another approach is joint embeddings between two categories.
Suppose you have two different categories that both fit within a single embedding task.
One way to create a joint embedding would be to train embeddings separately and multiply each of the columns (so for a N dimensional embedding and an M dimensional embedding, we create an N*M dimensional joint embedding) to create an interaction; however this ignores the interaction structure.
Another approach would be to treat the pair of categories as a single categorical variable and build an embedding on it; but that loses the relationships between the categories separately.
There should be an approach midway between the two that appropriately estimates the marginal embeddings but incorporates information from the joint structure - but I don't know what that should look like (and I would look to neural network models for inspiration).


# Potential Case Studies

It would be nice if I had some examples to go with this.
Some potentially interesting historical Kaggle competitions to experiment on would be [Avito Demand Prediction](https://www.kaggle.com/c/avito-demand-prediction), [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) and [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction).
Also notable is the [Rossmann Store Sales Competition](https://www.kaggle.com/c/rossmann-store-sales) where third place was won by a [neural network model](https://github.com/entron/entity-embedding-rossmann).

Once you've built and embedding in a Pandas DataFrame indexed by the category name it can be wrapped in an sklearn transformer as below:

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Embedder(BaseEstimator, TransformerMixin):
    def __init__(self, embeddings: pd.DataFrame) -> None:
        self.embeddings = embeddings
        
        # Impute missing values with the mean
        # This could be extended to also handle a weight
        missing_vector = np.mean(embeddings, axis=0)
        
        self.embeddings_matrix = np.vstack([missing_vector, embeddings.to_numpy()])

        self.category_to_index = {v:k+1 for k,v in enumerate(embedings.index)}
        
    def fit(self, X: pd.Series, y=None):
        return self
    
    def transform(self, X: pd.Series) -> np.array:
        indices = X.map(self.category_to_index).fillna(0).astype('int64')
        return self.embeddings_matrix[indices]
```

This can then be used with a `ColumnTransformer` and `LogisticRegression` in an sklearn pipeline:

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
  ('columns', ColumnTransformer([
       ('embedding', Embedder(embeddings), 'category_column_name'),
       ])),
  ('classifier', LogisticRegression(C=1)),
])
```

I hope to build some open examples I can share in the future.