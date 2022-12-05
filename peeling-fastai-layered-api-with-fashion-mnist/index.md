---
categories:
- python
- data
- fastai
date: '2022-05-23T19:05:13+10:00'
image: /images/fashion_mnist_training_loop.jpg
title: Peeling back the fastai layered AI with Fashion MNIST
---

[Chapter 4 of the fastai book](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) covers how to build a Neural Network for distinguishing 3s and 7s on MNIST from scratch.
We're going to do a similar thing but instead of building the neural network from the ground up we're going to use fastai's *layered API* to build it top down.
We'll start with the high level API to train a dense neural network in a few lines.
Then we'll redo the problem going deeper and deeper into the API.
At the very core it's mainly PyTorch, and we'll have a pure PyTorch implementation like in the book.
Then we'll start rebuilding the abstractions from scratch to get a high level API like we started with.

Instead of using the MNIST digits we'll use [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), which contains little black and white images of different types of clothing.
This is a bit harder and a convolutional neural network would perform better here (as demonstrated in [v3 of fastai course](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist.ipynb)).
But to keep things simple we'll use a dense neural network.

This post was generated with a Jupyter notebook. You can also [view this notebook on Kaggle](https://www.kaggle.com/code/edwardjross/peeling-fastai-layered-api-with-fashion-mnist/notebook) or [download the Jupyter notebook](https://nbviewer.org/github/EdwardJRoss/skeptric/blob/master/static/notebooks/fashion-mnist-with-prototype-methods.ipynb).

# Training a model in 5 lines of code

We train a model to recognise these items of clothing from scratch in just 6 lines using fastai's high level API.
It should take a minute or two to run on a CPU (for such a small model and data there is marginal benefit running on a GPU).


```python
# 1. Import
from fastai.tabular.all import *
# 2. Data
df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv', dtype={'label':'category'})
# 3. Dataloader
dls = TabularDataLoaders.from_df(df, y_names='label', bs=4096, procs=[Normalize])
# 4. Learner
learn = tabular_learner(dls, layers=[100], opt_func=SGD, metrics=accuracy, config=dict(use_bn=False, bn_cont=False))
# 5. Fit
learn.fit(40, lr=0.2)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.204540</td>
      <td>0.779914</td>
      <td>0.735917</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.917599</td>
      <td>0.632047</td>
      <td>0.772750</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.787847</td>
      <td>0.568341</td>
      <td>0.791167</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.702365</td>
      <td>0.522808</td>
      <td>0.811083</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.642320</td>
      <td>0.510324</td>
      <td>0.811167</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.603035</td>
      <td>0.491477</td>
      <td>0.822083</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.568987</td>
      <td>0.463034</td>
      <td>0.831250</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.538885</td>
      <td>0.449788</td>
      <td>0.835583</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.514483</td>
      <td>0.440531</td>
      <td>0.839333</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.495511</td>
      <td>0.436088</td>
      <td>0.840583</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.479304</td>
      <td>0.446928</td>
      <td>0.833500</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.467378</td>
      <td>0.419800</td>
      <td>0.846667</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.453736</td>
      <td>0.412330</td>
      <td>0.851833</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.442266</td>
      <td>0.409911</td>
      <td>0.851667</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.432882</td>
      <td>0.413216</td>
      <td>0.849833</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.426103</td>
      <td>0.408956</td>
      <td>0.852667</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.417694</td>
      <td>0.396635</td>
      <td>0.858083</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.409261</td>
      <td>0.394431</td>
      <td>0.856333</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.402036</td>
      <td>0.396497</td>
      <td>0.856500</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.396574</td>
      <td>0.393080</td>
      <td>0.859083</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.391744</td>
      <td>0.395087</td>
      <td>0.857667</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.388619</td>
      <td>0.405253</td>
      <td>0.852083</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.383373</td>
      <td>0.388774</td>
      <td>0.859667</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.379811</td>
      <td>0.391994</td>
      <td>0.856500</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.376847</td>
      <td>0.384142</td>
      <td>0.861167</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.372048</td>
      <td>0.376191</td>
      <td>0.864167</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.368737</td>
      <td>0.383891</td>
      <td>0.858917</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.364437</td>
      <td>0.380743</td>
      <td>0.862833</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.362288</td>
      <td>0.370025</td>
      <td>0.865500</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.359911</td>
      <td>0.370142</td>
      <td>0.867500</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.357278</td>
      <td>0.384656</td>
      <td>0.859667</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.355596</td>
      <td>0.364638</td>
      <td>0.869250</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.351848</td>
      <td>0.363354</td>
      <td>0.868833</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.347400</td>
      <td>0.362254</td>
      <td>0.869667</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.343837</td>
      <td>0.364124</td>
      <td>0.869500</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.340539</td>
      <td>0.362926</td>
      <td>0.869583</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.339874</td>
      <td>0.368416</td>
      <td>0.866333</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.336479</td>
      <td>0.369149</td>
      <td>0.866917</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.335846</td>
      <td>0.371058</td>
      <td>0.865833</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.336446</td>
      <td>0.383330</td>
      <td>0.858583</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


We can then test the perormance on the test set; note that it's very close to the accuracy from the last line of the training above.

>


```python
df_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv', dtype={'label': df.label.dtype})

probs, actuals = learn.get_preds(dl=dls.test_dl(df_test))

print(f'Accuracy on test set {float(accuracy(probs, actuals)): 0.2%}')
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    Accuracy on test set  85.88%


Looking at the [sklearn benchmarks](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#) on this dataset it's outperformed by some other models such as Support Vector Machines (SV) with 89.7% accuracy, and Gradient Boosted Trees with 88.8% accuracy.
In fact our model is almost the same as the MLPClassifier (87.7%).
See if you can beat this baseline by changing the layers, learning rate, and number of epochs.

The [*best* results](https://github.com/zalandoresearch/fashion-mnist#benchmark) on this dataset, around 92-96%, come from Convolutional Neural Networks (CNN).
The kind of approach we use here can be extended to a CNN; the other kinds of models are quite different.

## What did we just do?

Let's go back through those 5 lines slowly to see what was going on.

### 1. Import

The first line imports all the libraries we need for tabular analysis.

This includes specific fastai libraries, as well as general utilities such as Pandas, numpy and PyTorch, and much more


```python
from fastai.tabular.all import *
```

If you want to see exactly what was imported you can look into the module or the [source code](https://github.com/fastai/fastai/blob/master/fastai/tabular/all.py).


```python
import fastai.tabular.all
L(dir(fastai.tabular.all))
```




    (#846) ['APScoreBinary','APScoreMulti','AccumMetric','ActivationStats','Adam','AdaptiveAvgPool','AdaptiveConcatPool1d','AdaptiveConcatPool2d','ArrayBase','ArrayImage'...]



This includes standard imports like "pandas as pd"


```python
fastai.tabular.all.pd
```




    <module 'pandas' from '/opt/conda/lib/python3.7/site-packages/pandas/__init__.py'>



### 2. Data

We read in the data from Pandas as a CSV, letting Pandas know that the `label` column is categorical.


```python
df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv', dtype={'label':'category'})
```

The dataframe contains a `label` column giving the kind of image, and then 784 columns for the pixel value from 0-255.


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59995</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>73</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59997</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>160</td>
      <td>162</td>
      <td>163</td>
      <td>135</td>
      <td>94</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59998</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59999</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>60000 rows × 785 columns</p>
</div>



A histogram of the pixels shows they are mostly 0, with values up to 255.


```python
_ = plt.hist(df.filter(like='pixel', axis=1).to_numpy().reshape(-1))
```



![png](/post/peeling-fastai-layered-api-with-fashion-mnist/output_22_0.png)



From a singe row of the dataframe we can read the label, and the pixels


```python
label, *pixels = df.iloc[0]

label, len(pixels)
```




    ('2', 784)



The 784 pixels are actually 28 rows of the image, each containing 28 columns.
If we rearrange them we can plot it as an image.


```python
image_array = np.array(pixels).reshape(28, 28)
_ = plt.imshow(image_array, cmap='Greys')
```



![png](/post/peeling-fastai-layered-api-with-fashion-mnist/output_26_0.png)



All we are seeing here are the pixel intensities from 0 (white) to 255 (black) on a grid.


```python
fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(image_array, cmap="Greys")

for i in range(image_array.shape[0]):
    for j in range(image_array.shape[1]):
        text = ax.text(j, i, image_array[i, j], ha="center", va="center", color="magenta")
```



![png](/post/peeling-fastai-layered-api-with-fashion-mnist/output_28_0.png)



The labels are categorical codes for different types of clothing.



We can copy the label description and convert it into a Python dictionary.


```python
labels_txt = """
Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
""".strip()

labels = dict([row.split('\t') for row in labels_txt.split('\n')[1:]])
labels
```




    {'0': 'T-shirt/top',
     '1': 'Trouser',
     '2': 'Pullover',
     '3': 'Dress',
     '4': 'Coat',
     '5': 'Sandal',
     '6': 'Shirt',
     '7': 'Sneaker',
     '8': 'Bag',
     '9': 'Ankle boot'}



The image above is of a Pullover


```python
label, labels[label]
```




    ('2', 'Pullover')



We've got 6000 images of each type.


```python
df.label.map(labels).value_counts()
```




    T-shirt/top    6000
    Trouser        6000
    Pullover       6000
    Dress          6000
    Coat           6000
    Sandal         6000
    Shirt          6000
    Sneaker        6000
    Bag            6000
    Ankle boot     6000
    Name: label, dtype: int64



### 3. Dataloader

Now we have our raw data we need a way to pass that into the model in a way it understands.
We do this with a *DataLoader* reading from the dataframe.
We need to tell it:

* df: the dataframe to read from
* y_names: the name of the column containing the outcome variable, here `label`
* bs: the batch size, how many rows to feed to the model each time. We use 4096 because the data and models are small
* procs: any preprocessing steps to do, here we use `Normalize` to map them from 0-255 to a more reasonable range.
* cont_names: The name of the continuous columns

Note that before we didn't pass `cont_names` and it automatically detected them; however it can reorder the columns so we specify it here for clarity.


```python
dls = TabularDataLoaders.from_df(df, y_names='label', bs=4096, procs=[Normalize], cont_names=list(df.columns[1:]))
```

This data loader can then produce the pixel arrays for a subset of rows, and the outcome labels on demand.
Note these are the values *before* using procs.


```python
dls.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>pixel10</th>
      <th>pixel11</th>
      <th>pixel12</th>
      <th>pixel13</th>
      <th>pixel14</th>
      <th>pixel15</th>
      <th>pixel16</th>
      <th>pixel17</th>
      <th>pixel18</th>
      <th>pixel19</th>
      <th>pixel20</th>
      <th>pixel21</th>
      <th>pixel22</th>
      <th>pixel23</th>
      <th>pixel24</th>
      <th>pixel25</th>
      <th>pixel26</th>
      <th>pixel27</th>
      <th>pixel28</th>
      <th>pixel29</th>
      <th>pixel30</th>
      <th>pixel31</th>
      <th>pixel32</th>
      <th>pixel33</th>
      <th>pixel34</th>
      <th>pixel35</th>
      <th>pixel36</th>
      <th>pixel37</th>
      <th>pixel38</th>
      <th>pixel39</th>
      <th>pixel40</th>
      <th>pixel41</th>
      <th>pixel42</th>
      <th>pixel43</th>
      <th>pixel44</th>
      <th>pixel45</th>
      <th>pixel46</th>
      <th>pixel47</th>
      <th>pixel48</th>
      <th>pixel49</th>
      <th>pixel50</th>
      <th>pixel51</th>
      <th>pixel52</th>
      <th>pixel53</th>
      <th>pixel54</th>
      <th>pixel55</th>
      <th>pixel56</th>
      <th>pixel57</th>
      <th>pixel58</th>
      <th>pixel59</th>
      <th>pixel60</th>
      <th>pixel61</th>
      <th>pixel62</th>
      <th>pixel63</th>
      <th>pixel64</th>
      <th>pixel65</th>
      <th>pixel66</th>
      <th>pixel67</th>
      <th>pixel68</th>
      <th>pixel69</th>
      <th>pixel70</th>
      <th>pixel71</th>
      <th>pixel72</th>
      <th>pixel73</th>
      <th>pixel74</th>
      <th>pixel75</th>
      <th>pixel76</th>
      <th>pixel77</th>
      <th>pixel78</th>
      <th>pixel79</th>
      <th>pixel80</th>
      <th>pixel81</th>
      <th>pixel82</th>
      <th>pixel83</th>
      <th>pixel84</th>
      <th>pixel85</th>
      <th>pixel86</th>
      <th>pixel87</th>
      <th>pixel88</th>
      <th>pixel89</th>
      <th>pixel90</th>
      <th>pixel91</th>
      <th>pixel92</th>
      <th>pixel93</th>
      <th>pixel94</th>
      <th>pixel95</th>
      <th>pixel96</th>
      <th>pixel97</th>
      <th>pixel98</th>
      <th>pixel99</th>
      <th>pixel100</th>
      <th>pixel101</th>
      <th>pixel102</th>
      <th>pixel103</th>
      <th>pixel104</th>
      <th>pixel105</th>
      <th>pixel106</th>
      <th>pixel107</th>
      <th>pixel108</th>
      <th>pixel109</th>
      <th>pixel110</th>
      <th>pixel111</th>
      <th>pixel112</th>
      <th>pixel113</th>
      <th>pixel114</th>
      <th>pixel115</th>
      <th>pixel116</th>
      <th>pixel117</th>
      <th>pixel118</th>
      <th>pixel119</th>
      <th>pixel120</th>
      <th>pixel121</th>
      <th>pixel122</th>
      <th>pixel123</th>
      <th>pixel124</th>
      <th>pixel125</th>
      <th>pixel126</th>
      <th>pixel127</th>
      <th>pixel128</th>
      <th>pixel129</th>
      <th>pixel130</th>
      <th>pixel131</th>
      <th>pixel132</th>
      <th>pixel133</th>
      <th>pixel134</th>
      <th>pixel135</th>
      <th>pixel136</th>
      <th>pixel137</th>
      <th>pixel138</th>
      <th>pixel139</th>
      <th>pixel140</th>
      <th>pixel141</th>
      <th>pixel142</th>
      <th>pixel143</th>
      <th>pixel144</th>
      <th>pixel145</th>
      <th>pixel146</th>
      <th>pixel147</th>
      <th>pixel148</th>
      <th>pixel149</th>
      <th>pixel150</th>
      <th>pixel151</th>
      <th>pixel152</th>
      <th>pixel153</th>
      <th>pixel154</th>
      <th>pixel155</th>
      <th>pixel156</th>
      <th>pixel157</th>
      <th>pixel158</th>
      <th>pixel159</th>
      <th>pixel160</th>
      <th>pixel161</th>
      <th>pixel162</th>
      <th>pixel163</th>
      <th>pixel164</th>
      <th>pixel165</th>
      <th>pixel166</th>
      <th>pixel167</th>
      <th>pixel168</th>
      <th>pixel169</th>
      <th>pixel170</th>
      <th>pixel171</th>
      <th>pixel172</th>
      <th>pixel173</th>
      <th>pixel174</th>
      <th>pixel175</th>
      <th>pixel176</th>
      <th>pixel177</th>
      <th>pixel178</th>
      <th>pixel179</th>
      <th>pixel180</th>
      <th>pixel181</th>
      <th>pixel182</th>
      <th>pixel183</th>
      <th>pixel184</th>
      <th>pixel185</th>
      <th>pixel186</th>
      <th>pixel187</th>
      <th>pixel188</th>
      <th>pixel189</th>
      <th>pixel190</th>
      <th>pixel191</th>
      <th>pixel192</th>
      <th>pixel193</th>
      <th>pixel194</th>
      <th>pixel195</th>
      <th>pixel196</th>
      <th>pixel197</th>
      <th>pixel198</th>
      <th>pixel199</th>
      <th>pixel200</th>
      <th>pixel201</th>
      <th>pixel202</th>
      <th>pixel203</th>
      <th>pixel204</th>
      <th>pixel205</th>
      <th>pixel206</th>
      <th>pixel207</th>
      <th>pixel208</th>
      <th>pixel209</th>
      <th>pixel210</th>
      <th>pixel211</th>
      <th>pixel212</th>
      <th>pixel213</th>
      <th>pixel214</th>
      <th>pixel215</th>
      <th>pixel216</th>
      <th>pixel217</th>
      <th>pixel218</th>
      <th>pixel219</th>
      <th>pixel220</th>
      <th>pixel221</th>
      <th>pixel222</th>
      <th>pixel223</th>
      <th>pixel224</th>
      <th>pixel225</th>
      <th>pixel226</th>
      <th>pixel227</th>
      <th>pixel228</th>
      <th>pixel229</th>
      <th>pixel230</th>
      <th>pixel231</th>
      <th>pixel232</th>
      <th>pixel233</th>
      <th>pixel234</th>
      <th>pixel235</th>
      <th>pixel236</th>
      <th>pixel237</th>
      <th>pixel238</th>
      <th>pixel239</th>
      <th>pixel240</th>
      <th>pixel241</th>
      <th>pixel242</th>
      <th>pixel243</th>
      <th>pixel244</th>
      <th>pixel245</th>
      <th>pixel246</th>
      <th>pixel247</th>
      <th>pixel248</th>
      <th>pixel249</th>
      <th>pixel250</th>
      <th>pixel251</th>
      <th>pixel252</th>
      <th>pixel253</th>
      <th>pixel254</th>
      <th>pixel255</th>
      <th>pixel256</th>
      <th>pixel257</th>
      <th>pixel258</th>
      <th>pixel259</th>
      <th>pixel260</th>
      <th>pixel261</th>
      <th>pixel262</th>
      <th>pixel263</th>
      <th>pixel264</th>
      <th>pixel265</th>
      <th>pixel266</th>
      <th>pixel267</th>
      <th>pixel268</th>
      <th>pixel269</th>
      <th>pixel270</th>
      <th>pixel271</th>
      <th>pixel272</th>
      <th>pixel273</th>
      <th>pixel274</th>
      <th>pixel275</th>
      <th>pixel276</th>
      <th>pixel277</th>
      <th>pixel278</th>
      <th>pixel279</th>
      <th>pixel280</th>
      <th>pixel281</th>
      <th>pixel282</th>
      <th>pixel283</th>
      <th>pixel284</th>
      <th>pixel285</th>
      <th>pixel286</th>
      <th>pixel287</th>
      <th>pixel288</th>
      <th>pixel289</th>
      <th>pixel290</th>
      <th>pixel291</th>
      <th>pixel292</th>
      <th>pixel293</th>
      <th>pixel294</th>
      <th>pixel295</th>
      <th>pixel296</th>
      <th>pixel297</th>
      <th>pixel298</th>
      <th>pixel299</th>
      <th>pixel300</th>
      <th>pixel301</th>
      <th>pixel302</th>
      <th>pixel303</th>
      <th>pixel304</th>
      <th>pixel305</th>
      <th>pixel306</th>
      <th>pixel307</th>
      <th>pixel308</th>
      <th>pixel309</th>
      <th>pixel310</th>
      <th>pixel311</th>
      <th>pixel312</th>
      <th>pixel313</th>
      <th>pixel314</th>
      <th>pixel315</th>
      <th>pixel316</th>
      <th>pixel317</th>
      <th>pixel318</th>
      <th>pixel319</th>
      <th>pixel320</th>
      <th>pixel321</th>
      <th>pixel322</th>
      <th>pixel323</th>
      <th>pixel324</th>
      <th>pixel325</th>
      <th>pixel326</th>
      <th>pixel327</th>
      <th>pixel328</th>
      <th>pixel329</th>
      <th>pixel330</th>
      <th>pixel331</th>
      <th>pixel332</th>
      <th>pixel333</th>
      <th>pixel334</th>
      <th>pixel335</th>
      <th>pixel336</th>
      <th>pixel337</th>
      <th>pixel338</th>
      <th>pixel339</th>
      <th>pixel340</th>
      <th>pixel341</th>
      <th>pixel342</th>
      <th>pixel343</th>
      <th>pixel344</th>
      <th>pixel345</th>
      <th>pixel346</th>
      <th>pixel347</th>
      <th>pixel348</th>
      <th>pixel349</th>
      <th>pixel350</th>
      <th>pixel351</th>
      <th>pixel352</th>
      <th>pixel353</th>
      <th>pixel354</th>
      <th>pixel355</th>
      <th>pixel356</th>
      <th>pixel357</th>
      <th>pixel358</th>
      <th>pixel359</th>
      <th>pixel360</th>
      <th>pixel361</th>
      <th>pixel362</th>
      <th>pixel363</th>
      <th>pixel364</th>
      <th>pixel365</th>
      <th>pixel366</th>
      <th>pixel367</th>
      <th>pixel368</th>
      <th>pixel369</th>
      <th>pixel370</th>
      <th>pixel371</th>
      <th>pixel372</th>
      <th>pixel373</th>
      <th>pixel374</th>
      <th>pixel375</th>
      <th>pixel376</th>
      <th>pixel377</th>
      <th>pixel378</th>
      <th>pixel379</th>
      <th>pixel380</th>
      <th>pixel381</th>
      <th>pixel382</th>
      <th>pixel383</th>
      <th>pixel384</th>
      <th>pixel385</th>
      <th>pixel386</th>
      <th>pixel387</th>
      <th>pixel388</th>
      <th>pixel389</th>
      <th>pixel390</th>
      <th>pixel391</th>
      <th>pixel392</th>
      <th>pixel393</th>
      <th>pixel394</th>
      <th>pixel395</th>
      <th>pixel396</th>
      <th>pixel397</th>
      <th>pixel398</th>
      <th>pixel399</th>
      <th>pixel400</th>
      <th>pixel401</th>
      <th>pixel402</th>
      <th>pixel403</th>
      <th>pixel404</th>
      <th>pixel405</th>
      <th>pixel406</th>
      <th>pixel407</th>
      <th>pixel408</th>
      <th>pixel409</th>
      <th>pixel410</th>
      <th>pixel411</th>
      <th>pixel412</th>
      <th>pixel413</th>
      <th>pixel414</th>
      <th>pixel415</th>
      <th>pixel416</th>
      <th>pixel417</th>
      <th>pixel418</th>
      <th>pixel419</th>
      <th>pixel420</th>
      <th>pixel421</th>
      <th>pixel422</th>
      <th>pixel423</th>
      <th>pixel424</th>
      <th>pixel425</th>
      <th>pixel426</th>
      <th>pixel427</th>
      <th>pixel428</th>
      <th>pixel429</th>
      <th>pixel430</th>
      <th>pixel431</th>
      <th>pixel432</th>
      <th>pixel433</th>
      <th>pixel434</th>
      <th>pixel435</th>
      <th>pixel436</th>
      <th>pixel437</th>
      <th>pixel438</th>
      <th>pixel439</th>
      <th>pixel440</th>
      <th>pixel441</th>
      <th>pixel442</th>
      <th>pixel443</th>
      <th>pixel444</th>
      <th>pixel445</th>
      <th>pixel446</th>
      <th>pixel447</th>
      <th>pixel448</th>
      <th>pixel449</th>
      <th>pixel450</th>
      <th>pixel451</th>
      <th>pixel452</th>
      <th>pixel453</th>
      <th>pixel454</th>
      <th>pixel455</th>
      <th>pixel456</th>
      <th>pixel457</th>
      <th>pixel458</th>
      <th>pixel459</th>
      <th>pixel460</th>
      <th>pixel461</th>
      <th>pixel462</th>
      <th>pixel463</th>
      <th>pixel464</th>
      <th>pixel465</th>
      <th>pixel466</th>
      <th>pixel467</th>
      <th>pixel468</th>
      <th>pixel469</th>
      <th>pixel470</th>
      <th>pixel471</th>
      <th>pixel472</th>
      <th>pixel473</th>
      <th>pixel474</th>
      <th>pixel475</th>
      <th>pixel476</th>
      <th>pixel477</th>
      <th>pixel478</th>
      <th>pixel479</th>
      <th>pixel480</th>
      <th>pixel481</th>
      <th>pixel482</th>
      <th>pixel483</th>
      <th>pixel484</th>
      <th>pixel485</th>
      <th>pixel486</th>
      <th>pixel487</th>
      <th>pixel488</th>
      <th>pixel489</th>
      <th>pixel490</th>
      <th>pixel491</th>
      <th>pixel492</th>
      <th>pixel493</th>
      <th>pixel494</th>
      <th>pixel495</th>
      <th>pixel496</th>
      <th>pixel497</th>
      <th>pixel498</th>
      <th>pixel499</th>
      <th>pixel500</th>
      <th>pixel501</th>
      <th>pixel502</th>
      <th>pixel503</th>
      <th>pixel504</th>
      <th>pixel505</th>
      <th>pixel506</th>
      <th>pixel507</th>
      <th>pixel508</th>
      <th>pixel509</th>
      <th>pixel510</th>
      <th>pixel511</th>
      <th>pixel512</th>
      <th>pixel513</th>
      <th>pixel514</th>
      <th>pixel515</th>
      <th>pixel516</th>
      <th>pixel517</th>
      <th>pixel518</th>
      <th>pixel519</th>
      <th>pixel520</th>
      <th>pixel521</th>
      <th>pixel522</th>
      <th>pixel523</th>
      <th>pixel524</th>
      <th>pixel525</th>
      <th>pixel526</th>
      <th>pixel527</th>
      <th>pixel528</th>
      <th>pixel529</th>
      <th>pixel530</th>
      <th>pixel531</th>
      <th>pixel532</th>
      <th>pixel533</th>
      <th>pixel534</th>
      <th>pixel535</th>
      <th>pixel536</th>
      <th>pixel537</th>
      <th>pixel538</th>
      <th>pixel539</th>
      <th>pixel540</th>
      <th>pixel541</th>
      <th>pixel542</th>
      <th>pixel543</th>
      <th>pixel544</th>
      <th>pixel545</th>
      <th>pixel546</th>
      <th>pixel547</th>
      <th>pixel548</th>
      <th>pixel549</th>
      <th>pixel550</th>
      <th>pixel551</th>
      <th>pixel552</th>
      <th>pixel553</th>
      <th>pixel554</th>
      <th>pixel555</th>
      <th>pixel556</th>
      <th>pixel557</th>
      <th>pixel558</th>
      <th>pixel559</th>
      <th>pixel560</th>
      <th>pixel561</th>
      <th>pixel562</th>
      <th>pixel563</th>
      <th>pixel564</th>
      <th>pixel565</th>
      <th>pixel566</th>
      <th>pixel567</th>
      <th>pixel568</th>
      <th>pixel569</th>
      <th>pixel570</th>
      <th>pixel571</th>
      <th>pixel572</th>
      <th>pixel573</th>
      <th>pixel574</th>
      <th>pixel575</th>
      <th>pixel576</th>
      <th>pixel577</th>
      <th>pixel578</th>
      <th>pixel579</th>
      <th>pixel580</th>
      <th>pixel581</th>
      <th>pixel582</th>
      <th>pixel583</th>
      <th>pixel584</th>
      <th>pixel585</th>
      <th>pixel586</th>
      <th>pixel587</th>
      <th>pixel588</th>
      <th>pixel589</th>
      <th>pixel590</th>
      <th>pixel591</th>
      <th>pixel592</th>
      <th>pixel593</th>
      <th>pixel594</th>
      <th>pixel595</th>
      <th>pixel596</th>
      <th>pixel597</th>
      <th>pixel598</th>
      <th>pixel599</th>
      <th>pixel600</th>
      <th>pixel601</th>
      <th>pixel602</th>
      <th>pixel603</th>
      <th>pixel604</th>
      <th>pixel605</th>
      <th>pixel606</th>
      <th>pixel607</th>
      <th>pixel608</th>
      <th>pixel609</th>
      <th>pixel610</th>
      <th>pixel611</th>
      <th>pixel612</th>
      <th>pixel613</th>
      <th>pixel614</th>
      <th>pixel615</th>
      <th>pixel616</th>
      <th>pixel617</th>
      <th>pixel618</th>
      <th>pixel619</th>
      <th>pixel620</th>
      <th>pixel621</th>
      <th>pixel622</th>
      <th>pixel623</th>
      <th>pixel624</th>
      <th>pixel625</th>
      <th>pixel626</th>
      <th>pixel627</th>
      <th>pixel628</th>
      <th>pixel629</th>
      <th>pixel630</th>
      <th>pixel631</th>
      <th>pixel632</th>
      <th>pixel633</th>
      <th>pixel634</th>
      <th>pixel635</th>
      <th>pixel636</th>
      <th>pixel637</th>
      <th>pixel638</th>
      <th>pixel639</th>
      <th>pixel640</th>
      <th>pixel641</th>
      <th>pixel642</th>
      <th>pixel643</th>
      <th>pixel644</th>
      <th>pixel645</th>
      <th>pixel646</th>
      <th>pixel647</th>
      <th>pixel648</th>
      <th>pixel649</th>
      <th>pixel650</th>
      <th>pixel651</th>
      <th>pixel652</th>
      <th>pixel653</th>
      <th>pixel654</th>
      <th>pixel655</th>
      <th>pixel656</th>
      <th>pixel657</th>
      <th>pixel658</th>
      <th>pixel659</th>
      <th>pixel660</th>
      <th>pixel661</th>
      <th>pixel662</th>
      <th>pixel663</th>
      <th>pixel664</th>
      <th>pixel665</th>
      <th>pixel666</th>
      <th>pixel667</th>
      <th>pixel668</th>
      <th>pixel669</th>
      <th>pixel670</th>
      <th>pixel671</th>
      <th>pixel672</th>
      <th>pixel673</th>
      <th>pixel674</th>
      <th>pixel675</th>
      <th>pixel676</th>
      <th>pixel677</th>
      <th>pixel678</th>
      <th>pixel679</th>
      <th>pixel680</th>
      <th>pixel681</th>
      <th>pixel682</th>
      <th>pixel683</th>
      <th>pixel684</th>
      <th>pixel685</th>
      <th>pixel686</th>
      <th>pixel687</th>
      <th>pixel688</th>
      <th>pixel689</th>
      <th>pixel690</th>
      <th>pixel691</th>
      <th>pixel692</th>
      <th>pixel693</th>
      <th>pixel694</th>
      <th>pixel695</th>
      <th>pixel696</th>
      <th>pixel697</th>
      <th>pixel698</th>
      <th>pixel699</th>
      <th>pixel700</th>
      <th>pixel701</th>
      <th>pixel702</th>
      <th>pixel703</th>
      <th>pixel704</th>
      <th>pixel705</th>
      <th>pixel706</th>
      <th>pixel707</th>
      <th>pixel708</th>
      <th>pixel709</th>
      <th>pixel710</th>
      <th>pixel711</th>
      <th>pixel712</th>
      <th>pixel713</th>
      <th>pixel714</th>
      <th>pixel715</th>
      <th>pixel716</th>
      <th>pixel717</th>
      <th>pixel718</th>
      <th>pixel719</th>
      <th>pixel720</th>
      <th>pixel721</th>
      <th>pixel722</th>
      <th>pixel723</th>
      <th>pixel724</th>
      <th>pixel725</th>
      <th>pixel726</th>
      <th>pixel727</th>
      <th>pixel728</th>
      <th>pixel729</th>
      <th>pixel730</th>
      <th>pixel731</th>
      <th>pixel732</th>
      <th>pixel733</th>
      <th>pixel734</th>
      <th>pixel735</th>
      <th>pixel736</th>
      <th>pixel737</th>
      <th>pixel738</th>
      <th>pixel739</th>
      <th>pixel740</th>
      <th>pixel741</th>
      <th>pixel742</th>
      <th>pixel743</th>
      <th>pixel744</th>
      <th>pixel745</th>
      <th>pixel746</th>
      <th>pixel747</th>
      <th>pixel748</th>
      <th>pixel749</th>
      <th>pixel750</th>
      <th>pixel751</th>
      <th>pixel752</th>
      <th>pixel753</th>
      <th>pixel754</th>
      <th>pixel755</th>
      <th>pixel756</th>
      <th>pixel757</th>
      <th>pixel758</th>
      <th>pixel759</th>
      <th>pixel760</th>
      <th>pixel761</th>
      <th>pixel762</th>
      <th>pixel763</th>
      <th>pixel764</th>
      <th>pixel765</th>
      <th>pixel766</th>
      <th>pixel767</th>
      <th>pixel768</th>
      <th>pixel769</th>
      <th>pixel770</th>
      <th>pixel771</th>
      <th>pixel772</th>
      <th>pixel773</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>8.642878e-08</td>
      <td>-4.371375e-07</td>
      <td>9.682471e-07</td>
      <td>-0.000002</td>
      <td>-5.505288e-07</td>
      <td>4.053924e-07</td>
      <td>4.035032e-07</td>
      <td>2.956210e-07</td>
      <td>4.005417e-07</td>
      <td>-0.000001</td>
      <td>5.516976e-08</td>
      <td>-2.068072e-07</td>
      <td>-9.467687e-08</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>-8.031146e-08</td>
      <td>1.985495e-07</td>
      <td>-1.276844e-07</td>
      <td>-0.000001</td>
      <td>7.044741e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-7.585758e-07</td>
      <td>-0.000006</td>
      <td>0.000006</td>
      <td>0.000002</td>
      <td>-1.279166e-07</td>
      <td>0.000001</td>
      <td>-9.767781e-07</td>
      <td>-6.219337e-07</td>
      <td>1.444539e-07</td>
      <td>-1.736492e-07</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>2.055711e-07</td>
      <td>4.611180e-07</td>
      <td>3.908779e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>0.000003</td>
      <td>-4.594104e-07</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>0.000001</td>
      <td>-4.233165e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>7.057845e-07</td>
      <td>-1.318873e-07</td>
      <td>-1.672773e-07</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>7.656082e-07</td>
      <td>-3.514349e-07</td>
      <td>-9.688781e-07</td>
      <td>8.386749e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.000004</td>
      <td>-0.000003</td>
      <td>-0.000003</td>
      <td>-0.000003</td>
      <td>-0.000004</td>
      <td>-6.836428e-07</td>
      <td>-3.299791e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>5.257434e-07</td>
      <td>8.999998e+00</td>
      <td>6.600000e+01</td>
      <td>9.500000e+01</td>
      <td>9.700000e+01</td>
      <td>1.000000e+01</td>
      <td>-4.949160e-09</td>
      <td>1.000000e+00</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>2.273738e-07</td>
      <td>6.555034e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>2.000001</td>
      <td>0.000003</td>
      <td>10.000000</td>
      <td>1.880000e+02</td>
      <td>2.170000e+02</td>
      <td>238.000000</td>
      <td>250.000003</td>
      <td>251.000002</td>
      <td>2.550000e+02</td>
      <td>218.000003</td>
      <td>2.290000e+02</td>
      <td>2.550000e+02</td>
      <td>2.550000e+02</td>
      <td>2.360000e+02</td>
      <td>2.550000e+02</td>
      <td>1.420000e+02</td>
      <td>-1.021340e-08</td>
      <td>2.000000e+00</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>-0.000005</td>
      <td>9.999999</td>
      <td>6.513797e-07</td>
      <td>70.999999</td>
      <td>255.000005</td>
      <td>236.999996</td>
      <td>236.000001</td>
      <td>236.999995</td>
      <td>229.000004</td>
      <td>225.000003</td>
      <td>234.999999</td>
      <td>2.310000e+02</td>
      <td>224.000001</td>
      <td>2.240000e+02</td>
      <td>2.190000e+02</td>
      <td>2.440000e+02</td>
      <td>1.430000e+02</td>
      <td>1.621610e-07</td>
      <td>1.000000e+00</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>-2.410009e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-8.460442e-08</td>
      <td>0.000003</td>
      <td>6.000000e+00</td>
      <td>-0.000002</td>
      <td>58.000001</td>
      <td>246.999997</td>
      <td>228.000003</td>
      <td>232.999999</td>
      <td>233.999998</td>
      <td>227.999995</td>
      <td>227.000000</td>
      <td>232.000000</td>
      <td>220.999996</td>
      <td>2.280000e+02</td>
      <td>239.000001</td>
      <td>2.310000e+02</td>
      <td>2.450000e+02</td>
      <td>1.570000e+02</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.715096e-07</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-0.000004</td>
      <td>3.999998e+00</td>
      <td>-0.000004</td>
      <td>1.440000e+02</td>
      <td>248.999999</td>
      <td>228.000004</td>
      <td>241.999999</td>
      <td>240.999997</td>
      <td>240.000003</td>
      <td>237.999997</td>
      <td>242.999999</td>
      <td>238.000004</td>
      <td>234.000004</td>
      <td>240.000001</td>
      <td>230.999996</td>
      <td>2.460000e+02</td>
      <td>2.150000e+02</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>4.334051e-07</td>
      <td>-0.000002</td>
      <td>7.406876e-07</td>
      <td>-0.000003</td>
      <td>-9.455209e-07</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>-0.000002</td>
      <td>154.999998</td>
      <td>246.999996</td>
      <td>217.000003</td>
      <td>230.000003</td>
      <td>228.000001</td>
      <td>228.000003</td>
      <td>229.000003</td>
      <td>223.999996</td>
      <td>2.300000e+02</td>
      <td>229.999996</td>
      <td>237.999998</td>
      <td>220.000002</td>
      <td>244.999993</td>
      <td>2.220000e+02</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>-8.785177e-07</td>
      <td>-5.893633e-07</td>
      <td>9.804652e-07</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>-0.000005</td>
      <td>-0.000004</td>
      <td>-0.000004</td>
      <td>183.999997</td>
      <td>250.000004</td>
      <td>2.260000e+02</td>
      <td>232.000004</td>
      <td>228.999998</td>
      <td>228.000002</td>
      <td>226.999997</td>
      <td>221.999995</td>
      <td>229.000000</td>
      <td>234.000001</td>
      <td>231.999997</td>
      <td>228.000004</td>
      <td>2.410000e+02</td>
      <td>2.440000e+02</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>5.449854e-07</td>
      <td>6.690875e-07</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>0.000002</td>
      <td>254.999999</td>
      <td>241.000002</td>
      <td>229.000004</td>
      <td>233.999998</td>
      <td>235.999997</td>
      <td>236.000002</td>
      <td>233.999998</td>
      <td>231.000000</td>
      <td>2.350000e+02</td>
      <td>239.000004</td>
      <td>234.999997</td>
      <td>2.370000e+02</td>
      <td>230.999995</td>
      <td>255.000001</td>
      <td>1.000000e+02</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>-8.247956e-07</td>
      <td>2.124748e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>1.999999</td>
      <td>0.000005</td>
      <td>29.000004</td>
      <td>254.999996</td>
      <td>229.999997</td>
      <td>237.000003</td>
      <td>234.000000</td>
      <td>236.999997</td>
      <td>238.000002</td>
      <td>236.000000</td>
      <td>235.000003</td>
      <td>235.999998</td>
      <td>232.000001</td>
      <td>236.999995</td>
      <td>237.000003</td>
      <td>229.000001</td>
      <td>2.390000e+02</td>
      <td>226.999998</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>-0.000004</td>
      <td>176.999998</td>
      <td>249.999996</td>
      <td>225.999996</td>
      <td>2.370000e+02</td>
      <td>233.000003</td>
      <td>234.999999</td>
      <td>237.000003</td>
      <td>234.999996</td>
      <td>233.000005</td>
      <td>236.000004</td>
      <td>235.999998</td>
      <td>235.000000</td>
      <td>233.000000</td>
      <td>230.000003</td>
      <td>226.000004</td>
      <td>255.000000</td>
      <td>2.700000e+01</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-7.203892e-07</td>
      <td>0.000001</td>
      <td>3.000000</td>
      <td>0.000001</td>
      <td>24.999996</td>
      <td>254.999996</td>
      <td>230.999999</td>
      <td>245.000004</td>
      <td>234.999997</td>
      <td>234.000001</td>
      <td>232.000003</td>
      <td>235.999999</td>
      <td>235.999998</td>
      <td>233.999996</td>
      <td>2.370000e+02</td>
      <td>235.999995</td>
      <td>235.999998</td>
      <td>230.000004</td>
      <td>227.999997</td>
      <td>2.210000e+02</td>
      <td>254.999991</td>
      <td>9.300000e+01</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>0.000002</td>
      <td>-8.765240e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>-0.000001</td>
      <td>0.000003</td>
      <td>201.999999</td>
      <td>243.999999</td>
      <td>226.000003</td>
      <td>2.350000e+02</td>
      <td>237.000004</td>
      <td>234.000001</td>
      <td>230.999999</td>
      <td>233.999997</td>
      <td>235.000000</td>
      <td>232.000001</td>
      <td>234.999997</td>
      <td>2.350000e+02</td>
      <td>2.300000e+02</td>
      <td>226.000000</td>
      <td>224.999997</td>
      <td>2.210000e+02</td>
      <td>255.000004</td>
      <td>1.510000e+02</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>7.308653e-07</td>
      <td>-0.000002</td>
      <td>1.000003</td>
      <td>1.999999e+00</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>1.070000e+02</td>
      <td>254.999997</td>
      <td>231.999999</td>
      <td>232.999999</td>
      <td>231.000002</td>
      <td>233.999996</td>
      <td>236.000002</td>
      <td>232.999996</td>
      <td>233.000004</td>
      <td>231.000002</td>
      <td>229.999999</td>
      <td>234.000000</td>
      <td>232.999996</td>
      <td>2.280000e+02</td>
      <td>229.999995</td>
      <td>231.999996</td>
      <td>2.230000e+02</td>
      <td>2.550000e+02</td>
      <td>1.730000e+02</td>
      <td>-2.135472e-07</td>
      <td>1.000000e+00</td>
      <td>5.000000e+00</td>
      <td>8.000000</td>
      <td>-5.332055e-07</td>
      <td>5.139264e-07</td>
      <td>-0.000002</td>
      <td>-0.000004</td>
      <td>74.000001</td>
      <td>254.999999</td>
      <td>231.999998</td>
      <td>229.000002</td>
      <td>233.000003</td>
      <td>232.999999</td>
      <td>232.999998</td>
      <td>234.999997</td>
      <td>232.999999</td>
      <td>232.000003</td>
      <td>232.999998</td>
      <td>2.280000e+02</td>
      <td>223.000004</td>
      <td>2.330000e+02</td>
      <td>228.000000</td>
      <td>227.000001</td>
      <td>229.000000</td>
      <td>220.000000</td>
      <td>2.550000e+02</td>
      <td>1.340000e+02</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>-5.423990e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>68.999999</td>
      <td>197.999998</td>
      <td>254.999996</td>
      <td>237.000001</td>
      <td>229.000001</td>
      <td>231.000004</td>
      <td>233.000000</td>
      <td>231.999996</td>
      <td>233.999999</td>
      <td>2.340000e+02</td>
      <td>234.000004</td>
      <td>230.999996</td>
      <td>231.000002</td>
      <td>2.280000e+02</td>
      <td>2.310000e+02</td>
      <td>2.230000e+02</td>
      <td>227.999996</td>
      <td>231.999995</td>
      <td>2.380000e+02</td>
      <td>2.330000e+02</td>
      <td>255.000009</td>
      <td>2.000000e+00</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>74.999998</td>
      <td>1.330000e+02</td>
      <td>196.000003</td>
      <td>245.999997</td>
      <td>254.999999</td>
      <td>244.000001</td>
      <td>227.999995</td>
      <td>230.999998</td>
      <td>231.999996</td>
      <td>233.000002</td>
      <td>237.000000</td>
      <td>232.999998</td>
      <td>234.000001</td>
      <td>233.999996</td>
      <td>232.999998</td>
      <td>233.000000</td>
      <td>230.000000</td>
      <td>222.000002</td>
      <td>227.000002</td>
      <td>250.999998</td>
      <td>246.000005</td>
      <td>2.320000e+02</td>
      <td>2.200000e+02</td>
      <td>232.999997</td>
      <td>1.860000e+02</td>
      <td>-3.012789e-07</td>
      <td>6.600000e+01</td>
      <td>2.280000e+02</td>
      <td>2.410000e+02</td>
      <td>2.420000e+02</td>
      <td>2.370000e+02</td>
      <td>2.350000e+02</td>
      <td>224.999999</td>
      <td>211.999995</td>
      <td>2.300000e+02</td>
      <td>232.000002</td>
      <td>233.000002</td>
      <td>235.000004</td>
      <td>235.999999</td>
      <td>232.000002</td>
      <td>234.000000</td>
      <td>2.340000e+02</td>
      <td>235.999999</td>
      <td>226.000000</td>
      <td>227.000003</td>
      <td>254.999995</td>
      <td>254.999998</td>
      <td>197.999995</td>
      <td>215.000005</td>
      <td>202.000000</td>
      <td>186.999999</td>
      <td>2.360000e+02</td>
      <td>140.000000</td>
      <td>4.011770e-07</td>
      <td>1.750000e+02</td>
      <td>2.470000e+02</td>
      <td>2.140000e+02</td>
      <td>224.000000</td>
      <td>222.000002</td>
      <td>221.000005</td>
      <td>2.280000e+02</td>
      <td>233.000004</td>
      <td>2.310000e+02</td>
      <td>2.330000e+02</td>
      <td>231.000003</td>
      <td>231.000000</td>
      <td>229.999996</td>
      <td>230.999999</td>
      <td>234.999996</td>
      <td>236.000004</td>
      <td>227.000001</td>
      <td>235.999997</td>
      <td>255.000001</td>
      <td>217.000005</td>
      <td>17.000001</td>
      <td>-2.610268e-07</td>
      <td>204.999999</td>
      <td>220.000001</td>
      <td>204.999997</td>
      <td>218.000003</td>
      <td>7.500000e+01</td>
      <td>-1.202886e-08</td>
      <td>9.600000e+01</td>
      <td>2.540000e+02</td>
      <td>2.380000e+02</td>
      <td>2.210000e+02</td>
      <td>2.290000e+02</td>
      <td>222.999997</td>
      <td>225.999995</td>
      <td>2.320000e+02</td>
      <td>2.300000e+02</td>
      <td>231.000001</td>
      <td>2.310000e+02</td>
      <td>2.350000e+02</td>
      <td>236.000004</td>
      <td>237.999995</td>
      <td>224.000005</td>
      <td>230.000000</td>
      <td>2.410000e+02</td>
      <td>254.999998</td>
      <td>78.000001</td>
      <td>0.000004</td>
      <td>6.923674e-07</td>
      <td>0.000001</td>
      <td>2.170000e+02</td>
      <td>2.070000e+02</td>
      <td>1.980000e+02</td>
      <td>211.999995</td>
      <td>2.700000e+01</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>9.100000e+01</td>
      <td>2.550000e+02</td>
      <td>2.520000e+02</td>
      <td>243.999992</td>
      <td>235.000002</td>
      <td>2.190000e+02</td>
      <td>2.200000e+02</td>
      <td>224.000004</td>
      <td>2.270000e+02</td>
      <td>232.000003</td>
      <td>235.000003</td>
      <td>224.999995</td>
      <td>2.230000e+02</td>
      <td>234.000001</td>
      <td>254.999996</td>
      <td>2.000000e+02</td>
      <td>0.000005</td>
      <td>-0.000004</td>
      <td>-0.000004</td>
      <td>4.021377e-07</td>
      <td>0.000002</td>
      <td>187.000001</td>
      <td>216.000000</td>
      <td>2.200000e+02</td>
      <td>2.150000e+02</td>
      <td>5.000000e+00</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>9.100000e+01</td>
      <td>199.999998</td>
      <td>2.550000e+02</td>
      <td>255.000003</td>
      <td>2.550000e+02</td>
      <td>253.999999</td>
      <td>245.000000</td>
      <td>233.999998</td>
      <td>235.999999</td>
      <td>243.999996</td>
      <td>254.999996</td>
      <td>255.000000</td>
      <td>1.170000e+02</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>0.999999</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>209.999999</td>
      <td>2.060000e+02</td>
      <td>1.810000e+02</td>
      <td>1.470000e+02</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>7.000000e+00</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>-8.448512e-07</td>
      <td>0.000002</td>
      <td>49.000000</td>
      <td>1.160000e+02</td>
      <td>1.810000e+02</td>
      <td>208.999997</td>
      <td>2.260000e+02</td>
      <td>223.000000</td>
      <td>205.999999</td>
      <td>139.999999</td>
      <td>-0.000001</td>
      <td>8.076585e-07</td>
      <td>-0.000005</td>
      <td>3.000001</td>
      <td>0.000005</td>
      <td>-1.852046e-07</td>
      <td>6.976350e-08</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-3.723668e-07</td>
      <td>-0.000002</td>
      <td>7.009453e-07</td>
      <td>-0.000002</td>
      <td>3.933536e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>-0.000005</td>
      <td>-5.581073e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>2.139088e-07</td>
      <td>0.000002</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>-8.592667e-07</td>
      <td>0.000002</td>
      <td>-7.577064e-07</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>3.743793e-07</td>
      <td>-6.565379e-07</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>-2.304249e-08</td>
      <td>9.522988e-07</td>
      <td>-3.426212e-07</td>
      <td>-6.002562e-07</td>
      <td>-3.102456e-07</td>
      <td>8.776177e-07</td>
      <td>-0.000002</td>
      <td>6.155732e-07</td>
      <td>-3.593068e-07</td>
      <td>0.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>3.400000e+01</td>
      <td>4.500000e+01</td>
      <td>1.650000e+02</td>
      <td>1.480000e+02</td>
      <td>34.000000</td>
      <td>-5.505288e-07</td>
      <td>4.053924e-07</td>
      <td>4.035032e-07</td>
      <td>2.956210e-07</td>
      <td>6.300000e+01</td>
      <td>190.000002</td>
      <td>1.390000e+02</td>
      <td>2.600000e+01</td>
      <td>2.800000e+01</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>2.000000e+00</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>1.070000e+02</td>
      <td>1.310000e+02</td>
      <td>6.400000e+01</td>
      <td>151.000002</td>
      <td>2.290000e+02</td>
      <td>226.000002</td>
      <td>219.999994</td>
      <td>2.410000e+02</td>
      <td>236.999998</td>
      <td>243.000004</td>
      <td>255.000000</td>
      <td>1.950000e+02</td>
      <td>92.000000</td>
      <td>9.000000e+01</td>
      <td>1.030000e+02</td>
      <td>9.000000e+01</td>
      <td>8.000000e+00</td>
      <td>-8.071513e-09</td>
      <td>1.000000e+00</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>1.040000e+02</td>
      <td>9.000000e+01</td>
      <td>5.200000e+01</td>
      <td>61.000000</td>
      <td>76.000000</td>
      <td>172.000000</td>
      <td>183.999998</td>
      <td>197.000003</td>
      <td>191.000000</td>
      <td>1.880000e+02</td>
      <td>169.000002</td>
      <td>160.000000</td>
      <td>128.999999</td>
      <td>7.000000e+01</td>
      <td>88.000001</td>
      <td>52.000000</td>
      <td>8.500000e+01</td>
      <td>7.700000e+01</td>
      <td>-1.672773e-07</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.700000e+01</td>
      <td>8.900000e+01</td>
      <td>5.700000e+01</td>
      <td>5.700000e+01</td>
      <td>7.000000e+01</td>
      <td>59.000001</td>
      <td>127.000000</td>
      <td>184.999999</td>
      <td>182.000001</td>
      <td>179.000002</td>
      <td>168.999998</td>
      <td>163.000000</td>
      <td>1.710000e+02</td>
      <td>8.900000e+01</td>
      <td>85.000000</td>
      <td>77.000000</td>
      <td>5.900000e+01</td>
      <td>5.700000e+01</td>
      <td>8.100000e+01</td>
      <td>8.000000e+00</td>
      <td>-1.558985e-07</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>5.000000e+01</td>
      <td>6.700000e+01</td>
      <td>6.400000e+01</td>
      <td>65.000000</td>
      <td>65.000000</td>
      <td>78.000000</td>
      <td>69.000000</td>
      <td>139.000001</td>
      <td>215.000005</td>
      <td>1.780000e+02</td>
      <td>1.960000e+02</td>
      <td>177.000000</td>
      <td>81.999999</td>
      <td>83.000001</td>
      <td>8.200000e+01</td>
      <td>65.999999</td>
      <td>5.900000e+01</td>
      <td>6.000000e+01</td>
      <td>8.800000e+01</td>
      <td>4.100000e+01</td>
      <td>-4.548926e-07</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>7.200000e+01</td>
      <td>69.000000</td>
      <td>67.000001</td>
      <td>64.000000</td>
      <td>67.000000</td>
      <td>58.999999</td>
      <td>66.999999</td>
      <td>5.800000e+01</td>
      <td>91.000000</td>
      <td>177.000000</td>
      <td>116.000000</td>
      <td>35.000001</td>
      <td>52.999998</td>
      <td>71.000001</td>
      <td>58.000002</td>
      <td>63.000000</td>
      <td>5.700000e+01</td>
      <td>72.000000</td>
      <td>6.400000e+01</td>
      <td>7.300000e+01</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.000000e+00</td>
      <td>8.400000e+01</td>
      <td>60.000001</td>
      <td>69.000000</td>
      <td>60.000000</td>
      <td>6.100000e+01</td>
      <td>56.000001</td>
      <td>5.200000e+01</td>
      <td>68.999999</td>
      <td>55.999999</td>
      <td>76.000000</td>
      <td>58.000000</td>
      <td>40.000001</td>
      <td>56.999998</td>
      <td>54.000001</td>
      <td>45.000001</td>
      <td>51.000001</td>
      <td>46.000000</td>
      <td>6.500000e+01</td>
      <td>61.000000</td>
      <td>9.700000e+01</td>
      <td>2.000000e+00</td>
      <td>-5.450623e-07</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>7.100000e+01</td>
      <td>5.400000e+01</td>
      <td>56.000000</td>
      <td>72.000000</td>
      <td>82.000000</td>
      <td>66.000000</td>
      <td>48.000000</td>
      <td>5.200000e+01</td>
      <td>52.999997</td>
      <td>7.600000e+01</td>
      <td>106.000000</td>
      <td>48.000000</td>
      <td>60.999999</td>
      <td>73.000000</td>
      <td>73.000000</td>
      <td>61.000000</td>
      <td>78.000000</td>
      <td>71.999999</td>
      <td>63.999999</td>
      <td>56.999999</td>
      <td>82.000000</td>
      <td>1.290000e+02</td>
      <td>-6.404821e-07</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>1.540000e+02</td>
      <td>1.260000e+02</td>
      <td>16.000000</td>
      <td>7.200000e+01</td>
      <td>67.000000</td>
      <td>4.100000e+01</td>
      <td>73.000000</td>
      <td>61.000001</td>
      <td>57.000001</td>
      <td>63.000002</td>
      <td>97.000000</td>
      <td>65.000002</td>
      <td>60.999998</td>
      <td>71.000000</td>
      <td>61.000002</td>
      <td>70.000001</td>
      <td>77.999999</td>
      <td>5.400000e+01</td>
      <td>108.000000</td>
      <td>83.000000</td>
      <td>144.000001</td>
      <td>117.000001</td>
      <td>8.170825e-07</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>1.450000e+02</td>
      <td>1.470000e+02</td>
      <td>1.500000e+02</td>
      <td>122.000001</td>
      <td>106.000000</td>
      <td>106.000000</td>
      <td>51.999998</td>
      <td>55.999999</td>
      <td>59.999999</td>
      <td>81.999999</td>
      <td>6.500000e+01</td>
      <td>55.999997</td>
      <td>69.000001</td>
      <td>52.999997</td>
      <td>70.000001</td>
      <td>61.000002</td>
      <td>94.000000</td>
      <td>205.999996</td>
      <td>168.000003</td>
      <td>127.000002</td>
      <td>-1.907800e-08</td>
      <td>-6.678754e-07</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>5.449854e-07</td>
      <td>9.500000e+01</td>
      <td>188.000003</td>
      <td>190.999996</td>
      <td>98.000000</td>
      <td>70.000000</td>
      <td>49.999998</td>
      <td>52.999998</td>
      <td>57.000000</td>
      <td>58.000001</td>
      <td>52.000002</td>
      <td>52.000001</td>
      <td>76.000000</td>
      <td>52.999998</td>
      <td>53.999996</td>
      <td>64.000001</td>
      <td>7.700000e+01</td>
      <td>161.999997</td>
      <td>85.000000</td>
      <td>1.135173e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>-8.247956e-07</td>
      <td>2.124748e-07</td>
      <td>-0.000002</td>
      <td>31.999998</td>
      <td>60.000001</td>
      <td>63.999999</td>
      <td>51.999997</td>
      <td>53.000000</td>
      <td>50.000003</td>
      <td>61.000000</td>
      <td>51.000002</td>
      <td>50.999997</td>
      <td>75.000002</td>
      <td>69.999999</td>
      <td>60.999997</td>
      <td>86.000000</td>
      <td>19.999996</td>
      <td>-0.000002</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>1.000002</td>
      <td>-2.314611e-07</td>
      <td>-0.000002</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>0.000001</td>
      <td>2.000000</td>
      <td>0.000001</td>
      <td>40.999999</td>
      <td>73.000000</td>
      <td>64.000000</td>
      <td>53.000000</td>
      <td>47.999999</td>
      <td>52.000004</td>
      <td>58.000004</td>
      <td>5.200000e+01</td>
      <td>52.999999</td>
      <td>56.999997</td>
      <td>64.000002</td>
      <td>53.000000</td>
      <td>77.000002</td>
      <td>43.999999</td>
      <td>-0.000005</td>
      <td>3.000005</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-7.203892e-07</td>
      <td>32.000000</td>
      <td>69.000001</td>
      <td>47.000001</td>
      <td>58.000002</td>
      <td>51.999996</td>
      <td>52.999996</td>
      <td>53.999996</td>
      <td>52.000001</td>
      <td>56.000004</td>
      <td>56.999997</td>
      <td>58.999996</td>
      <td>51.000000</td>
      <td>66.000002</td>
      <td>3.900000e+01</td>
      <td>-0.000001</td>
      <td>1.000001</td>
      <td>0.000004</td>
      <td>0.000002</td>
      <td>3.101954e-07</td>
      <td>-0.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>0.000002</td>
      <td>1.999999e+00</td>
      <td>0.000002</td>
      <td>26.000003</td>
      <td>59.999999</td>
      <td>47.999997</td>
      <td>58.000000</td>
      <td>52.999997</td>
      <td>49.999999</td>
      <td>5.200000e+01</td>
      <td>50.999996</td>
      <td>56.999999</td>
      <td>62.999997</td>
      <td>57.000003</td>
      <td>57.999998</td>
      <td>73.000001</td>
      <td>56.999999</td>
      <td>7.423447e-07</td>
      <td>9.999991e-01</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>3.281200e-07</td>
      <td>-0.000002</td>
      <td>-1.307099e-07</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>7.308653e-07</td>
      <td>-0.000002</td>
      <td>1.999998</td>
      <td>-5.223541e-09</td>
      <td>26.000001</td>
      <td>53.000002</td>
      <td>5.000000e+01</td>
      <td>63.000003</td>
      <td>57.000000</td>
      <td>53.000003</td>
      <td>60.999999</td>
      <td>56.999998</td>
      <td>55.999997</td>
      <td>64.000004</td>
      <td>56.999997</td>
      <td>60.999998</td>
      <td>69.000000</td>
      <td>64.999998</td>
      <td>0.000005</td>
      <td>1.000005e+00</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>3.892250e-07</td>
      <td>6.972123e-07</td>
      <td>-2.291508e-07</td>
      <td>-2.135472e-07</td>
      <td>-4.984975e-07</td>
      <td>-8.258425e-07</td>
      <td>0.000001</td>
      <td>-5.332055e-07</td>
      <td>2.000003e+00</td>
      <td>-0.000002</td>
      <td>32.000000</td>
      <td>50.000000</td>
      <td>66.000002</td>
      <td>76.000002</td>
      <td>60.999997</td>
      <td>55.999997</td>
      <td>58.999998</td>
      <td>57.000004</td>
      <td>59.999999</td>
      <td>64.999999</td>
      <td>51.000002</td>
      <td>63.999998</td>
      <td>6.100000e+01</td>
      <td>59.000000</td>
      <td>-4.653743e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-7.986391e-07</td>
      <td>-4.286650e-07</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>-5.423990e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>51.000002</td>
      <td>47.000002</td>
      <td>69.000002</td>
      <td>66.999997</td>
      <td>58.999998</td>
      <td>56.000002</td>
      <td>59.999997</td>
      <td>58.000000</td>
      <td>6.100000e+01</td>
      <td>63.000000</td>
      <td>52.999998</td>
      <td>66.000002</td>
      <td>6.400000e+01</td>
      <td>5.800000e+01</td>
      <td>-6.241514e-07</td>
      <td>-0.000005</td>
      <td>1.000003</td>
      <td>5.651978e-07</td>
      <td>-5.163183e-07</td>
      <td>0.000002</td>
      <td>-1.174552e-07</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>-0.000001</td>
      <td>5.352320e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>59.999999</td>
      <td>61.000001</td>
      <td>77.000000</td>
      <td>71.999997</td>
      <td>64.000003</td>
      <td>57.999997</td>
      <td>62.999996</td>
      <td>58.999998</td>
      <td>65.999999</td>
      <td>63.000000</td>
      <td>58.000000</td>
      <td>67.000003</td>
      <td>72.000000</td>
      <td>71.999999</td>
      <td>11.000003</td>
      <td>-0.000005</td>
      <td>9.999988e-01</td>
      <td>-3.174695e-07</td>
      <td>-0.000002</td>
      <td>-5.629809e-07</td>
      <td>-3.012789e-07</td>
      <td>-1.447285e-07</td>
      <td>-6.046403e-07</td>
      <td>1.118729e-07</td>
      <td>-9.513843e-07</td>
      <td>1.000001e+00</td>
      <td>2.894738e-08</td>
      <td>0.000004</td>
      <td>76.000000</td>
      <td>5.800000e+01</td>
      <td>81.999998</td>
      <td>82.999998</td>
      <td>68.999996</td>
      <td>63.000002</td>
      <td>66.000002</td>
      <td>64.000001</td>
      <td>7.200000e+01</td>
      <td>66.000002</td>
      <td>64.000004</td>
      <td>65.000000</td>
      <td>76.000001</td>
      <td>87.999999</td>
      <td>32.000002</td>
      <td>-0.000002</td>
      <td>2.999998</td>
      <td>-0.000002</td>
      <td>9.265024e-07</td>
      <td>0.000002</td>
      <td>4.011770e-07</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>-2.624320e-07</td>
      <td>-0.000002</td>
      <td>2.000001</td>
      <td>-0.000003</td>
      <td>1.100000e+01</td>
      <td>92.000000</td>
      <td>6.400000e+01</td>
      <td>8.500000e+01</td>
      <td>84.999998</td>
      <td>78.999999</td>
      <td>71.999999</td>
      <td>69.000002</td>
      <td>72.999999</td>
      <td>85.999999</td>
      <td>73.000002</td>
      <td>70.999998</td>
      <td>67.000000</td>
      <td>83.999999</td>
      <td>98.000000</td>
      <td>2.800000e+01</td>
      <td>0.000003</td>
      <td>3.000001</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>7.528400e-07</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>-2.873870e-07</td>
      <td>1.999999e+00</td>
      <td>0.000001</td>
      <td>15.999998</td>
      <td>9.800000e+01</td>
      <td>8.500000e+01</td>
      <td>88.000001</td>
      <td>8.100000e+01</td>
      <td>8.800000e+01</td>
      <td>79.000001</td>
      <td>72.999999</td>
      <td>83.000000</td>
      <td>91.000001</td>
      <td>7.800000e+01</td>
      <td>81.000000</td>
      <td>84.000001</td>
      <td>88.000000</td>
      <td>1.220000e+02</td>
      <td>35.000001</td>
      <td>-8.325330e-07</td>
      <td>2.000000e+00</td>
      <td>-1.849482e-07</td>
      <td>0.000002</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>4.654391e-07</td>
      <td>1.000002</td>
      <td>-0.000001</td>
      <td>3.300000e+01</td>
      <td>1.350000e+02</td>
      <td>96.000000</td>
      <td>8.900000e+01</td>
      <td>89.999999</td>
      <td>96.000000</td>
      <td>83.999999</td>
      <td>9.000000e+01</td>
      <td>97.999999</td>
      <td>99.999999</td>
      <td>9.000000e+01</td>
      <td>90.000000</td>
      <td>102.000000</td>
      <td>72.999999</td>
      <td>1.350000e+02</td>
      <td>52.000001</td>
      <td>0.000002</td>
      <td>2.000001</td>
      <td>7.859119e-07</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>-6.257008e-07</td>
      <td>2.000002</td>
      <td>9.323434e-07</td>
      <td>101.000000</td>
      <td>1.290000e+02</td>
      <td>110.000000</td>
      <td>115.000000</td>
      <td>97.999999</td>
      <td>96.000001</td>
      <td>91.000000</td>
      <td>95.000001</td>
      <td>101.000000</td>
      <td>1.030000e+02</td>
      <td>97.000000</td>
      <td>92.000000</td>
      <td>104.000000</td>
      <td>101.000000</td>
      <td>113.000001</td>
      <td>59.000000</td>
      <td>-0.000001</td>
      <td>3.000000e+00</td>
      <td>7.765040e-07</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>2.999998e+00</td>
      <td>0.000002</td>
      <td>46.000000</td>
      <td>1.560000e+02</td>
      <td>9.400000e+01</td>
      <td>117.000000</td>
      <td>1.170000e+02</td>
      <td>127.000000</td>
      <td>112.000000</td>
      <td>112.000000</td>
      <td>128.000000</td>
      <td>1.230000e+02</td>
      <td>133.000000</td>
      <td>121.000001</td>
      <td>102.000000</td>
      <td>9.200000e+01</td>
      <td>1.500000e+02</td>
      <td>51.000000</td>
      <td>-0.000002</td>
      <td>3.000002</td>
      <td>0.000002</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>1.270000e+02</td>
      <td>160.000000</td>
      <td>1.040000e+02</td>
      <td>58.999999</td>
      <td>7.300000e+01</td>
      <td>85.000000</td>
      <td>82.000000</td>
      <td>95.000000</td>
      <td>92.000000</td>
      <td>86.000000</td>
      <td>90.000000</td>
      <td>1.060000e+02</td>
      <td>165.999997</td>
      <td>112.999998</td>
      <td>-0.000001</td>
      <td>2.139088e-07</td>
      <td>1.000002</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>-8.592667e-07</td>
      <td>0.000002</td>
      <td>-7.577064e-07</td>
      <td>109.999998</td>
      <td>222.000005</td>
      <td>213.999997</td>
      <td>181.000000</td>
      <td>163.000000</td>
      <td>169.000001</td>
      <td>175.999999</td>
      <td>169.000001</td>
      <td>171.999999</td>
      <td>188.000002</td>
      <td>187.000002</td>
      <td>82.999999</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>3.743793e-07</td>
      <td>-6.565379e-07</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>-2.304249e-08</td>
      <td>6.100000e+01</td>
      <td>1.260000e+02</td>
      <td>1.420000e+02</td>
      <td>1.620000e+02</td>
      <td>1.830000e+02</td>
      <td>171.000000</td>
      <td>1.450000e+02</td>
      <td>5.400000e+01</td>
      <td>0.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>6.900000e+01</td>
      <td>4.800000e+01</td>
      <td>9.682471e-07</td>
      <td>3.000002</td>
      <td>-5.505288e-07</td>
      <td>4.053924e-07</td>
      <td>4.035032e-07</td>
      <td>2.956210e-07</td>
      <td>4.005417e-07</td>
      <td>-0.000001</td>
      <td>5.516976e-08</td>
      <td>1.300000e+02</td>
      <td>3.200000e+01</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>-8.031146e-08</td>
      <td>1.985495e-07</td>
      <td>1.770000e+02</td>
      <td>173.000001</td>
      <td>7.044741e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-7.585758e-07</td>
      <td>0.999997</td>
      <td>0.000006</td>
      <td>1.999996</td>
      <td>-1.279166e-07</td>
      <td>0.000001</td>
      <td>2.040000e+02</td>
      <td>7.800000e+01</td>
      <td>1.444539e-07</td>
      <td>-1.736492e-07</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>2.055711e-07</td>
      <td>4.611180e-07</td>
      <td>3.908779e-07</td>
      <td>127.999997</td>
      <td>184.999995</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.999996</td>
      <td>1.000004</td>
      <td>-4.594104e-07</td>
      <td>0.000004</td>
      <td>5.000005</td>
      <td>0.000001</td>
      <td>5.100000e+01</td>
      <td>200.999995</td>
      <td>-0.000002</td>
      <td>7.057845e-07</td>
      <td>-1.318873e-07</td>
      <td>-1.672773e-07</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>7.656082e-07</td>
      <td>-3.514349e-07</td>
      <td>-9.688781e-07</td>
      <td>7.300000e+01</td>
      <td>212.000004</td>
      <td>17.999997</td>
      <td>0.000004</td>
      <td>0.999997</td>
      <td>-0.000003</td>
      <td>-0.000003</td>
      <td>-0.000004</td>
      <td>-6.836428e-07</td>
      <td>-3.299791e-07</td>
      <td>121.999999</td>
      <td>189.000002</td>
      <td>5.257434e-07</td>
      <td>-3.949514e-07</td>
      <td>-1.805364e-07</td>
      <td>4.216997e-07</td>
      <td>-1.558985e-07</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>2.273738e-07</td>
      <td>6.555034e-07</td>
      <td>-0.000001</td>
      <td>44.000000</td>
      <td>219.999997</td>
      <td>97.000000</td>
      <td>0.000003</td>
      <td>2.000004</td>
      <td>9.626116e-07</td>
      <td>6.564152e-07</td>
      <td>1.000003</td>
      <td>0.999998</td>
      <td>-0.000005</td>
      <td>1.770000e+02</td>
      <td>166.000002</td>
      <td>5.942894e-07</td>
      <td>9.570876e-07</td>
      <td>7.753735e-07</td>
      <td>-8.642634e-07</td>
      <td>-4.548926e-07</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>41.999999</td>
      <td>206.000005</td>
      <td>197.999999</td>
      <td>6.513797e-07</td>
      <td>-0.000001</td>
      <td>0.000005</td>
      <td>-0.000004</td>
      <td>0.000004</td>
      <td>-0.000001</td>
      <td>45.999999</td>
      <td>209.000003</td>
      <td>170.999998</td>
      <td>2.016850e-07</td>
      <td>-0.000002</td>
      <td>-4.557509e-07</td>
      <td>5.144796e-07</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>-2.410009e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>4.900000e+01</td>
      <td>224.000005</td>
      <td>1.940000e+02</td>
      <td>162.000001</td>
      <td>20.000002</td>
      <td>-0.000001</td>
      <td>-0.000004</td>
      <td>-0.000005</td>
      <td>47.999998</td>
      <td>203.000001</td>
      <td>195.999999</td>
      <td>194.999998</td>
      <td>0.000005</td>
      <td>-7.583148e-07</td>
      <td>-0.000002</td>
      <td>8.919840e-07</td>
      <td>-6.656718e-07</td>
      <td>-5.450623e-07</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.715096e-07</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>57.000000</td>
      <td>221.999997</td>
      <td>1.800000e+02</td>
      <td>202.999997</td>
      <td>2.150000e+02</td>
      <td>194.999999</td>
      <td>194.999999</td>
      <td>207.000000</td>
      <td>213.000002</td>
      <td>190.000000</td>
      <td>183.000001</td>
      <td>222.999998</td>
      <td>1.999996</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-7.762443e-07</td>
      <td>-6.404821e-07</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>4.334051e-07</td>
      <td>-0.000002</td>
      <td>7.406876e-07</td>
      <td>-0.000003</td>
      <td>1.130000e+02</td>
      <td>223.000002</td>
      <td>166.000000</td>
      <td>165.000001</td>
      <td>173.999998</td>
      <td>185.000001</td>
      <td>186.999998</td>
      <td>181.000001</td>
      <td>176.000001</td>
      <td>168.999999</td>
      <td>182.000001</td>
      <td>222.000000</td>
      <td>8.100000e+01</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>8.170825e-07</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>-8.785177e-07</td>
      <td>-5.893633e-07</td>
      <td>9.804652e-07</td>
      <td>0.000001</td>
      <td>165.999999</td>
      <td>243.000000</td>
      <td>233.000002</td>
      <td>223.999995</td>
      <td>183.000001</td>
      <td>186.000002</td>
      <td>1.800000e+02</td>
      <td>191.999998</td>
      <td>187.000002</td>
      <td>191.000000</td>
      <td>189.000001</td>
      <td>232.000000</td>
      <td>119.000000</td>
      <td>-0.000006</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-1.907800e-08</td>
      <td>-6.678754e-07</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>5.449854e-07</td>
      <td>6.690875e-07</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>129.000000</td>
      <td>230.000005</td>
      <td>231.000002</td>
      <td>233.999995</td>
      <td>222.000001</td>
      <td>215.000002</td>
      <td>223.999995</td>
      <td>215.000001</td>
      <td>216.000002</td>
      <td>225.000002</td>
      <td>221.000001</td>
      <td>255.000004</td>
      <td>4.500000e+01</td>
      <td>-0.000003</td>
      <td>-0.000002</td>
      <td>1.135173e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>-8.247956e-07</td>
      <td>2.124748e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>108.000000</td>
      <td>221.000000</td>
      <td>213.000004</td>
      <td>223.000003</td>
      <td>215.000002</td>
      <td>220.999997</td>
      <td>230.999996</td>
      <td>216.999998</td>
      <td>230.000001</td>
      <td>188.000000</td>
      <td>222.000000</td>
      <td>239.000005</td>
      <td>15.000004</td>
      <td>-0.000002</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>-2.314611e-07</td>
      <td>-0.000002</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>71.000000</td>
      <td>250.999996</td>
      <td>229.000000</td>
      <td>189.999998</td>
      <td>192.000001</td>
      <td>200.999998</td>
      <td>1.990000e+02</td>
      <td>195.000001</td>
      <td>199.999999</td>
      <td>200.000002</td>
      <td>214.000000</td>
      <td>216.000002</td>
      <td>-0.000003</td>
      <td>-0.000005</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-7.203892e-07</td>
      <td>0.000001</td>
      <td>36.000001</td>
      <td>235.999998</td>
      <td>191.000001</td>
      <td>186.999999</td>
      <td>182.999999</td>
      <td>176.000000</td>
      <td>174.000000</td>
      <td>174.000001</td>
      <td>176.999999</td>
      <td>179.999999</td>
      <td>215.999998</td>
      <td>194.000001</td>
      <td>5.151848e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>0.000004</td>
      <td>0.000002</td>
      <td>3.101954e-07</td>
      <td>-0.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>0.000002</td>
      <td>-8.765240e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>10.999995</td>
      <td>231.999995</td>
      <td>191.999999</td>
      <td>189.000001</td>
      <td>185.999999</td>
      <td>1.880000e+02</td>
      <td>185.999999</td>
      <td>183.000000</td>
      <td>186.000000</td>
      <td>182.999999</td>
      <td>213.000002</td>
      <td>191.999998</td>
      <td>0.000001</td>
      <td>7.423447e-07</td>
      <td>1.327465e-07</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>3.281200e-07</td>
      <td>-0.000002</td>
      <td>-1.307099e-07</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>7.308653e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-5.223541e-09</td>
      <td>0.000004</td>
      <td>1.999999</td>
      <td>2.340000e+02</td>
      <td>195.999998</td>
      <td>187.000000</td>
      <td>185.000001</td>
      <td>180.999999</td>
      <td>180.999998</td>
      <td>181.000001</td>
      <td>182.000001</td>
      <td>183.000001</td>
      <td>206.000000</td>
      <td>191.000001</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>4.826824e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>3.892250e-07</td>
      <td>6.972123e-07</td>
      <td>-2.291508e-07</td>
      <td>-2.135472e-07</td>
      <td>-4.984975e-07</td>
      <td>-8.258425e-07</td>
      <td>0.000001</td>
      <td>-5.332055e-07</td>
      <td>5.139264e-07</td>
      <td>-0.000002</td>
      <td>-0.000004</td>
      <td>40.999999</td>
      <td>225.999998</td>
      <td>183.000001</td>
      <td>189.000000</td>
      <td>184.999999</td>
      <td>181.000000</td>
      <td>182.000001</td>
      <td>181.000001</td>
      <td>181.000001</td>
      <td>185.000000</td>
      <td>188.999999</td>
      <td>2.230000e+02</td>
      <td>11.000000</td>
      <td>-4.653743e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-7.986391e-07</td>
      <td>-4.286650e-07</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>-5.423990e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>186.000001</td>
      <td>207.999998</td>
      <td>181.000001</td>
      <td>186.000000</td>
      <td>181.999999</td>
      <td>181.000000</td>
      <td>181.000001</td>
      <td>1.800000e+02</td>
      <td>182.000000</td>
      <td>179.000000</td>
      <td>180.000000</td>
      <td>2.120000e+02</td>
      <td>1.350000e+02</td>
      <td>-6.241514e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>5.651978e-07</td>
      <td>-5.163183e-07</td>
      <td>0.000002</td>
      <td>-1.174552e-07</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>-0.000001</td>
      <td>5.352320e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000005</td>
      <td>144.000000</td>
      <td>220.999998</td>
      <td>182.000001</td>
      <td>183.000000</td>
      <td>180.000001</td>
      <td>181.999999</td>
      <td>178.999999</td>
      <td>179.999999</td>
      <td>176.999999</td>
      <td>188.000001</td>
      <td>222.999998</td>
      <td>151.999999</td>
      <td>0.999998</td>
      <td>0.000004</td>
      <td>-0.000005</td>
      <td>-2.677840e-07</td>
      <td>-3.174695e-07</td>
      <td>-0.000002</td>
      <td>-5.629809e-07</td>
      <td>-3.012789e-07</td>
      <td>-1.447285e-07</td>
      <td>-6.046403e-07</td>
      <td>1.118729e-07</td>
      <td>-9.513843e-07</td>
      <td>-2.911069e-07</td>
      <td>2.894738e-08</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>-3.351748e-07</td>
      <td>93.000001</td>
      <td>225.999998</td>
      <td>185.000000</td>
      <td>181.000001</td>
      <td>180.999999</td>
      <td>180.000001</td>
      <td>1.770000e+02</td>
      <td>191.000000</td>
      <td>213.000000</td>
      <td>67.000001</td>
      <td>0.000005</td>
      <td>0.000003</td>
      <td>0.000005</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>9.265024e-07</td>
      <td>0.000002</td>
      <td>4.011770e-07</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>-2.624320e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000003</td>
      <td>3.360567e-07</td>
      <td>0.000004</td>
      <td>9.999977e-01</td>
      <td>2.258955e-07</td>
      <td>84.000001</td>
      <td>226.000001</td>
      <td>180.000000</td>
      <td>182.000001</td>
      <td>181.999998</td>
      <td>177.999999</td>
      <td>226.000001</td>
      <td>43.999998</td>
      <td>-0.000004</td>
      <td>1.000000</td>
      <td>-0.000001</td>
      <td>-2.610268e-07</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>7.528400e-07</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>-2.873870e-07</td>
      <td>5.500636e-07</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>6.275720e-07</td>
      <td>4.274649e-07</td>
      <td>0.000001</td>
      <td>5.958969e-07</td>
      <td>1.540000e+02</td>
      <td>213.000001</td>
      <td>178.000001</td>
      <td>172.999998</td>
      <td>216.000001</td>
      <td>1.180000e+02</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.999996</td>
      <td>6.923674e-07</td>
      <td>0.000001</td>
      <td>-8.325330e-07</td>
      <td>-5.767398e-07</td>
      <td>-1.849482e-07</td>
      <td>0.000002</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>4.654391e-07</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>-4.551377e-07</td>
      <td>7.378563e-07</td>
      <td>-0.000002</td>
      <td>1.000002e+00</td>
      <td>0.000003</td>
      <td>29.000002</td>
      <td>221.999998</td>
      <td>1.800000e+02</td>
      <td>182.000000</td>
      <td>223.000000</td>
      <td>1.000002e+00</td>
      <td>0.000005</td>
      <td>-0.000004</td>
      <td>-0.000004</td>
      <td>1.000000e+00</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>7.859119e-07</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>-6.257008e-07</td>
      <td>0.000001</td>
      <td>9.323434e-07</td>
      <td>0.000002</td>
      <td>8.744090e-07</td>
      <td>0.000002</td>
      <td>1.000005</td>
      <td>0.000003</td>
      <td>0.000005</td>
      <td>197.999999</td>
      <td>187.000003</td>
      <td>194.000002</td>
      <td>1.530000e+02</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>1.000002</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>3.297868e-07</td>
      <td>7.765040e-07</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>-8.448512e-07</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>8.809801e-07</td>
      <td>1.723858e-07</td>
      <td>0.000001</td>
      <td>-7.570235e-07</td>
      <td>0.000001</td>
      <td>135.000000</td>
      <td>196.999999</td>
      <td>199.000002</td>
      <td>8.800000e+01</td>
      <td>-0.000005</td>
      <td>1.000000</td>
      <td>0.000005</td>
      <td>-1.852046e-07</td>
      <td>1.000000e+00</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-3.723668e-07</td>
      <td>-0.000002</td>
      <td>7.009453e-07</td>
      <td>0.999999</td>
      <td>3.933536e-07</td>
      <td>58.999999</td>
      <td>199.000005</td>
      <td>197.000002</td>
      <td>51.000001</td>
      <td>0.000002</td>
      <td>0.999998</td>
      <td>-5.581073e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>2.139088e-07</td>
      <td>0.000002</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>-8.592667e-07</td>
      <td>0.000002</td>
      <td>-7.577064e-07</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>2.000002</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>201.000001</td>
      <td>223.999998</td>
      <td>26.999997</td>
      <td>-0.000003</td>
      <td>2.999998</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>3.743793e-07</td>
      <td>-6.565379e-07</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>-2.304249e-08</td>
      <td>1.000000e+00</td>
      <td>-3.426212e-07</td>
      <td>-6.002562e-07</td>
      <td>9.900000e+01</td>
      <td>1.280000e+02</td>
      <td>-0.000002</td>
      <td>6.155732e-07</td>
      <td>1.000001e+00</td>
      <td>0.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>8.642878e-08</td>
      <td>-4.371375e-07</td>
      <td>9.682471e-07</td>
      <td>-0.000002</td>
      <td>-5.505288e-07</td>
      <td>4.053924e-07</td>
      <td>4.035032e-07</td>
      <td>2.956210e-07</td>
      <td>4.005417e-07</td>
      <td>-0.000001</td>
      <td>5.516976e-08</td>
      <td>-2.068072e-07</td>
      <td>-9.467687e-08</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>-8.031146e-08</td>
      <td>1.985495e-07</td>
      <td>-1.276844e-07</td>
      <td>-0.000001</td>
      <td>7.044741e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-7.585758e-07</td>
      <td>-0.000006</td>
      <td>0.000006</td>
      <td>0.000002</td>
      <td>-1.279166e-07</td>
      <td>0.000001</td>
      <td>-9.767781e-07</td>
      <td>-6.219337e-07</td>
      <td>1.444539e-07</td>
      <td>-1.736492e-07</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>2.055711e-07</td>
      <td>4.611180e-07</td>
      <td>3.908779e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>1.000000</td>
      <td>-0.000003</td>
      <td>0.000003</td>
      <td>1.290000e+02</td>
      <td>197.000002</td>
      <td>181.000000</td>
      <td>207.000005</td>
      <td>2.370000e+02</td>
      <td>219.000005</td>
      <td>180.999997</td>
      <td>1.610000e+02</td>
      <td>1.110000e+02</td>
      <td>9.700000e+01</td>
      <td>7.500000e+01</td>
      <td>4.400000e+01</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>7.656082e-07</td>
      <td>-3.514349e-07</td>
      <td>-9.688781e-07</td>
      <td>8.386749e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>2.000005</td>
      <td>-0.000003</td>
      <td>-0.000003</td>
      <td>203.999995</td>
      <td>214.000003</td>
      <td>1.880000e+02</td>
      <td>2.080000e+02</td>
      <td>247.000002</td>
      <td>217.000003</td>
      <td>2.300000e+02</td>
      <td>2.220000e+02</td>
      <td>2.220000e+02</td>
      <td>2.400000e+02</td>
      <td>2.170000e+02</td>
      <td>2.520000e+02</td>
      <td>9.000000e+00</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>2.273738e-07</td>
      <td>6.555034e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>2.999999</td>
      <td>0.000002</td>
      <td>9.626116e-07</td>
      <td>2.170000e+02</td>
      <td>206.000004</td>
      <td>209.000005</td>
      <td>213.000000</td>
      <td>2.010000e+02</td>
      <td>174.000002</td>
      <td>2.000000e+02</td>
      <td>1.990000e+02</td>
      <td>1.940000e+02</td>
      <td>2.030000e+02</td>
      <td>2.010000e+02</td>
      <td>1.860000e+02</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>-0.000005</td>
      <td>-0.000001</td>
      <td>3.999999e+00</td>
      <td>-0.000001</td>
      <td>3.999997</td>
      <td>193.000001</td>
      <td>200.000002</td>
      <td>201.000002</td>
      <td>197.000000</td>
      <td>199.000002</td>
      <td>185.000003</td>
      <td>2.300000e+02</td>
      <td>189.999995</td>
      <td>1.960000e+02</td>
      <td>1.980000e+02</td>
      <td>2.100000e+02</td>
      <td>2.060000e+02</td>
      <td>1.880000e+02</td>
      <td>2.200000e+01</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>-2.410009e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-8.460442e-08</td>
      <td>0.000003</td>
      <td>-9.104431e-07</td>
      <td>4.999997</td>
      <td>0.000005</td>
      <td>13.000001</td>
      <td>194.999999</td>
      <td>198.999998</td>
      <td>202.000002</td>
      <td>219.000000</td>
      <td>229.999995</td>
      <td>197.000001</td>
      <td>198.000001</td>
      <td>1.650000e+02</td>
      <td>218.000001</td>
      <td>2.160000e+02</td>
      <td>2.120000e+02</td>
      <td>2.210000e+02</td>
      <td>2.250000e+02</td>
      <td>8.100000e+01</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.715096e-07</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-0.000004</td>
      <td>1.262333e-07</td>
      <td>4.999995</td>
      <td>-4.162347e-07</td>
      <td>14.999996</td>
      <td>237.000002</td>
      <td>200.000002</td>
      <td>196.000000</td>
      <td>190.000000</td>
      <td>192.999998</td>
      <td>253.000005</td>
      <td>223.999996</td>
      <td>223.000005</td>
      <td>226.000000</td>
      <td>216.999997</td>
      <td>2.100000e+02</td>
      <td>1.840000e+02</td>
      <td>1.900000e+01</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>4.334051e-07</td>
      <td>-0.000002</td>
      <td>7.406876e-07</td>
      <td>-0.000003</td>
      <td>-9.455209e-07</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>5.000001</td>
      <td>-0.000002</td>
      <td>14.000000</td>
      <td>223.999995</td>
      <td>201.000001</td>
      <td>205.999998</td>
      <td>215.999998</td>
      <td>186.000002</td>
      <td>188.000000</td>
      <td>2.120000e+02</td>
      <td>192.999997</td>
      <td>194.000003</td>
      <td>191.999997</td>
      <td>219.999991</td>
      <td>1.210000e+02</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>-8.785177e-07</td>
      <td>-5.893633e-07</td>
      <td>9.804652e-07</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>-0.000005</td>
      <td>-0.000004</td>
      <td>2.999995</td>
      <td>-0.000002</td>
      <td>14.000003</td>
      <td>2.260000e+02</td>
      <td>199.000000</td>
      <td>209.000001</td>
      <td>214.999999</td>
      <td>213.000000</td>
      <td>199.000001</td>
      <td>200.000000</td>
      <td>202.999995</td>
      <td>203.000000</td>
      <td>189.999996</td>
      <td>2.170000e+02</td>
      <td>1.420000e+02</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>5.449854e-07</td>
      <td>6.690875e-07</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>1.000002</td>
      <td>-0.000004</td>
      <td>4.999998</td>
      <td>211.000002</td>
      <td>210.999999</td>
      <td>211.999998</td>
      <td>197.000001</td>
      <td>204.000001</td>
      <td>203.000002</td>
      <td>2.000000e+02</td>
      <td>201.999999</td>
      <td>200.000002</td>
      <td>1.920000e+02</td>
      <td>202.000001</td>
      <td>206.000000</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>-8.247956e-07</td>
      <td>2.124748e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>144.000000</td>
      <td>210.000002</td>
      <td>189.000001</td>
      <td>233.000002</td>
      <td>252.999998</td>
      <td>228.999998</td>
      <td>222.999998</td>
      <td>214.999998</td>
      <td>212.000005</td>
      <td>204.999999</td>
      <td>210.000003</td>
      <td>196.000004</td>
      <td>2.090000e+02</td>
      <td>62.000001</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>1.999995</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>222.999996</td>
      <td>1.890000e+02</td>
      <td>196.000002</td>
      <td>164.000000</td>
      <td>146.000000</td>
      <td>221.000000</td>
      <td>180.000002</td>
      <td>199.999999</td>
      <td>207.999997</td>
      <td>207.000001</td>
      <td>201.999999</td>
      <td>196.000002</td>
      <td>205.000000</td>
      <td>163.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-7.203892e-07</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000001</td>
      <td>4.000004</td>
      <td>0.000002</td>
      <td>49.999998</td>
      <td>233.000001</td>
      <td>178.999998</td>
      <td>218.000001</td>
      <td>190.999999</td>
      <td>155.000000</td>
      <td>205.000002</td>
      <td>217.000000</td>
      <td>2.270000e+02</td>
      <td>185.000000</td>
      <td>201.000000</td>
      <td>203.000004</td>
      <td>200.000001</td>
      <td>2.000000e+02</td>
      <td>200.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>0.000002</td>
      <td>-8.765240e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.999998</td>
      <td>0.000003</td>
      <td>-0.000001</td>
      <td>-0.000003</td>
      <td>154.000000</td>
      <td>2.110000e+02</td>
      <td>199.000000</td>
      <td>216.999998</td>
      <td>190.000001</td>
      <td>228.000001</td>
      <td>208.000000</td>
      <td>201.999998</td>
      <td>231.000001</td>
      <td>2.050000e+02</td>
      <td>1.880000e+02</td>
      <td>193.999997</td>
      <td>200.000003</td>
      <td>2.000000e+02</td>
      <td>212.000002</td>
      <td>9.000000e+00</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>9.999992e-01</td>
      <td>1.999999</td>
      <td>1.999998</td>
      <td>2.999998e+00</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>-1.976124e-07</td>
      <td>44.999999</td>
      <td>161.000000</td>
      <td>177.000000</td>
      <td>145.000000</td>
      <td>220.000000</td>
      <td>222.999999</td>
      <td>141.000000</td>
      <td>223.999999</td>
      <td>218.999999</td>
      <td>223.999996</td>
      <td>215.000002</td>
      <td>205.000002</td>
      <td>2.250000e+02</td>
      <td>206.999995</td>
      <td>207.999997</td>
      <td>1.970000e+02</td>
      <td>2.240000e+02</td>
      <td>3.900000e+01</td>
      <td>-2.135472e-07</td>
      <td>2.000001e+00</td>
      <td>2.999999e+00</td>
      <td>0.000001</td>
      <td>-5.332055e-07</td>
      <td>5.139264e-07</td>
      <td>-0.000002</td>
      <td>-0.000004</td>
      <td>-0.000002</td>
      <td>101.000001</td>
      <td>174.000000</td>
      <td>165.000000</td>
      <td>171.000001</td>
      <td>174.999999</td>
      <td>195.999998</td>
      <td>232.000004</td>
      <td>237.000001</td>
      <td>222.999999</td>
      <td>174.999999</td>
      <td>1.800000e+02</td>
      <td>171.000001</td>
      <td>1.960000e+02</td>
      <td>192.000003</td>
      <td>186.999997</td>
      <td>189.999995</td>
      <td>191.000004</td>
      <td>2.160000e+02</td>
      <td>1.700000e+01</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>-5.423990e-07</td>
      <td>6.000001</td>
      <td>2.000002</td>
      <td>29.000001</td>
      <td>84.000000</td>
      <td>150.999999</td>
      <td>147.000000</td>
      <td>113.999999</td>
      <td>164.000000</td>
      <td>196.000001</td>
      <td>211.999998</td>
      <td>207.000000</td>
      <td>2.040000e+02</td>
      <td>208.000000</td>
      <td>204.999999</td>
      <td>190.000001</td>
      <td>1.930000e+02</td>
      <td>1.960000e+02</td>
      <td>1.850000e+02</td>
      <td>180.000000</td>
      <td>182.000000</td>
      <td>1.840000e+02</td>
      <td>1.840000e+02</td>
      <td>216.000002</td>
      <td>1.400000e+01</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>101.000000</td>
      <td>1.340000e+02</td>
      <td>144.000002</td>
      <td>163.999999</td>
      <td>161.999999</td>
      <td>144.000000</td>
      <td>109.000000</td>
      <td>113.000000</td>
      <td>169.000000</td>
      <td>193.999998</td>
      <td>204.000002</td>
      <td>209.999999</td>
      <td>212.000002</td>
      <td>203.000001</td>
      <td>199.000001</td>
      <td>188.999999</td>
      <td>176.999999</td>
      <td>178.999998</td>
      <td>188.000002</td>
      <td>204.000003</td>
      <td>205.000005</td>
      <td>2.010000e+02</td>
      <td>1.990000e+02</td>
      <td>190.000001</td>
      <td>2.190000e+02</td>
      <td>1.600000e+01</td>
      <td>-1.447285e-07</td>
      <td>9.400000e+01</td>
      <td>1.930000e+02</td>
      <td>1.260000e+02</td>
      <td>6.400000e+01</td>
      <td>3.200000e+01</td>
      <td>11.000002</td>
      <td>90.000000</td>
      <td>1.520000e+02</td>
      <td>170.999999</td>
      <td>191.000000</td>
      <td>199.000000</td>
      <td>204.000002</td>
      <td>209.000002</td>
      <td>207.999998</td>
      <td>1.900000e+02</td>
      <td>186.999999</td>
      <td>180.000001</td>
      <td>214.000002</td>
      <td>233.000001</td>
      <td>204.000005</td>
      <td>182.000002</td>
      <td>176.999998</td>
      <td>175.000000</td>
      <td>178.000005</td>
      <td>1.760000e+02</td>
      <td>215.000007</td>
      <td>1.700000e+01</td>
      <td>2.200000e+01</td>
      <td>1.870000e+02</td>
      <td>2.160000e+02</td>
      <td>215.999999</td>
      <td>201.999998</td>
      <td>199.999997</td>
      <td>1.890000e+02</td>
      <td>195.000001</td>
      <td>1.960000e+02</td>
      <td>1.960000e+02</td>
      <td>195.999998</td>
      <td>199.999999</td>
      <td>204.000001</td>
      <td>199.999998</td>
      <td>204.999999</td>
      <td>196.000001</td>
      <td>215.999998</td>
      <td>255.000000</td>
      <td>143.000000</td>
      <td>90.000000</td>
      <td>190.000005</td>
      <td>1.630000e+02</td>
      <td>171.000000</td>
      <td>175.999995</td>
      <td>179.999997</td>
      <td>180.000004</td>
      <td>2.110000e+02</td>
      <td>1.900000e+01</td>
      <td>1.090000e+02</td>
      <td>1.960000e+02</td>
      <td>1.720000e+02</td>
      <td>1.960000e+02</td>
      <td>2.080000e+02</td>
      <td>214.999998</td>
      <td>222.999998</td>
      <td>2.100000e+02</td>
      <td>2.050000e+02</td>
      <td>202.999998</td>
      <td>2.010000e+02</td>
      <td>2.060000e+02</td>
      <td>210.999999</td>
      <td>214.000003</td>
      <td>200.000002</td>
      <td>219.000001</td>
      <td>1.870000e+02</td>
      <td>31.000004</td>
      <td>0.000001</td>
      <td>27.000001</td>
      <td>2.280000e+02</td>
      <td>165.000001</td>
      <td>1.790000e+02</td>
      <td>1.810000e+02</td>
      <td>1.800000e+02</td>
      <td>179.000000</td>
      <td>2.060000e+02</td>
      <td>2.900000e+01</td>
      <td>4.600000e+01</td>
      <td>1.900000e+02</td>
      <td>2.140000e+02</td>
      <td>1.950000e+02</td>
      <td>178.999996</td>
      <td>187.999999</td>
      <td>1.930000e+02</td>
      <td>2.020000e+02</td>
      <td>205.000001</td>
      <td>2.120000e+02</td>
      <td>210.000000</td>
      <td>209.000001</td>
      <td>199.999999</td>
      <td>1.830000e+02</td>
      <td>197.999999</td>
      <td>109.000000</td>
      <td>9.842208e-07</td>
      <td>0.000005</td>
      <td>-0.000004</td>
      <td>44.000002</td>
      <td>1.880000e+02</td>
      <td>168.000004</td>
      <td>176.999996</td>
      <td>164.999996</td>
      <td>1.750000e+02</td>
      <td>1.760000e+02</td>
      <td>1.760000e+02</td>
      <td>2.500000e+01</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>9.100000e+01</td>
      <td>1.930000e+02</td>
      <td>210.999997</td>
      <td>2.050000e+02</td>
      <td>202.000003</td>
      <td>2.010000e+02</td>
      <td>199.000001</td>
      <td>188.000001</td>
      <td>189.999997</td>
      <td>191.000001</td>
      <td>183.000000</td>
      <td>211.000004</td>
      <td>152.000000</td>
      <td>8.867231e-07</td>
      <td>0.000003</td>
      <td>4.999995</td>
      <td>0.000005</td>
      <td>12.999999</td>
      <td>198.999999</td>
      <td>190.000003</td>
      <td>192.999998</td>
      <td>1.970000e+02</td>
      <td>1.990000e+02</td>
      <td>2.030000e+02</td>
      <td>2.120000e+02</td>
      <td>5.900000e+01</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>2.200000e+01</td>
      <td>62.000000</td>
      <td>152.000000</td>
      <td>2.050000e+02</td>
      <td>1.860000e+02</td>
      <td>232.000002</td>
      <td>2.290000e+02</td>
      <td>182.999998</td>
      <td>214.999995</td>
      <td>164.000003</td>
      <td>7.999998</td>
      <td>8.076585e-07</td>
      <td>2.000003</td>
      <td>-0.000006</td>
      <td>0.000005</td>
      <td>9.999992e-01</td>
      <td>1.840000e+02</td>
      <td>137.000002</td>
      <td>155.000001</td>
      <td>165.000005</td>
      <td>136.000003</td>
      <td>1.290000e+02</td>
      <td>1.130000e+02</td>
      <td>1.800000e+01</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.000000e+00</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-3.723668e-07</td>
      <td>-0.000002</td>
      <td>7.009453e-07</td>
      <td>20.000000</td>
      <td>3.933536e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>-0.000005</td>
      <td>-5.581073e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>2.139088e-07</td>
      <td>0.000002</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>-8.592667e-07</td>
      <td>0.000002</td>
      <td>-7.577064e-07</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>3.743793e-07</td>
      <td>-6.565379e-07</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>-2.304249e-08</td>
      <td>9.522988e-07</td>
      <td>-3.426212e-07</td>
      <td>-6.002562e-07</td>
      <td>-3.102456e-07</td>
      <td>8.776177e-07</td>
      <td>-0.000002</td>
      <td>6.155732e-07</td>
      <td>-3.593068e-07</td>
      <td>0.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>8.642878e-08</td>
      <td>-4.371375e-07</td>
      <td>1.930000e+02</td>
      <td>222.000000</td>
      <td>2.050000e+02</td>
      <td>1.790000e+02</td>
      <td>1.970000e+02</td>
      <td>1.700000e+02</td>
      <td>1.770000e+02</td>
      <td>201.000007</td>
      <td>1.480000e+02</td>
      <td>-2.068072e-07</td>
      <td>-9.467687e-08</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>-8.031146e-08</td>
      <td>1.985495e-07</td>
      <td>-1.276844e-07</td>
      <td>-0.000001</td>
      <td>2.550000e+02</td>
      <td>236.999996</td>
      <td>240.999995</td>
      <td>2.390000e+02</td>
      <td>219.000004</td>
      <td>206.000000</td>
      <td>207.000000</td>
      <td>2.100000e+02</td>
      <td>221.000003</td>
      <td>-9.767781e-07</td>
      <td>-6.219337e-07</td>
      <td>1.444539e-07</td>
      <td>-1.736492e-07</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>2.055711e-07</td>
      <td>4.611180e-07</td>
      <td>3.908779e-07</td>
      <td>0.000002</td>
      <td>24.000000</td>
      <td>255.000005</td>
      <td>227.000002</td>
      <td>227.000003</td>
      <td>232.000001</td>
      <td>2.140000e+02</td>
      <td>202.999998</td>
      <td>204.999995</td>
      <td>181.000003</td>
      <td>2.250000e+02</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>7.057845e-07</td>
      <td>-1.318873e-07</td>
      <td>-1.672773e-07</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>7.656082e-07</td>
      <td>-3.514349e-07</td>
      <td>-9.688781e-07</td>
      <td>8.386749e-07</td>
      <td>93.000000</td>
      <td>254.999997</td>
      <td>214.000001</td>
      <td>212.999998</td>
      <td>211.000002</td>
      <td>219.999995</td>
      <td>214.000003</td>
      <td>2.320000e+02</td>
      <td>2.150000e+02</td>
      <td>235.999998</td>
      <td>66.000000</td>
      <td>5.257434e-07</td>
      <td>-3.949514e-07</td>
      <td>-1.805364e-07</td>
      <td>4.216997e-07</td>
      <td>-1.558985e-07</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>2.273738e-07</td>
      <td>6.555034e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>153.000000</td>
      <td>254.999998</td>
      <td>205.000000</td>
      <td>195.000002</td>
      <td>1.950000e+02</td>
      <td>2.180000e+02</td>
      <td>227.000002</td>
      <td>233.999997</td>
      <td>197.999999</td>
      <td>1.900000e+02</td>
      <td>162.000002</td>
      <td>5.942894e-07</td>
      <td>9.570876e-07</td>
      <td>7.753735e-07</td>
      <td>-8.642634e-07</td>
      <td>-4.548926e-07</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>178.000001</td>
      <td>253.000005</td>
      <td>1.980000e+02</td>
      <td>161.999999</td>
      <td>215.000002</td>
      <td>241.000000</td>
      <td>228.000000</td>
      <td>237.999996</td>
      <td>207.000001</td>
      <td>166.000002</td>
      <td>194.000003</td>
      <td>2.016850e-07</td>
      <td>-0.000002</td>
      <td>-4.557509e-07</td>
      <td>5.144796e-07</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>-2.410009e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-8.460442e-08</td>
      <td>188.999996</td>
      <td>2.510000e+02</td>
      <td>196.999998</td>
      <td>170.000000</td>
      <td>232.000002</td>
      <td>205.000002</td>
      <td>251.999998</td>
      <td>238.999999</td>
      <td>211.999997</td>
      <td>166.999999</td>
      <td>206.000000</td>
      <td>0.000005</td>
      <td>-7.583148e-07</td>
      <td>-0.000002</td>
      <td>8.919840e-07</td>
      <td>-6.656718e-07</td>
      <td>-5.450623e-07</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.715096e-07</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>188.000003</td>
      <td>2.490000e+02</td>
      <td>199.000002</td>
      <td>1.860000e+02</td>
      <td>255.000003</td>
      <td>79.000000</td>
      <td>254.999997</td>
      <td>240.000002</td>
      <td>214.999997</td>
      <td>177.000001</td>
      <td>201.000002</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-7.762443e-07</td>
      <td>-6.404821e-07</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>4.334051e-07</td>
      <td>-0.000002</td>
      <td>7.406876e-07</td>
      <td>-0.000003</td>
      <td>-9.455209e-07</td>
      <td>184.999998</td>
      <td>246.999997</td>
      <td>204.000001</td>
      <td>197.999999</td>
      <td>255.000000</td>
      <td>27.000001</td>
      <td>246.999998</td>
      <td>245.999998</td>
      <td>223.000005</td>
      <td>192.999999</td>
      <td>188.999998</td>
      <td>3.125923e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>8.170825e-07</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>-8.785177e-07</td>
      <td>-5.893633e-07</td>
      <td>9.804652e-07</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>155.000002</td>
      <td>253.999999</td>
      <td>206.000004</td>
      <td>209.000003</td>
      <td>255.000000</td>
      <td>-6.622733e-07</td>
      <td>230.999997</td>
      <td>255.000000</td>
      <td>221.000004</td>
      <td>213.000000</td>
      <td>171.999998</td>
      <td>-0.000002</td>
      <td>-0.000006</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-1.907800e-08</td>
      <td>-6.678754e-07</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>5.449854e-07</td>
      <td>6.690875e-07</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>99.000000</td>
      <td>254.999997</td>
      <td>205.999997</td>
      <td>218.999996</td>
      <td>252.000002</td>
      <td>-0.000003</td>
      <td>179.000000</td>
      <td>255.000001</td>
      <td>219.000002</td>
      <td>227.000003</td>
      <td>148.000000</td>
      <td>8.404825e-07</td>
      <td>-0.000003</td>
      <td>-0.000002</td>
      <td>1.135173e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>-8.247956e-07</td>
      <td>2.124748e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>43.000000</td>
      <td>254.999995</td>
      <td>209.000002</td>
      <td>217.999997</td>
      <td>227.999996</td>
      <td>0.000004</td>
      <td>96.000000</td>
      <td>254.999999</td>
      <td>220.000001</td>
      <td>243.000002</td>
      <td>127.000000</td>
      <td>-0.000005</td>
      <td>-0.000002</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>-2.314611e-07</td>
      <td>-0.000002</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>251.000003</td>
      <td>217.000004</td>
      <td>221.000001</td>
      <td>187.999999</td>
      <td>-5.688553e-07</td>
      <td>47.000001</td>
      <td>255.000002</td>
      <td>220.000001</td>
      <td>251.999996</td>
      <td>103.999999</td>
      <td>-0.000003</td>
      <td>-0.000005</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-7.203892e-07</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000001</td>
      <td>218.000002</td>
      <td>222.999999</td>
      <td>225.000003</td>
      <td>150.000000</td>
      <td>0.000002</td>
      <td>26.999997</td>
      <td>255.000003</td>
      <td>216.000001</td>
      <td>253.999997</td>
      <td>76.000002</td>
      <td>5.151848e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>0.000004</td>
      <td>0.000002</td>
      <td>3.101954e-07</td>
      <td>-0.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>0.000002</td>
      <td>-8.765240e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>-0.000001</td>
      <td>0.000003</td>
      <td>169.000001</td>
      <td>229.000002</td>
      <td>230.000000</td>
      <td>1.200000e+02</td>
      <td>-0.000004</td>
      <td>5.999993</td>
      <td>255.000003</td>
      <td>218.000002</td>
      <td>250.999996</td>
      <td>59.000000</td>
      <td>0.000001</td>
      <td>7.423447e-07</td>
      <td>1.327465e-07</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>3.281200e-07</td>
      <td>-0.000002</td>
      <td>-1.307099e-07</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>7.308653e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-5.223541e-09</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>-1.976124e-07</td>
      <td>138.000000</td>
      <td>233.999999</td>
      <td>228.999997</td>
      <td>94.999998</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>255.000000</td>
      <td>221.000001</td>
      <td>221.999998</td>
      <td>63.000001</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>4.826824e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>3.892250e-07</td>
      <td>6.972123e-07</td>
      <td>-2.291508e-07</td>
      <td>-2.135472e-07</td>
      <td>-4.984975e-07</td>
      <td>-8.258425e-07</td>
      <td>0.000001</td>
      <td>-5.332055e-07</td>
      <td>5.139264e-07</td>
      <td>-0.000002</td>
      <td>-0.000004</td>
      <td>-0.000002</td>
      <td>-0.000003</td>
      <td>146.000000</td>
      <td>232.000000</td>
      <td>232.000002</td>
      <td>96.000000</td>
      <td>-0.000004</td>
      <td>-0.000006</td>
      <td>242.000001</td>
      <td>226.999999</td>
      <td>222.000000</td>
      <td>6.300000e+01</td>
      <td>-0.000005</td>
      <td>-4.653743e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-7.986391e-07</td>
      <td>-4.286650e-07</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>-5.423990e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-0.000005</td>
      <td>119.000000</td>
      <td>232.000000</td>
      <td>227.999997</td>
      <td>75.000000</td>
      <td>-0.000001</td>
      <td>-5.077659e-07</td>
      <td>208.000000</td>
      <td>232.000004</td>
      <td>212.999999</td>
      <td>2.900000e+01</td>
      <td>3.296967e-07</td>
      <td>-6.241514e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>5.651978e-07</td>
      <td>-5.163183e-07</td>
      <td>0.000002</td>
      <td>-1.174552e-07</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>-0.000001</td>
      <td>5.352320e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000005</td>
      <td>-0.000004</td>
      <td>0.000001</td>
      <td>139.000000</td>
      <td>235.000003</td>
      <td>214.999998</td>
      <td>18.000001</td>
      <td>0.000003</td>
      <td>-0.000008</td>
      <td>169.000000</td>
      <td>234.999997</td>
      <td>211.999998</td>
      <td>12.000004</td>
      <td>-0.000003</td>
      <td>0.000004</td>
      <td>-0.000005</td>
      <td>-2.677840e-07</td>
      <td>-3.174695e-07</td>
      <td>-0.000002</td>
      <td>-5.629809e-07</td>
      <td>-3.012789e-07</td>
      <td>-1.447285e-07</td>
      <td>-6.046403e-07</td>
      <td>1.118729e-07</td>
      <td>-9.513843e-07</td>
      <td>-2.911069e-07</td>
      <td>2.894738e-08</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>-3.351748e-07</td>
      <td>-0.000003</td>
      <td>158.000000</td>
      <td>236.000003</td>
      <td>213.999998</td>
      <td>14.000003</td>
      <td>-0.000002</td>
      <td>-9.949699e-07</td>
      <td>176.000001</td>
      <td>234.999997</td>
      <td>215.000001</td>
      <td>32.000002</td>
      <td>0.000003</td>
      <td>0.000005</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>9.265024e-07</td>
      <td>0.000002</td>
      <td>4.011770e-07</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>-2.624320e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000003</td>
      <td>3.360567e-07</td>
      <td>0.000004</td>
      <td>1.282302e-07</td>
      <td>2.258955e-07</td>
      <td>152.000000</td>
      <td>232.000001</td>
      <td>227.000003</td>
      <td>66.000000</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>203.000000</td>
      <td>234.999997</td>
      <td>217.000002</td>
      <td>35.000002</td>
      <td>-0.000001</td>
      <td>-2.610268e-07</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>7.528400e-07</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>-2.873870e-07</td>
      <td>5.500636e-07</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>6.275720e-07</td>
      <td>4.274649e-07</td>
      <td>0.000001</td>
      <td>1.450000e+02</td>
      <td>2.340000e+02</td>
      <td>229.999998</td>
      <td>90.000000</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>2.080000e+02</td>
      <td>235.000001</td>
      <td>212.999995</td>
      <td>8.999999</td>
      <td>6.923674e-07</td>
      <td>0.000001</td>
      <td>-8.325330e-07</td>
      <td>-5.767398e-07</td>
      <td>-1.849482e-07</td>
      <td>0.000002</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>4.654391e-07</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>-4.551377e-07</td>
      <td>7.378563e-07</td>
      <td>-0.000002</td>
      <td>3.950150e-07</td>
      <td>119.000000</td>
      <td>236.999998</td>
      <td>229.000001</td>
      <td>7.600000e+01</td>
      <td>0.000003</td>
      <td>-0.000005</td>
      <td>1.670000e+02</td>
      <td>241.000000</td>
      <td>208.000003</td>
      <td>-0.000004</td>
      <td>4.021377e-07</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>7.859119e-07</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>-6.257008e-07</td>
      <td>0.000001</td>
      <td>9.323434e-07</td>
      <td>0.000002</td>
      <td>8.744090e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>76.000002</td>
      <td>239.000004</td>
      <td>218.999999</td>
      <td>19.000002</td>
      <td>0.000005</td>
      <td>8.867231e-07</td>
      <td>93.000000</td>
      <td>244.999997</td>
      <td>206.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>3.297868e-07</td>
      <td>7.765040e-07</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>-8.448512e-07</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>8.809801e-07</td>
      <td>1.723858e-07</td>
      <td>0.000001</td>
      <td>4.900000e+01</td>
      <td>239.000004</td>
      <td>203.000003</td>
      <td>-0.000001</td>
      <td>-0.000001</td>
      <td>8.076585e-07</td>
      <td>31.000002</td>
      <td>238.000000</td>
      <td>205.000002</td>
      <td>-1.852046e-07</td>
      <td>6.976350e-08</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-3.723668e-07</td>
      <td>-0.000002</td>
      <td>7.009453e-07</td>
      <td>40.000001</td>
      <td>2.390000e+02</td>
      <td>191.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>18.000001</td>
      <td>228.000000</td>
      <td>2.010000e+02</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>2.139088e-07</td>
      <td>0.000002</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>-8.592667e-07</td>
      <td>0.000002</td>
      <td>-7.577064e-07</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>29.000000</td>
      <td>245.999998</td>
      <td>198.999995</td>
      <td>0.000002</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>9.000003</td>
      <td>239.000002</td>
      <td>219.999997</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>3.743793e-07</td>
      <td>-6.565379e-07</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>-2.304249e-08</td>
      <td>9.522988e-07</td>
      <td>1.600000e+02</td>
      <td>1.340000e+02</td>
      <td>-3.102456e-07</td>
      <td>8.776177e-07</td>
      <td>-0.000002</td>
      <td>6.155732e-07</td>
      <td>1.380000e+02</td>
      <td>128.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>8.642878e-08</td>
      <td>-4.371375e-07</td>
      <td>9.682471e-07</td>
      <td>38.000000</td>
      <td>3.000000e+01</td>
      <td>6.000002e+00</td>
      <td>1.600000e+01</td>
      <td>3.900000e+01</td>
      <td>2.500000e+01</td>
      <td>-0.000001</td>
      <td>5.516976e-08</td>
      <td>-2.068072e-07</td>
      <td>-9.467687e-08</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>-8.031146e-08</td>
      <td>1.985495e-07</td>
      <td>-1.276844e-07</td>
      <td>7.000000</td>
      <td>3.000000e+01</td>
      <td>68.000001</td>
      <td>64.999999</td>
      <td>7.500000e+01</td>
      <td>39.999998</td>
      <td>67.000000</td>
      <td>54.999999</td>
      <td>3.000000e+00</td>
      <td>0.000001</td>
      <td>-9.767781e-07</td>
      <td>-6.219337e-07</td>
      <td>1.444539e-07</td>
      <td>-1.736492e-07</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>2.055711e-07</td>
      <td>3.000000e+00</td>
      <td>3.900000e+01</td>
      <td>75.999999</td>
      <td>56.000000</td>
      <td>9.000000</td>
      <td>132.999999</td>
      <td>161.999998</td>
      <td>151.000001</td>
      <td>1.520000e+02</td>
      <td>164.000000</td>
      <td>72.999999</td>
      <td>22.999999</td>
      <td>4.800000e+01</td>
      <td>61.000000</td>
      <td>12.000000</td>
      <td>7.057845e-07</td>
      <td>-1.318873e-07</td>
      <td>1.000000e+00</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>1.600000e+01</td>
      <td>6.800000e+01</td>
      <td>6.300000e+01</td>
      <td>3.800000e+01</td>
      <td>24.999998</td>
      <td>0.000005</td>
      <td>109.000000</td>
      <td>241.000002</td>
      <td>240.000000</td>
      <td>255.000003</td>
      <td>117.000000</td>
      <td>2.000000e+01</td>
      <td>1.880000e+02</td>
      <td>47.000001</td>
      <td>30.000001</td>
      <td>5.400000e+01</td>
      <td>5.500000e+01</td>
      <td>-1.805364e-07</td>
      <td>4.216997e-07</td>
      <td>-1.558985e-07</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>2.400000e+01</td>
      <td>3.800000e+01</td>
      <td>20.000000</td>
      <td>54.000000</td>
      <td>58.999999</td>
      <td>100.000000</td>
      <td>0.000003</td>
      <td>138.000000</td>
      <td>1.840000e+02</td>
      <td>1.860000e+02</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>10.999999</td>
      <td>1.050000e+02</td>
      <td>42.000001</td>
      <td>1.900000e+01</td>
      <td>4.100000e+01</td>
      <td>3.000001e+00</td>
      <td>-8.642634e-07</td>
      <td>-4.548926e-07</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>41.000000</td>
      <td>33.000000</td>
      <td>35.000001</td>
      <td>59.000000</td>
      <td>41.999999</td>
      <td>22.999999</td>
      <td>9.999996e+00</td>
      <td>-0.000001</td>
      <td>101.000000</td>
      <td>56.000002</td>
      <td>0.000004</td>
      <td>19.000002</td>
      <td>8.000001</td>
      <td>34.999998</td>
      <td>41.000000</td>
      <td>1.600000e+01</td>
      <td>41.000001</td>
      <td>2.400000e+01</td>
      <td>5.144796e-07</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>1.000000e+00</td>
      <td>45.000000</td>
      <td>36.999999</td>
      <td>34.000001</td>
      <td>2.200000e+01</td>
      <td>24.000000</td>
      <td>2.600000e+01</td>
      <td>-0.000002</td>
      <td>0.000005</td>
      <td>-0.000001</td>
      <td>25.000001</td>
      <td>15.999998</td>
      <td>7.000003</td>
      <td>14.000005</td>
      <td>95.000000</td>
      <td>12.999997</td>
      <td>16.999999</td>
      <td>3.200000e+01</td>
      <td>34.000001</td>
      <td>8.919840e-07</td>
      <td>-6.656718e-07</td>
      <td>-5.450623e-07</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.000000e+00</td>
      <td>44.000000</td>
      <td>34.000000</td>
      <td>42.000000</td>
      <td>28.999999</td>
      <td>51.999999</td>
      <td>1.370000e+02</td>
      <td>136.999999</td>
      <td>1.110000e+02</td>
      <td>49.999997</td>
      <td>17.000002</td>
      <td>26.999997</td>
      <td>23.999996</td>
      <td>72.000000</td>
      <td>124.000000</td>
      <td>5.000002</td>
      <td>28.000002</td>
      <td>30.999998</td>
      <td>37.000000</td>
      <td>1.000000</td>
      <td>-7.762443e-07</td>
      <td>-6.404821e-07</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>1.400000e+01</td>
      <td>49.000000</td>
      <td>2.500000e+01</td>
      <td>48.999999</td>
      <td>3.900000e+01</td>
      <td>26.000002</td>
      <td>39.999999</td>
      <td>81.000000</td>
      <td>117.000000</td>
      <td>125.000000</td>
      <td>95.000000</td>
      <td>54.000002</td>
      <td>17.999999</td>
      <td>42.999996</td>
      <td>55.000001</td>
      <td>2.999996</td>
      <td>3.000000e+01</td>
      <td>26.999999</td>
      <td>33.000001</td>
      <td>9.000000</td>
      <td>0.000002</td>
      <td>8.170825e-07</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>2.600000e+01</td>
      <td>4.900000e+01</td>
      <td>8.000000e+00</td>
      <td>141.000003</td>
      <td>75.000000</td>
      <td>11.000000</td>
      <td>14.000002</td>
      <td>6.000004</td>
      <td>14.000001</td>
      <td>35.000000</td>
      <td>7.200000e+01</td>
      <td>93.999999</td>
      <td>13.000003</td>
      <td>72.999998</td>
      <td>8.999999</td>
      <td>81.000001</td>
      <td>126.000000</td>
      <td>4.999995</td>
      <td>34.000002</td>
      <td>19.000001</td>
      <td>-1.907800e-08</td>
      <td>-6.678754e-07</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>3.100000e+01</td>
      <td>5.100000e+01</td>
      <td>13.999998</td>
      <td>159.999997</td>
      <td>88.000000</td>
      <td>13.000001</td>
      <td>20.000000</td>
      <td>25.000000</td>
      <td>17.000003</td>
      <td>19.000002</td>
      <td>11.000003</td>
      <td>38.000001</td>
      <td>133.000000</td>
      <td>69.000003</td>
      <td>0.000005</td>
      <td>159.000000</td>
      <td>1.380000e+02</td>
      <td>1.000002</td>
      <td>33.000002</td>
      <td>2.600000e+01</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>3.400000e+01</td>
      <td>4.700000e+01</td>
      <td>45.000000</td>
      <td>99.999999</td>
      <td>23.000000</td>
      <td>71.000000</td>
      <td>31.000002</td>
      <td>17.999999</td>
      <td>41.999996</td>
      <td>75.999998</td>
      <td>13.000002</td>
      <td>23.999999</td>
      <td>82.000002</td>
      <td>59.000001</td>
      <td>22.000000</td>
      <td>37.999999</td>
      <td>82.999999</td>
      <td>12.999999</td>
      <td>32.999999</td>
      <td>28.999998</td>
      <td>-0.000001</td>
      <td>-2.314611e-07</td>
      <td>-0.000002</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>41.000000</td>
      <td>39.999999</td>
      <td>72.000000</td>
      <td>68.000000</td>
      <td>5.000001</td>
      <td>100.000000</td>
      <td>50.000000</td>
      <td>23.000004</td>
      <td>32.999996</td>
      <td>95.000000</td>
      <td>9.999981e-01</td>
      <td>67.999997</td>
      <td>62.999997</td>
      <td>66.999996</td>
      <td>24.999999</td>
      <td>16.000001</td>
      <td>93.999999</td>
      <td>22.000005</td>
      <td>33.999999</td>
      <td>33.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>42.000000</td>
      <td>42.000001</td>
      <td>8.500000e+01</td>
      <td>57.000001</td>
      <td>72.000001</td>
      <td>49.000002</td>
      <td>33.999996</td>
      <td>27.999999</td>
      <td>37.999997</td>
      <td>122.000000</td>
      <td>0.000002</td>
      <td>105.999999</td>
      <td>82.000001</td>
      <td>12.999995</td>
      <td>94.999998</td>
      <td>0.000004</td>
      <td>8.900000e+01</td>
      <td>35.999998</td>
      <td>38.000000</td>
      <td>32.999998</td>
      <td>3.999999</td>
      <td>3.101954e-07</td>
      <td>-0.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>7.000000e+00</td>
      <td>40.000000</td>
      <td>3.600000e+01</td>
      <td>97.000000</td>
      <td>54.000000</td>
      <td>45.999999</td>
      <td>27.000003</td>
      <td>36.000000</td>
      <td>29.999999</td>
      <td>40.999997</td>
      <td>1.110000e+02</td>
      <td>-0.000004</td>
      <td>119.000000</td>
      <td>56.000002</td>
      <td>3.000001</td>
      <td>79.000004</td>
      <td>46.000002</td>
      <td>62.999998</td>
      <td>6.800000e+01</td>
      <td>3.800000e+01</td>
      <td>24.999999</td>
      <td>9.000003</td>
      <td>3.281200e-07</td>
      <td>-0.000002</td>
      <td>-1.307099e-07</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>1.600000e+01</td>
      <td>36.999999</td>
      <td>35.000001</td>
      <td>1.140000e+02</td>
      <td>30.000001</td>
      <td>26.000002</td>
      <td>3.000000e+01</td>
      <td>32.999997</td>
      <td>22.000002</td>
      <td>37.000002</td>
      <td>110.000000</td>
      <td>0.000001</td>
      <td>125.999999</td>
      <td>32.999996</td>
      <td>16.000001</td>
      <td>29.000002</td>
      <td>53.000004</td>
      <td>55.999998</td>
      <td>90.999999</td>
      <td>3.700000e+01</td>
      <td>22.000000</td>
      <td>16.000000</td>
      <td>3.892250e-07</td>
      <td>6.972123e-07</td>
      <td>-2.291508e-07</td>
      <td>-2.135472e-07</td>
      <td>-4.984975e-07</td>
      <td>-8.258425e-07</td>
      <td>22.000000</td>
      <td>3.500000e+01</td>
      <td>3.900000e+01</td>
      <td>131.000000</td>
      <td>2.000004</td>
      <td>40.999999</td>
      <td>52.000000</td>
      <td>50.999997</td>
      <td>33.999998</td>
      <td>40.999997</td>
      <td>104.999999</td>
      <td>0.999995</td>
      <td>136.000000</td>
      <td>19.000007</td>
      <td>28.000002</td>
      <td>34.000002</td>
      <td>1.800000e+01</td>
      <td>35.999998</td>
      <td>1.160000e+02</td>
      <td>31.000001</td>
      <td>23.000002</td>
      <td>18.000002</td>
      <td>0.000001</td>
      <td>-7.986391e-07</td>
      <td>-4.286650e-07</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>2.200000e+01</td>
      <td>33.000000</td>
      <td>54.000001</td>
      <td>137.999999</td>
      <td>0.000005</td>
      <td>47.000002</td>
      <td>63.999998</td>
      <td>41.000000</td>
      <td>28.000001</td>
      <td>56.999998</td>
      <td>94.999998</td>
      <td>43.999996</td>
      <td>1.070000e+02</td>
      <td>38.000001</td>
      <td>45.000003</td>
      <td>42.999999</td>
      <td>2.600000e+01</td>
      <td>8.000003e+00</td>
      <td>1.440000e+02</td>
      <td>30.000003</td>
      <td>22.000002</td>
      <td>1.900000e+01</td>
      <td>-5.163183e-07</td>
      <td>0.000002</td>
      <td>-1.174552e-07</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>-0.000001</td>
      <td>2.700000e+01</td>
      <td>23.999998</td>
      <td>62.000001</td>
      <td>130.000001</td>
      <td>-0.000005</td>
      <td>22.999996</td>
      <td>50.999996</td>
      <td>43.000002</td>
      <td>35.000004</td>
      <td>42.000002</td>
      <td>19.999999</td>
      <td>58.999998</td>
      <td>18.000000</td>
      <td>39.999998</td>
      <td>42.000000</td>
      <td>24.999998</td>
      <td>26.000004</td>
      <td>-0.000003</td>
      <td>142.999999</td>
      <td>34.000001</td>
      <td>2.200000e+01</td>
      <td>2.400000e+01</td>
      <td>-0.000002</td>
      <td>-5.629809e-07</td>
      <td>-3.012789e-07</td>
      <td>-1.447285e-07</td>
      <td>-6.046403e-07</td>
      <td>1.118729e-07</td>
      <td>2.900000e+01</td>
      <td>2.400000e+01</td>
      <td>7.200000e+01</td>
      <td>109.999999</td>
      <td>2.000004</td>
      <td>5.800000e+01</td>
      <td>59.000000</td>
      <td>59.999996</td>
      <td>58.000004</td>
      <td>73.999996</td>
      <td>52.999996</td>
      <td>83.999998</td>
      <td>4.500000e+01</td>
      <td>52.000000</td>
      <td>55.999999</td>
      <td>37.000002</td>
      <td>39.999998</td>
      <td>6.000001</td>
      <td>120.000001</td>
      <td>41.000001</td>
      <td>17.999997</td>
      <td>20.000000</td>
      <td>9.265024e-07</td>
      <td>0.000002</td>
      <td>4.011770e-07</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>-2.624320e-07</td>
      <td>27.000000</td>
      <td>18.999999</td>
      <td>75.000000</td>
      <td>9.700000e+01</td>
      <td>0.000004</td>
      <td>4.500000e+01</td>
      <td>4.300000e+01</td>
      <td>60.000004</td>
      <td>62.999998</td>
      <td>77.000001</td>
      <td>58.000001</td>
      <td>120.000001</td>
      <td>98.999999</td>
      <td>29.000004</td>
      <td>47.999997</td>
      <td>36.000001</td>
      <td>29.000001</td>
      <td>6.999998</td>
      <td>1.180000e+02</td>
      <td>42.000001</td>
      <td>20.000002</td>
      <td>24.000000</td>
      <td>-0.000002</td>
      <td>7.528400e-07</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>6.200000e+01</td>
      <td>3.600000e+01</td>
      <td>64.000000</td>
      <td>81.000000</td>
      <td>6.275720e-07</td>
      <td>4.274649e-07</td>
      <td>0.000001</td>
      <td>5.958969e-07</td>
      <td>6.911157e-07</td>
      <td>-0.000003</td>
      <td>-0.000004</td>
      <td>9.999997</td>
      <td>1.999998</td>
      <td>-3.494023e-08</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000004</td>
      <td>6.923674e-07</td>
      <td>100.000000</td>
      <td>3.700000e+01</td>
      <td>5.800000e+01</td>
      <td>5.100000e+01</td>
      <td>0.000002</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>8.300000e+01</td>
      <td>35.000001</td>
      <td>58.000000</td>
      <td>9.100000e+01</td>
      <td>7.378563e-07</td>
      <td>-0.000002</td>
      <td>3.950150e-07</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>0.000004</td>
      <td>-4.699235e-07</td>
      <td>0.000003</td>
      <td>-0.000005</td>
      <td>9.842208e-07</td>
      <td>0.000005</td>
      <td>-0.000004</td>
      <td>-0.000004</td>
      <td>4.021377e-07</td>
      <td>100.000000</td>
      <td>27.999998</td>
      <td>70.000000</td>
      <td>6.000000e+01</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>8.100000e+01</td>
      <td>36.000000</td>
      <td>5.500000e+01</td>
      <td>95.000000</td>
      <td>8.744090e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.000003</td>
      <td>0.000005</td>
      <td>-0.000001</td>
      <td>0.000004</td>
      <td>1.000001</td>
      <td>8.867231e-07</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>96.000000</td>
      <td>29.999999</td>
      <td>7.500000e+01</td>
      <td>5.800000e+01</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>6.900000e+01</td>
      <td>4.100000e+01</td>
      <td>51.000000</td>
      <td>102.000001</td>
      <td>8.809801e-07</td>
      <td>1.723858e-07</td>
      <td>0.000001</td>
      <td>-7.570235e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>-0.000001</td>
      <td>8.076585e-07</td>
      <td>-0.000005</td>
      <td>-0.000006</td>
      <td>0.000005</td>
      <td>-1.852046e-07</td>
      <td>6.976350e-08</td>
      <td>88.000001</td>
      <td>28.000000</td>
      <td>73.999999</td>
      <td>52.999999</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>6.900000e+01</td>
      <td>38.000000</td>
      <td>46.000000</td>
      <td>106.000000</td>
      <td>-3.723668e-07</td>
      <td>-0.000002</td>
      <td>7.009453e-07</td>
      <td>-0.000002</td>
      <td>3.933536e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>-0.000005</td>
      <td>1.000001e+00</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>90.999999</td>
      <td>1.900000e+01</td>
      <td>79.999999</td>
      <td>5.300000e+01</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>7.600000e+01</td>
      <td>5.000000e+01</td>
      <td>6.000000e+01</td>
      <td>106.000001</td>
      <td>6.000001e+00</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>1.000001</td>
      <td>-0.000002</td>
      <td>3.000000</td>
      <td>99.000002</td>
      <td>43.000000</td>
      <td>99.999998</td>
      <td>6.000000e+01</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>3.800000e+01</td>
      <td>1.600000e+01</td>
      <td>3.743793e-07</td>
      <td>1.100000e+01</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>9.999998e-01</td>
      <td>9.522988e-07</td>
      <td>-3.426212e-07</td>
      <td>-6.002562e-07</td>
      <td>-3.102456e-07</td>
      <td>8.776177e-07</td>
      <td>-0.000002</td>
      <td>6.155732e-07</td>
      <td>-3.593068e-07</td>
      <td>0.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>3.000001e+00</td>
      <td>-6.984834e-07</td>
      <td>2.100000e+01</td>
      <td>9.000000e+00</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>8.642878e-08</td>
      <td>-4.371375e-07</td>
      <td>1.250000e+02</td>
      <td>114.000003</td>
      <td>1.090000e+02</td>
      <td>1.060000e+02</td>
      <td>1.110000e+02</td>
      <td>1.200000e+02</td>
      <td>1.130000e+02</td>
      <td>131.999999</td>
      <td>9.900000e+01</td>
      <td>-2.068072e-07</td>
      <td>-9.467687e-08</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>-8.031146e-08</td>
      <td>1.985495e-07</td>
      <td>-1.276844e-07</td>
      <td>4.000002</td>
      <td>2.550000e+02</td>
      <td>237.999997</td>
      <td>254.999999</td>
      <td>2.550000e+02</td>
      <td>255.000004</td>
      <td>255.000004</td>
      <td>255.000000</td>
      <td>2.460000e+02</td>
      <td>255.000007</td>
      <td>-9.767781e-07</td>
      <td>-6.219337e-07</td>
      <td>1.444539e-07</td>
      <td>-1.736492e-07</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>2.055711e-07</td>
      <td>4.611180e-07</td>
      <td>3.908779e-07</td>
      <td>0.000002</td>
      <td>110.000001</td>
      <td>236.999999</td>
      <td>217.999997</td>
      <td>218.000003</td>
      <td>219.999999</td>
      <td>2.190000e+02</td>
      <td>215.999998</td>
      <td>218.000001</td>
      <td>216.999996</td>
      <td>2.550000e+02</td>
      <td>21.999999</td>
      <td>-0.000002</td>
      <td>7.057845e-07</td>
      <td>-1.318873e-07</td>
      <td>-1.672773e-07</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>7.656082e-07</td>
      <td>-3.514349e-07</td>
      <td>-9.688781e-07</td>
      <td>8.386749e-07</td>
      <td>219.000000</td>
      <td>232.999996</td>
      <td>218.999999</td>
      <td>224.999995</td>
      <td>224.999998</td>
      <td>224.000003</td>
      <td>223.000003</td>
      <td>2.240000e+02</td>
      <td>2.150000e+02</td>
      <td>255.000000</td>
      <td>110.000001</td>
      <td>5.257434e-07</td>
      <td>-3.949514e-07</td>
      <td>-1.805364e-07</td>
      <td>4.216997e-07</td>
      <td>-1.558985e-07</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>2.273738e-07</td>
      <td>6.555034e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>253.999995</td>
      <td>225.000000</td>
      <td>217.000005</td>
      <td>220.000004</td>
      <td>2.220000e+02</td>
      <td>2.230000e+02</td>
      <td>220.999998</td>
      <td>223.000002</td>
      <td>211.000002</td>
      <td>2.520000e+02</td>
      <td>146.000002</td>
      <td>5.942894e-07</td>
      <td>9.570876e-07</td>
      <td>7.753735e-07</td>
      <td>-8.642634e-07</td>
      <td>-4.548926e-07</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>250.000000</td>
      <td>223.000005</td>
      <td>2.180000e+02</td>
      <td>220.000001</td>
      <td>220.999995</td>
      <td>218.999997</td>
      <td>221.000002</td>
      <td>219.000003</td>
      <td>216.000004</td>
      <td>246.000001</td>
      <td>133.000000</td>
      <td>2.016850e-07</td>
      <td>-0.000002</td>
      <td>-4.557509e-07</td>
      <td>5.144796e-07</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>-2.410009e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-8.460442e-08</td>
      <td>240.000000</td>
      <td>2.240000e+02</td>
      <td>220.000000</td>
      <td>221.999997</td>
      <td>226.000002</td>
      <td>227.000001</td>
      <td>218.999998</td>
      <td>221.000000</td>
      <td>215.000002</td>
      <td>243.000003</td>
      <td>140.999999</td>
      <td>0.000005</td>
      <td>-7.583148e-07</td>
      <td>-0.000002</td>
      <td>8.919840e-07</td>
      <td>-6.656718e-07</td>
      <td>-5.450623e-07</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.715096e-07</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>251.000000</td>
      <td>2.270000e+02</td>
      <td>217.000000</td>
      <td>2.250000e+02</td>
      <td>240.000004</td>
      <td>232.999997</td>
      <td>225.999998</td>
      <td>218.999998</td>
      <td>214.999997</td>
      <td>239.000004</td>
      <td>160.000002</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-7.762443e-07</td>
      <td>-6.404821e-07</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>4.334051e-07</td>
      <td>-0.000002</td>
      <td>7.406876e-07</td>
      <td>-0.000003</td>
      <td>-9.455209e-07</td>
      <td>254.000002</td>
      <td>220.000004</td>
      <td>215.999999</td>
      <td>230.000000</td>
      <td>220.999998</td>
      <td>130.000000</td>
      <td>255.000002</td>
      <td>215.999997</td>
      <td>216.999998</td>
      <td>248.999996</td>
      <td>131.000000</td>
      <td>3.125923e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>8.170825e-07</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>-8.785177e-07</td>
      <td>-5.893633e-07</td>
      <td>9.804652e-07</td>
      <td>0.000001</td>
      <td>22.999998</td>
      <td>255.000003</td>
      <td>215.999997</td>
      <td>211.000002</td>
      <td>252.000001</td>
      <td>175.999998</td>
      <td>2.300000e+01</td>
      <td>254.999998</td>
      <td>217.999999</td>
      <td>214.000001</td>
      <td>255.000002</td>
      <td>102.000000</td>
      <td>-0.000002</td>
      <td>-0.000006</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-1.907800e-08</td>
      <td>-6.678754e-07</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>5.449854e-07</td>
      <td>6.690875e-07</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>37.999998</td>
      <td>254.999999</td>
      <td>216.000001</td>
      <td>208.999997</td>
      <td>254.999999</td>
      <td>86.000000</td>
      <td>-0.000003</td>
      <td>255.000002</td>
      <td>222.000002</td>
      <td>217.999998</td>
      <td>255.000001</td>
      <td>74.999998</td>
      <td>8.404825e-07</td>
      <td>-0.000003</td>
      <td>-0.000002</td>
      <td>1.135173e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>-8.247956e-07</td>
      <td>2.124748e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>37.000002</td>
      <td>255.000002</td>
      <td>214.000003</td>
      <td>211.999997</td>
      <td>254.999996</td>
      <td>19.000001</td>
      <td>0.000004</td>
      <td>254.000001</td>
      <td>228.000004</td>
      <td>217.000001</td>
      <td>254.999998</td>
      <td>54.000001</td>
      <td>-0.000005</td>
      <td>-0.000002</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>-2.314611e-07</td>
      <td>-0.000002</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>33.000001</td>
      <td>255.000001</td>
      <td>212.999995</td>
      <td>214.000002</td>
      <td>254.999999</td>
      <td>-0.000002</td>
      <td>-5.688553e-07</td>
      <td>220.000002</td>
      <td>233.999997</td>
      <td>218.000000</td>
      <td>255.000004</td>
      <td>12.999999</td>
      <td>-0.000003</td>
      <td>-0.000005</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-7.203892e-07</td>
      <td>0.000001</td>
      <td>37.000002</td>
      <td>255.000000</td>
      <td>212.000002</td>
      <td>222.000002</td>
      <td>214.000000</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>186.999999</td>
      <td>236.000002</td>
      <td>219.999998</td>
      <td>246.000000</td>
      <td>0.000004</td>
      <td>5.151848e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>0.000004</td>
      <td>0.000002</td>
      <td>3.101954e-07</td>
      <td>-0.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>0.000002</td>
      <td>-8.765240e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>55.000001</td>
      <td>255.000002</td>
      <td>209.000000</td>
      <td>227.000000</td>
      <td>193.000001</td>
      <td>4.167304e-07</td>
      <td>-0.000004</td>
      <td>160.000000</td>
      <td>239.000001</td>
      <td>220.999999</td>
      <td>238.000001</td>
      <td>0.000005</td>
      <td>0.000001</td>
      <td>7.423447e-07</td>
      <td>1.327465e-07</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>3.281200e-07</td>
      <td>-0.000002</td>
      <td>-1.307099e-07</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>7.308653e-07</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-5.223541e-09</td>
      <td>0.000004</td>
      <td>41.000002</td>
      <td>2.550000e+02</td>
      <td>211.000001</td>
      <td>229.000003</td>
      <td>189.000001</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>140.000000</td>
      <td>239.999997</td>
      <td>222.999999</td>
      <td>233.000001</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>4.826824e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>3.892250e-07</td>
      <td>6.972123e-07</td>
      <td>-2.291508e-07</td>
      <td>-2.135472e-07</td>
      <td>-4.984975e-07</td>
      <td>-8.258425e-07</td>
      <td>0.000001</td>
      <td>-5.332055e-07</td>
      <td>5.139264e-07</td>
      <td>-0.000002</td>
      <td>-0.000004</td>
      <td>-0.000002</td>
      <td>254.000003</td>
      <td>216.000002</td>
      <td>227.999997</td>
      <td>215.000000</td>
      <td>-0.000004</td>
      <td>-0.000004</td>
      <td>113.000001</td>
      <td>241.000002</td>
      <td>223.999999</td>
      <td>221.000002</td>
      <td>-1.050891e-07</td>
      <td>-0.000005</td>
      <td>-4.653743e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-7.986391e-07</td>
      <td>-4.286650e-07</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>-5.423990e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>240.000004</td>
      <td>224.000001</td>
      <td>235.000004</td>
      <td>196.000001</td>
      <td>0.000004</td>
      <td>-0.000001</td>
      <td>7.200000e+01</td>
      <td>238.999997</td>
      <td>225.000001</td>
      <td>221.999998</td>
      <td>8.073550e-07</td>
      <td>3.296967e-07</td>
      <td>-6.241514e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>5.651978e-07</td>
      <td>-5.163183e-07</td>
      <td>0.000002</td>
      <td>-1.174552e-07</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>-0.000001</td>
      <td>5.352320e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000005</td>
      <td>-0.000004</td>
      <td>176.999999</td>
      <td>231.999996</td>
      <td>229.000000</td>
      <td>200.999998</td>
      <td>-0.000004</td>
      <td>0.000003</td>
      <td>48.999999</td>
      <td>234.000003</td>
      <td>228.000003</td>
      <td>226.000003</td>
      <td>-0.000003</td>
      <td>-0.000003</td>
      <td>0.000004</td>
      <td>-0.000005</td>
      <td>-2.677840e-07</td>
      <td>-3.174695e-07</td>
      <td>-0.000002</td>
      <td>-5.629809e-07</td>
      <td>-3.012789e-07</td>
      <td>-1.447285e-07</td>
      <td>-6.046403e-07</td>
      <td>1.118729e-07</td>
      <td>-9.513843e-07</td>
      <td>-2.911069e-07</td>
      <td>2.894738e-08</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>-3.351748e-07</td>
      <td>112.000001</td>
      <td>236.999999</td>
      <td>217.000000</td>
      <td>255.000002</td>
      <td>-0.000003</td>
      <td>-0.000002</td>
      <td>5.000000e+01</td>
      <td>233.999996</td>
      <td>229.999998</td>
      <td>226.000004</td>
      <td>0.000005</td>
      <td>0.000003</td>
      <td>0.000005</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>9.265024e-07</td>
      <td>0.000002</td>
      <td>4.011770e-07</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>-2.624320e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000003</td>
      <td>3.360567e-07</td>
      <td>0.000004</td>
      <td>1.282302e-07</td>
      <td>6.400000e+01</td>
      <td>238.000001</td>
      <td>213.000000</td>
      <td>232.000001</td>
      <td>31.000001</td>
      <td>-0.000002</td>
      <td>62.000000</td>
      <td>240.000004</td>
      <td>227.000000</td>
      <td>213.000000</td>
      <td>-0.000004</td>
      <td>-0.000001</td>
      <td>-2.610268e-07</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>7.528400e-07</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>-2.873870e-07</td>
      <td>5.500636e-07</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>6.275720e-07</td>
      <td>4.274649e-07</td>
      <td>28.000004</td>
      <td>2.290000e+02</td>
      <td>2.160000e+02</td>
      <td>236.000004</td>
      <td>71.000000</td>
      <td>-0.000002</td>
      <td>59.000001</td>
      <td>2.390000e+02</td>
      <td>227.999996</td>
      <td>211.000002</td>
      <td>0.000004</td>
      <td>6.923674e-07</td>
      <td>0.000001</td>
      <td>-8.325330e-07</td>
      <td>-5.767398e-07</td>
      <td>-1.849482e-07</td>
      <td>0.000002</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>4.654391e-07</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>-4.551377e-07</td>
      <td>7.378563e-07</td>
      <td>-0.000002</td>
      <td>3.950150e-07</td>
      <td>212.000002</td>
      <td>222.000000</td>
      <td>237.000004</td>
      <td>1.090000e+02</td>
      <td>0.000003</td>
      <td>51.000001</td>
      <td>2.370000e+02</td>
      <td>229.000001</td>
      <td>219.999996</td>
      <td>-0.000004</td>
      <td>4.021377e-07</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>7.859119e-07</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>-6.257008e-07</td>
      <td>0.000001</td>
      <td>9.323434e-07</td>
      <td>0.000002</td>
      <td>8.744090e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>207.999998</td>
      <td>227.999997</td>
      <td>236.000003</td>
      <td>158.000001</td>
      <td>0.000005</td>
      <td>5.600000e+01</td>
      <td>239.000004</td>
      <td>229.000003</td>
      <td>218.000000</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>3.297868e-07</td>
      <td>7.765040e-07</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>-8.448512e-07</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>8.809801e-07</td>
      <td>1.000002e+00</td>
      <td>0.000001</td>
      <td>1.510000e+02</td>
      <td>238.000000</td>
      <td>230.999997</td>
      <td>206.999996</td>
      <td>-0.000001</td>
      <td>5.100000e+01</td>
      <td>239.999999</td>
      <td>231.000004</td>
      <td>214.000003</td>
      <td>-1.852046e-07</td>
      <td>6.976350e-08</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-3.723668e-07</td>
      <td>3.000001</td>
      <td>7.009453e-07</td>
      <td>103.000000</td>
      <td>2.360000e+02</td>
      <td>225.000002</td>
      <td>238.000004</td>
      <td>-0.000001</td>
      <td>41.000001</td>
      <td>234.000001</td>
      <td>230.999997</td>
      <td>2.050000e+02</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>2.139088e-07</td>
      <td>0.000002</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>-8.592667e-07</td>
      <td>0.000002</td>
      <td>-7.577064e-07</td>
      <td>3.000000</td>
      <td>0.000001</td>
      <td>40.000002</td>
      <td>242.999999</td>
      <td>223.999995</td>
      <td>254.999998</td>
      <td>-0.000003</td>
      <td>34.999997</td>
      <td>247.000003</td>
      <td>245.999994</td>
      <td>200.000003</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>3.743793e-07</td>
      <td>-6.565379e-07</td>
      <td>6.560168e-08</td>
      <td>3.000001e+00</td>
      <td>-2.304249e-08</td>
      <td>9.522988e-07</td>
      <td>1.720000e+02</td>
      <td>2.240000e+02</td>
      <td>1.390000e+02</td>
      <td>8.776177e-07</td>
      <td>-0.000002</td>
      <td>1.050000e+02</td>
      <td>1.670000e+02</td>
      <td>27.000000</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>8.642878e-08</td>
      <td>-4.371375e-07</td>
      <td>9.682471e-07</td>
      <td>-0.000002</td>
      <td>-5.505288e-07</td>
      <td>4.053924e-07</td>
      <td>4.035032e-07</td>
      <td>2.956210e-07</td>
      <td>4.005417e-07</td>
      <td>-0.000001</td>
      <td>5.516976e-08</td>
      <td>-2.068072e-07</td>
      <td>-9.467687e-08</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>-8.031146e-08</td>
      <td>1.985495e-07</td>
      <td>-1.276844e-07</td>
      <td>-0.000001</td>
      <td>7.044741e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-7.585758e-07</td>
      <td>-0.000006</td>
      <td>0.000006</td>
      <td>0.000002</td>
      <td>-1.279166e-07</td>
      <td>0.000001</td>
      <td>-9.767781e-07</td>
      <td>-6.219337e-07</td>
      <td>1.444539e-07</td>
      <td>-1.736492e-07</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>2.055711e-07</td>
      <td>4.611180e-07</td>
      <td>3.908779e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>0.000003</td>
      <td>-4.594104e-07</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>0.000001</td>
      <td>-4.233165e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>7.057845e-07</td>
      <td>-1.318873e-07</td>
      <td>-1.672773e-07</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>7.656082e-07</td>
      <td>-3.514349e-07</td>
      <td>-9.688781e-07</td>
      <td>8.386749e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.000004</td>
      <td>-0.000003</td>
      <td>-0.000003</td>
      <td>-0.000003</td>
      <td>-0.000004</td>
      <td>-6.836428e-07</td>
      <td>-3.299791e-07</td>
      <td>-0.000005</td>
      <td>0.000002</td>
      <td>5.257434e-07</td>
      <td>-3.949514e-07</td>
      <td>-1.805364e-07</td>
      <td>4.216997e-07</td>
      <td>-1.558985e-07</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>2.273738e-07</td>
      <td>6.555034e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>9.626116e-07</td>
      <td>6.564152e-07</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>-0.000005</td>
      <td>2.960771e-07</td>
      <td>-0.000002</td>
      <td>5.942894e-07</td>
      <td>9.570876e-07</td>
      <td>7.753735e-07</td>
      <td>-8.642634e-07</td>
      <td>-4.548926e-07</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>-0.000005</td>
      <td>-0.000001</td>
      <td>6.513797e-07</td>
      <td>-0.000001</td>
      <td>0.000005</td>
      <td>-0.000004</td>
      <td>0.000004</td>
      <td>-0.000001</td>
      <td>0.000004</td>
      <td>0.000002</td>
      <td>0.000004</td>
      <td>2.016850e-07</td>
      <td>-0.000002</td>
      <td>-4.557509e-07</td>
      <td>5.144796e-07</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>1.030000e+02</td>
      <td>6.700000e+01</td>
      <td>6.900000e+01</td>
      <td>68.000000</td>
      <td>60.000000</td>
      <td>56.000000</td>
      <td>5.400000e+01</td>
      <td>49.999999</td>
      <td>5.100000e+01</td>
      <td>51.000002</td>
      <td>49.000003</td>
      <td>55.000002</td>
      <td>59.000002</td>
      <td>55.999998</td>
      <td>58.000001</td>
      <td>60.000001</td>
      <td>71.000001</td>
      <td>72.000001</td>
      <td>74.000000</td>
      <td>7.500000e+01</td>
      <td>81.000000</td>
      <td>8.300000e+01</td>
      <td>7.900000e+01</td>
      <td>8.100000e+01</td>
      <td>1.220000e+02</td>
      <td>1.100000e+01</td>
      <td>-2.890140e-08</td>
      <td>5.900000e+01</td>
      <td>2.370000e+02</td>
      <td>2.280000e+02</td>
      <td>2.270000e+02</td>
      <td>254.999999</td>
      <td>255.000005</td>
      <td>254.999999</td>
      <td>255.000005</td>
      <td>252.999998</td>
      <td>2.530000e+02</td>
      <td>252.999997</td>
      <td>2.550000e+02</td>
      <td>255.000003</td>
      <td>254.999997</td>
      <td>254.999997</td>
      <td>254.999999</td>
      <td>254.999995</td>
      <td>255.000000</td>
      <td>254.999999</td>
      <td>254.999996</td>
      <td>254.999998</td>
      <td>254.999999</td>
      <td>255.000001</td>
      <td>2.550000e+02</td>
      <td>2.290000e+02</td>
      <td>2.480000e+02</td>
      <td>1.250000e+02</td>
      <td>2.976238e-08</td>
      <td>1.230000e+02</td>
      <td>2.380000e+02</td>
      <td>2.120000e+02</td>
      <td>2.140000e+02</td>
      <td>213.999998</td>
      <td>2.150000e+02</td>
      <td>214.999996</td>
      <td>2.130000e+02</td>
      <td>209.000005</td>
      <td>209.000001</td>
      <td>209.000004</td>
      <td>210.000003</td>
      <td>213.000005</td>
      <td>213.999999</td>
      <td>215.000003</td>
      <td>215.999997</td>
      <td>216.999998</td>
      <td>217.000003</td>
      <td>217.000002</td>
      <td>2.180000e+02</td>
      <td>217.999999</td>
      <td>217.999995</td>
      <td>218.999997</td>
      <td>218.000006</td>
      <td>2.180000e+02</td>
      <td>2.330000e+02</td>
      <td>1.480000e+02</td>
      <td>-2.152347e-08</td>
      <td>1.610000e+02</td>
      <td>2.370000e+02</td>
      <td>2.110000e+02</td>
      <td>2.180000e+02</td>
      <td>2.170000e+02</td>
      <td>2.190000e+02</td>
      <td>217.999998</td>
      <td>217.000003</td>
      <td>215.000001</td>
      <td>212.999998</td>
      <td>213.000001</td>
      <td>214.000000</td>
      <td>214.999997</td>
      <td>2.160000e+02</td>
      <td>215.999999</td>
      <td>217.999999</td>
      <td>217.999999</td>
      <td>220.000001</td>
      <td>219.999999</td>
      <td>221.000005</td>
      <td>221.999999</td>
      <td>220.999998</td>
      <td>222.000002</td>
      <td>2.250000e+02</td>
      <td>2.240000e+02</td>
      <td>2.350000e+02</td>
      <td>1.770000e+02</td>
      <td>3.391509e-08</td>
      <td>1.520000e+02</td>
      <td>2.400000e+02</td>
      <td>2.150000e+02</td>
      <td>2.190000e+02</td>
      <td>2.150000e+02</td>
      <td>217.999996</td>
      <td>217.999999</td>
      <td>217.000001</td>
      <td>214.000000</td>
      <td>211.999998</td>
      <td>210.999997</td>
      <td>210.999996</td>
      <td>213.000003</td>
      <td>216.000002</td>
      <td>216.000001</td>
      <td>218.000002</td>
      <td>217.999998</td>
      <td>219.000002</td>
      <td>219.000003</td>
      <td>2.200000e+02</td>
      <td>220.000005</td>
      <td>221.000002</td>
      <td>2.250000e+02</td>
      <td>226.000000</td>
      <td>221.000008</td>
      <td>2.320000e+02</td>
      <td>1.970000e+02</td>
      <td>-3.134657e-08</td>
      <td>1.000000e+02</td>
      <td>2.410000e+02</td>
      <td>2.210000e+02</td>
      <td>2.150000e+02</td>
      <td>2.190000e+02</td>
      <td>217.999998</td>
      <td>221.000004</td>
      <td>219.000001</td>
      <td>218.000002</td>
      <td>217.000002</td>
      <td>215.999997</td>
      <td>215.000002</td>
      <td>218.000000</td>
      <td>217.999998</td>
      <td>217.999998</td>
      <td>221.000000</td>
      <td>220.999999</td>
      <td>218.999999</td>
      <td>220.999997</td>
      <td>220.000001</td>
      <td>221.000002</td>
      <td>220.000004</td>
      <td>222.999999</td>
      <td>224.000002</td>
      <td>2.180000e+02</td>
      <td>255.000002</td>
      <td>1.500000e+02</td>
      <td>-1.048992e-07</td>
      <td>1.610000e+02</td>
      <td>2.290000e+02</td>
      <td>2.080000e+02</td>
      <td>231.999995</td>
      <td>212.000005</td>
      <td>219.000000</td>
      <td>219.000001</td>
      <td>217.999997</td>
      <td>218.999999</td>
      <td>217.999998</td>
      <td>218.000000</td>
      <td>217.000002</td>
      <td>219.000000</td>
      <td>2.190000e+02</td>
      <td>218.000000</td>
      <td>222.000001</td>
      <td>223.000000</td>
      <td>223.999999</td>
      <td>224.000001</td>
      <td>218.999998</td>
      <td>218.999997</td>
      <td>224.999997</td>
      <td>223.999995</td>
      <td>227.999995</td>
      <td>223.000005</td>
      <td>255.000000</td>
      <td>1.500000e+01</td>
      <td>-1.249524e-07</td>
      <td>2.550000e+02</td>
      <td>2.420000e+02</td>
      <td>2.040000e+02</td>
      <td>205.999994</td>
      <td>222.999998</td>
      <td>2.180000e+02</td>
      <td>222.000004</td>
      <td>220.999997</td>
      <td>217.999997</td>
      <td>218.000002</td>
      <td>218.000002</td>
      <td>216.000000</td>
      <td>217.000002</td>
      <td>218.000005</td>
      <td>218.000001</td>
      <td>218.999999</td>
      <td>222.000001</td>
      <td>220.000001</td>
      <td>216.000001</td>
      <td>2.170000e+02</td>
      <td>219.999997</td>
      <td>222.999995</td>
      <td>227.000001</td>
      <td>225.999996</td>
      <td>2.150000e+02</td>
      <td>254.999991</td>
      <td>3.500000e+01</td>
      <td>5.450883e-08</td>
      <td>2.250000e+02</td>
      <td>2.490000e+02</td>
      <td>2.380000e+02</td>
      <td>209.999997</td>
      <td>1.990000e+02</td>
      <td>220.000004</td>
      <td>225.999999</td>
      <td>222.999996</td>
      <td>222.000002</td>
      <td>221.000000</td>
      <td>220.000000</td>
      <td>219.999999</td>
      <td>2.180000e+02</td>
      <td>226.999999</td>
      <td>221.000001</td>
      <td>215.999999</td>
      <td>222.000002</td>
      <td>220.000000</td>
      <td>217.000000</td>
      <td>225.999998</td>
      <td>2.280000e+02</td>
      <td>2.230000e+02</td>
      <td>208.000003</td>
      <td>218.000004</td>
      <td>2.320000e+02</td>
      <td>255.000004</td>
      <td>6.200000e+01</td>
      <td>1.800000e+01</td>
      <td>2.370000e+02</td>
      <td>2.400000e+02</td>
      <td>2.430000e+02</td>
      <td>254.999998</td>
      <td>237.000000</td>
      <td>2.050000e+02</td>
      <td>206.999999</td>
      <td>219.000000</td>
      <td>2.250000e+02</td>
      <td>225.000000</td>
      <td>223.999998</td>
      <td>221.000000</td>
      <td>227.000003</td>
      <td>213.000002</td>
      <td>226.000002</td>
      <td>218.999998</td>
      <td>223.999999</td>
      <td>222.999998</td>
      <td>222.999998</td>
      <td>225.000005</td>
      <td>211.999997</td>
      <td>2.050000e+02</td>
      <td>230.999996</td>
      <td>251.999997</td>
      <td>2.400000e+02</td>
      <td>2.530000e+02</td>
      <td>7.000000e+01</td>
      <td>3.300000e+01</td>
      <td>2.410000e+02</td>
      <td>2.380000e+02</td>
      <td>237.999997</td>
      <td>2.400000e+02</td>
      <td>2.500000e+02</td>
      <td>252.999996</td>
      <td>234.000005</td>
      <td>214.999999</td>
      <td>213.000002</td>
      <td>216.000002</td>
      <td>219.999999</td>
      <td>221.999999</td>
      <td>235.999996</td>
      <td>99.000001</td>
      <td>210.999999</td>
      <td>226.000002</td>
      <td>223.999999</td>
      <td>224.000000</td>
      <td>2.150000e+02</td>
      <td>205.000002</td>
      <td>2.220000e+02</td>
      <td>246.999999</td>
      <td>254.999999</td>
      <td>244.000005</td>
      <td>236.000001</td>
      <td>2.550000e+02</td>
      <td>1.000000e+02</td>
      <td>2.900000e+01</td>
      <td>2.410000e+02</td>
      <td>2.340000e+02</td>
      <td>2.410000e+02</td>
      <td>239.000008</td>
      <td>238.999999</td>
      <td>237.999999</td>
      <td>243.999999</td>
      <td>250.000003</td>
      <td>236.000004</td>
      <td>221.000002</td>
      <td>210.999999</td>
      <td>213.000001</td>
      <td>225.999997</td>
      <td>223.999998</td>
      <td>2.240000e+02</td>
      <td>217.000000</td>
      <td>214.000001</td>
      <td>209.000001</td>
      <td>2.170000e+02</td>
      <td>2.430000e+02</td>
      <td>2.550000e+02</td>
      <td>245.000001</td>
      <td>237.999995</td>
      <td>2.430000e+02</td>
      <td>2.350000e+02</td>
      <td>252.999993</td>
      <td>1.120000e+02</td>
      <td>2.000000e+00</td>
      <td>2.300000e+02</td>
      <td>242.000000</td>
      <td>2.390000e+02</td>
      <td>235.999996</td>
      <td>242.000004</td>
      <td>238.000000</td>
      <td>237.000001</td>
      <td>238.000002</td>
      <td>245.000004</td>
      <td>242.000002</td>
      <td>236.000004</td>
      <td>227.000003</td>
      <td>215.000002</td>
      <td>215.999998</td>
      <td>208.000001</td>
      <td>216.999998</td>
      <td>222.999998</td>
      <td>242.000002</td>
      <td>249.000001</td>
      <td>248.000001</td>
      <td>243.999999</td>
      <td>240.999996</td>
      <td>2.350000e+02</td>
      <td>2.410000e+02</td>
      <td>233.999990</td>
      <td>2.530000e+02</td>
      <td>1.190000e+02</td>
      <td>-1.447285e-07</td>
      <td>1.890000e+02</td>
      <td>2.550000e+02</td>
      <td>2.320000e+02</td>
      <td>2.350000e+02</td>
      <td>2.340000e+02</td>
      <td>236.999997</td>
      <td>235.999997</td>
      <td>2.340000e+02</td>
      <td>234.999998</td>
      <td>238.999998</td>
      <td>241.999999</td>
      <td>242.000004</td>
      <td>241.000002</td>
      <td>236.999999</td>
      <td>2.410000e+02</td>
      <td>243.000004</td>
      <td>240.999998</td>
      <td>237.999996</td>
      <td>239.999997</td>
      <td>234.999998</td>
      <td>235.999999</td>
      <td>237.999996</td>
      <td>236.999996</td>
      <td>237.000008</td>
      <td>2.340000e+02</td>
      <td>252.999998</td>
      <td>5.000000e+01</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>2.060000e+02</td>
      <td>244.000003</td>
      <td>252.999992</td>
      <td>254.999998</td>
      <td>2.550000e+02</td>
      <td>254.999996</td>
      <td>2.550000e+02</td>
      <td>2.500000e+02</td>
      <td>243.999996</td>
      <td>238.000001</td>
      <td>235.000004</td>
      <td>232.000003</td>
      <td>230.999998</td>
      <td>230.000005</td>
      <td>235.000000</td>
      <td>238.999996</td>
      <td>248.999995</td>
      <td>254.000005</td>
      <td>254.999995</td>
      <td>2.550000e+02</td>
      <td>252.999996</td>
      <td>248.999999</td>
      <td>252.999990</td>
      <td>253.999994</td>
      <td>1.850000e+02</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>2.500000e+01</td>
      <td>7.700000e+01</td>
      <td>78.000000</td>
      <td>76.000000</td>
      <td>7.100000e+01</td>
      <td>7.000000e+01</td>
      <td>38.999999</td>
      <td>2.800000e+01</td>
      <td>1.500000e+01</td>
      <td>4.000001</td>
      <td>-0.000004</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>1.999997e+00</td>
      <td>14.000001</td>
      <td>51.000002</td>
      <td>55.000002</td>
      <td>7.700000e+01</td>
      <td>108.000000</td>
      <td>1.340000e+02</td>
      <td>1.530000e+02</td>
      <td>1.380000e+02</td>
      <td>76.999998</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>4.654391e-07</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>-4.551377e-07</td>
      <td>7.378563e-07</td>
      <td>-0.000002</td>
      <td>3.950150e-07</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>0.000004</td>
      <td>-4.699235e-07</td>
      <td>0.000003</td>
      <td>-0.000005</td>
      <td>9.842208e-07</td>
      <td>0.000005</td>
      <td>-0.000004</td>
      <td>-0.000004</td>
      <td>4.021377e-07</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>7.859119e-07</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>-6.257008e-07</td>
      <td>0.000001</td>
      <td>9.323434e-07</td>
      <td>0.000002</td>
      <td>8.744090e-07</td>
      <td>0.000002</td>
      <td>0.000005</td>
      <td>0.000003</td>
      <td>0.000005</td>
      <td>-0.000001</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>8.867231e-07</td>
      <td>0.000003</td>
      <td>-0.000003</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-0.000001</td>
      <td>3.297868e-07</td>
      <td>7.765040e-07</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>-8.448512e-07</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>8.809801e-07</td>
      <td>1.723858e-07</td>
      <td>0.000001</td>
      <td>-7.570235e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>-0.000001</td>
      <td>8.076585e-07</td>
      <td>-0.000005</td>
      <td>-0.000006</td>
      <td>0.000005</td>
      <td>-1.852046e-07</td>
      <td>6.976350e-08</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000003</td>
      <td>-3.723668e-07</td>
      <td>-0.000002</td>
      <td>7.009453e-07</td>
      <td>-0.000002</td>
      <td>3.933536e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>-0.000005</td>
      <td>-5.581073e-07</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>2.139088e-07</td>
      <td>0.000002</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>-8.592667e-07</td>
      <td>0.000002</td>
      <td>-7.577064e-07</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>3.743793e-07</td>
      <td>-6.565379e-07</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>-2.304249e-08</td>
      <td>9.522988e-07</td>
      <td>-3.426212e-07</td>
      <td>-6.002562e-07</td>
      <td>-3.102456e-07</td>
      <td>8.776177e-07</td>
      <td>-0.000002</td>
      <td>6.155732e-07</td>
      <td>-3.593068e-07</td>
      <td>0.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>-2.594887e-09</td>
      <td>-2.792910e-08</td>
      <td>-3.333367e-08</td>
      <td>8.642878e-08</td>
      <td>2.900000e+01</td>
      <td>1.570000e+02</td>
      <td>168.000001</td>
      <td>1.850000e+02</td>
      <td>1.820000e+02</td>
      <td>1.770000e+02</td>
      <td>1.740000e+02</td>
      <td>1.690000e+02</td>
      <td>181.000005</td>
      <td>1.640000e+02</td>
      <td>-2.068072e-07</td>
      <td>-9.467687e-08</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>4.300000e+01</td>
      <td>1.630000e+02</td>
      <td>1.880000e+02</td>
      <td>2.160000e+02</td>
      <td>255.000010</td>
      <td>2.550000e+02</td>
      <td>255.000006</td>
      <td>250.999996</td>
      <td>2.540000e+02</td>
      <td>255.000004</td>
      <td>255.000004</td>
      <td>255.000000</td>
      <td>2.500000e+02</td>
      <td>255.000007</td>
      <td>2.550000e+02</td>
      <td>1.750000e+02</td>
      <td>1.270000e+02</td>
      <td>1.600000e+01</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.690000e+02</td>
      <td>2.500000e+02</td>
      <td>2.550000e+02</td>
      <td>2.550000e+02</td>
      <td>252.999992</td>
      <td>240.000004</td>
      <td>234.999995</td>
      <td>230.999999</td>
      <td>231.999996</td>
      <td>228.999997</td>
      <td>2.300000e+02</td>
      <td>228.000002</td>
      <td>230.000004</td>
      <td>229.000001</td>
      <td>2.370000e+02</td>
      <td>245.999994</td>
      <td>255.000008</td>
      <td>2.550000e+02</td>
      <td>2.390000e+02</td>
      <td>2.110000e+02</td>
      <td>1.300000e+01</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>8.000000e+00</td>
      <td>2.160000e+02</td>
      <td>2.290000e+02</td>
      <td>2.420000e+02</td>
      <td>2.430000e+02</td>
      <td>2.410000e+02</td>
      <td>224.999998</td>
      <td>231.000003</td>
      <td>246.999995</td>
      <td>240.000002</td>
      <td>249.000000</td>
      <td>253.000005</td>
      <td>236.999995</td>
      <td>2.300000e+02</td>
      <td>2.370000e+02</td>
      <td>241.999999</td>
      <td>241.999997</td>
      <td>2.350000e+02</td>
      <td>2.330000e+02</td>
      <td>2.270000e+02</td>
      <td>2.500000e+02</td>
      <td>1.280000e+02</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>4.000000e+00</td>
      <td>2.520000e+02</td>
      <td>2.340000e+02</td>
      <td>2.300000e+02</td>
      <td>234.999991</td>
      <td>225.000000</td>
      <td>236.000002</td>
      <td>223.000004</td>
      <td>237.999998</td>
      <td>245.000000</td>
      <td>2.380000e+02</td>
      <td>2.330000e+02</td>
      <td>210.000000</td>
      <td>201.000002</td>
      <td>229.000004</td>
      <td>2.260000e+02</td>
      <td>232.999996</td>
      <td>2.450000e+02</td>
      <td>2.300000e+02</td>
      <td>2.260000e+02</td>
      <td>2.520000e+02</td>
      <td>1.080000e+02</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>3.100000e+01</td>
      <td>249.000004</td>
      <td>237.000009</td>
      <td>227.999999</td>
      <td>230.999999</td>
      <td>248.999997</td>
      <td>245.000005</td>
      <td>2.280000e+02</td>
      <td>241.000002</td>
      <td>246.000005</td>
      <td>230.999996</td>
      <td>194.999999</td>
      <td>222.999998</td>
      <td>254.999996</td>
      <td>227.000004</td>
      <td>225.000001</td>
      <td>2.230000e+02</td>
      <td>232.999995</td>
      <td>2.390000e+02</td>
      <td>1.890000e+02</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>-2.410009e-07</td>
      <td>181.000000</td>
      <td>238.999991</td>
      <td>204.999995</td>
      <td>2.160000e+02</td>
      <td>216.000002</td>
      <td>2.210000e+02</td>
      <td>225.000000</td>
      <td>214.000002</td>
      <td>225.000000</td>
      <td>229.999996</td>
      <td>242.999996</td>
      <td>238.999999</td>
      <td>223.999998</td>
      <td>208.000001</td>
      <td>206.000000</td>
      <td>204.000002</td>
      <td>2.160000e+02</td>
      <td>252.999997</td>
      <td>8.919840e-07</td>
      <td>-6.656718e-07</td>
      <td>-5.450623e-07</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.715096e-07</td>
      <td>254.999999</td>
      <td>219.000005</td>
      <td>207.000000</td>
      <td>199.999999</td>
      <td>205.000005</td>
      <td>2.040000e+02</td>
      <td>202.000003</td>
      <td>2.210000e+02</td>
      <td>161.000000</td>
      <td>157.000001</td>
      <td>220.000003</td>
      <td>197.000000</td>
      <td>207.999998</td>
      <td>207.000001</td>
      <td>205.999999</td>
      <td>209.999999</td>
      <td>203.000005</td>
      <td>254.999999</td>
      <td>55.000000</td>
      <td>-7.762443e-07</td>
      <td>-6.404821e-07</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>5.900000e+01</td>
      <td>233.000005</td>
      <td>2.050000e+02</td>
      <td>206.000005</td>
      <td>2.040000e+02</td>
      <td>204.000003</td>
      <td>206.999997</td>
      <td>200.000005</td>
      <td>219.999996</td>
      <td>173.000000</td>
      <td>156.000000</td>
      <td>222.999996</td>
      <td>204.000002</td>
      <td>205.999998</td>
      <td>207.000002</td>
      <td>209.000001</td>
      <td>2.060000e+02</td>
      <td>207.000001</td>
      <td>233.000003</td>
      <td>137.000002</td>
      <td>0.000002</td>
      <td>8.170825e-07</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>9.100000e+01</td>
      <td>2.340000e+02</td>
      <td>2.080000e+02</td>
      <td>202.000002</td>
      <td>207.999999</td>
      <td>207.999995</td>
      <td>209.000005</td>
      <td>206.000004</td>
      <td>218.999998</td>
      <td>196.000000</td>
      <td>1.910000e+02</td>
      <td>221.999999</td>
      <td>210.999998</td>
      <td>211.999999</td>
      <td>211.999998</td>
      <td>211.000003</td>
      <td>209.000002</td>
      <td>225.999999</td>
      <td>221.999998</td>
      <td>199.999996</td>
      <td>-1.907800e-08</td>
      <td>-6.678754e-07</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>1.210000e+02</td>
      <td>2.350000e+02</td>
      <td>226.999995</td>
      <td>224.999996</td>
      <td>205.999999</td>
      <td>211.000003</td>
      <td>210.000002</td>
      <td>209.999997</td>
      <td>220.999996</td>
      <td>197.000000</td>
      <td>188.999999</td>
      <td>224.000000</td>
      <td>214.000002</td>
      <td>213.000002</td>
      <td>214.000001</td>
      <td>214.000004</td>
      <td>2.180000e+02</td>
      <td>237.999999</td>
      <td>219.999995</td>
      <td>2.380000e+02</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>1.400000e+02</td>
      <td>2.390000e+02</td>
      <td>234.999997</td>
      <td>215.999995</td>
      <td>205.999996</td>
      <td>214.000001</td>
      <td>211.000005</td>
      <td>209.999997</td>
      <td>220.999996</td>
      <td>196.999998</td>
      <td>182.000002</td>
      <td>223.000000</td>
      <td>214.000001</td>
      <td>217.000001</td>
      <td>218.000000</td>
      <td>216.000005</td>
      <td>220.000001</td>
      <td>248.999995</td>
      <td>215.000003</td>
      <td>254.999999</td>
      <td>19.000002</td>
      <td>-2.314611e-07</td>
      <td>-0.000002</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>182.999996</td>
      <td>237.000004</td>
      <td>229.000002</td>
      <td>211.000002</td>
      <td>215.999995</td>
      <td>213.999998</td>
      <td>212.000000</td>
      <td>212.000000</td>
      <td>219.000001</td>
      <td>201.999999</td>
      <td>1.810000e+02</td>
      <td>226.999997</td>
      <td>213.000001</td>
      <td>218.000000</td>
      <td>219.999999</td>
      <td>216.000002</td>
      <td>218.999998</td>
      <td>255.000001</td>
      <td>207.000001</td>
      <td>254.999996</td>
      <td>77.000000</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>221.000003</td>
      <td>227.000005</td>
      <td>2.310000e+02</td>
      <td>212.000006</td>
      <td>215.000003</td>
      <td>213.000004</td>
      <td>212.999998</td>
      <td>213.000001</td>
      <td>218.000000</td>
      <td>209.999998</td>
      <td>192.999998</td>
      <td>225.999999</td>
      <td>212.000000</td>
      <td>216.000001</td>
      <td>220.999999</td>
      <td>221.999999</td>
      <td>2.120000e+02</td>
      <td>251.000001</td>
      <td>211.000003</td>
      <td>250.999999</td>
      <td>112.000001</td>
      <td>3.101954e-07</td>
      <td>-0.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>243.999994</td>
      <td>2.300000e+02</td>
      <td>232.000000</td>
      <td>212.999997</td>
      <td>219.999997</td>
      <td>211.999998</td>
      <td>214.000000</td>
      <td>214.000001</td>
      <td>217.000001</td>
      <td>2.170000e+02</td>
      <td>209.000000</td>
      <td>221.000001</td>
      <td>213.999999</td>
      <td>217.000000</td>
      <td>217.999999</td>
      <td>224.000000</td>
      <td>212.000002</td>
      <td>2.490000e+02</td>
      <td>2.170000e+02</td>
      <td>241.999995</td>
      <td>153.999998</td>
      <td>3.281200e-07</td>
      <td>-0.000002</td>
      <td>-1.307099e-07</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>7.308653e-07</td>
      <td>236.999994</td>
      <td>234.000003</td>
      <td>2.210000e+02</td>
      <td>214.999999</td>
      <td>225.999999</td>
      <td>2.090000e+02</td>
      <td>215.000002</td>
      <td>214.000002</td>
      <td>220.000002</td>
      <td>205.000002</td>
      <td>190.000000</td>
      <td>226.999997</td>
      <td>212.000001</td>
      <td>217.999999</td>
      <td>213.000000</td>
      <td>219.999997</td>
      <td>216.999996</td>
      <td>246.999999</td>
      <td>2.200000e+02</td>
      <td>230.999996</td>
      <td>182.000004</td>
      <td>3.892250e-07</td>
      <td>6.972123e-07</td>
      <td>-2.291508e-07</td>
      <td>-2.135472e-07</td>
      <td>-4.984975e-07</td>
      <td>-8.258425e-07</td>
      <td>0.000001</td>
      <td>2.490000e+02</td>
      <td>2.270000e+02</td>
      <td>235.999999</td>
      <td>217.999998</td>
      <td>230.999997</td>
      <td>210.999999</td>
      <td>216.000002</td>
      <td>215.000000</td>
      <td>220.000001</td>
      <td>209.999998</td>
      <td>184.000000</td>
      <td>227.000002</td>
      <td>214.999999</td>
      <td>216.999999</td>
      <td>216.000000</td>
      <td>2.230000e+02</td>
      <td>222.000000</td>
      <td>2.550000e+02</td>
      <td>225.000005</td>
      <td>217.000005</td>
      <td>200.000002</td>
      <td>0.000001</td>
      <td>-7.986391e-07</td>
      <td>-4.286650e-07</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>3.200000e+01</td>
      <td>255.000007</td>
      <td>223.000000</td>
      <td>241.999997</td>
      <td>220.999998</td>
      <td>229.000005</td>
      <td>215.000002</td>
      <td>214.999998</td>
      <td>215.999999</td>
      <td>218.999999</td>
      <td>217.000002</td>
      <td>199.999998</td>
      <td>2.250000e+02</td>
      <td>213.999999</td>
      <td>214.000001</td>
      <td>214.999999</td>
      <td>2.270000e+02</td>
      <td>2.220000e+02</td>
      <td>2.480000e+02</td>
      <td>234.000003</td>
      <td>213.999996</td>
      <td>2.090000e+02</td>
      <td>-5.163183e-07</td>
      <td>0.000002</td>
      <td>-1.174552e-07</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>-0.000001</td>
      <td>5.700000e+01</td>
      <td>254.999993</td>
      <td>221.999995</td>
      <td>240.000002</td>
      <td>221.999997</td>
      <td>227.000000</td>
      <td>215.999998</td>
      <td>216.000001</td>
      <td>217.000002</td>
      <td>220.000001</td>
      <td>215.000002</td>
      <td>201.000000</td>
      <td>225.999996</td>
      <td>213.999999</td>
      <td>216.999999</td>
      <td>213.000000</td>
      <td>222.000002</td>
      <td>223.999999</td>
      <td>240.999999</td>
      <td>233.999996</td>
      <td>2.190000e+02</td>
      <td>2.150000e+02</td>
      <td>-0.000002</td>
      <td>-5.629809e-07</td>
      <td>-3.012789e-07</td>
      <td>-1.447285e-07</td>
      <td>-6.046403e-07</td>
      <td>1.118729e-07</td>
      <td>8.300000e+01</td>
      <td>2.550000e+02</td>
      <td>2.170000e+02</td>
      <td>236.999997</td>
      <td>221.000004</td>
      <td>2.250000e+02</td>
      <td>213.999999</td>
      <td>216.999999</td>
      <td>217.999999</td>
      <td>221.000000</td>
      <td>217.000001</td>
      <td>202.999998</td>
      <td>2.290000e+02</td>
      <td>217.999999</td>
      <td>219.000002</td>
      <td>218.000001</td>
      <td>218.000003</td>
      <td>224.000005</td>
      <td>234.999994</td>
      <td>229.999998</td>
      <td>215.999999</td>
      <td>220.999998</td>
      <td>9.265024e-07</td>
      <td>0.000002</td>
      <td>4.011770e-07</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>-2.624320e-07</td>
      <td>97.999999</td>
      <td>255.000009</td>
      <td>217.000004</td>
      <td>2.370000e+02</td>
      <td>219.000001</td>
      <td>2.220000e+02</td>
      <td>2.140000e+02</td>
      <td>215.999999</td>
      <td>217.000002</td>
      <td>222.000001</td>
      <td>214.000002</td>
      <td>196.999999</td>
      <td>226.000005</td>
      <td>214.999998</td>
      <td>215.999999</td>
      <td>219.999995</td>
      <td>217.000005</td>
      <td>217.000004</td>
      <td>2.370000e+02</td>
      <td>231.000005</td>
      <td>216.000003</td>
      <td>224.000004</td>
      <td>-0.000002</td>
      <td>7.528400e-07</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>9.800000e+01</td>
      <td>2.550000e+02</td>
      <td>219.000003</td>
      <td>238.999997</td>
      <td>2.140000e+02</td>
      <td>2.190000e+02</td>
      <td>218.000000</td>
      <td>2.200000e+02</td>
      <td>2.190000e+02</td>
      <td>223.000001</td>
      <td>216.000002</td>
      <td>200.000002</td>
      <td>231.999996</td>
      <td>2.180000e+02</td>
      <td>221.000002</td>
      <td>219.000000</td>
      <td>219.000004</td>
      <td>2.080000e+02</td>
      <td>239.999997</td>
      <td>2.320000e+02</td>
      <td>2.160000e+02</td>
      <td>2.240000e+02</td>
      <td>0.000002</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>1.020000e+02</td>
      <td>255.000008</td>
      <td>220.000000</td>
      <td>2.380000e+02</td>
      <td>2.120000e+02</td>
      <td>220.999996</td>
      <td>2.140000e+02</td>
      <td>216.999998</td>
      <td>219.000002</td>
      <td>224.999995</td>
      <td>2.170000e+02</td>
      <td>202.000000</td>
      <td>228.999995</td>
      <td>2.170000e+02</td>
      <td>219.999997</td>
      <td>219.000000</td>
      <td>219.000002</td>
      <td>2.090000e+02</td>
      <td>233.999996</td>
      <td>234.000003</td>
      <td>216.999995</td>
      <td>2.240000e+02</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>1.090000e+02</td>
      <td>254.999991</td>
      <td>2.210000e+02</td>
      <td>243.999996</td>
      <td>2.220000e+02</td>
      <td>215.000000</td>
      <td>222.000005</td>
      <td>227.000004</td>
      <td>226.000004</td>
      <td>225.999999</td>
      <td>212.999999</td>
      <td>201.000003</td>
      <td>2.210000e+02</td>
      <td>221.999999</td>
      <td>224.000005</td>
      <td>224.000005</td>
      <td>215.999996</td>
      <td>215.999996</td>
      <td>241.999996</td>
      <td>224.000002</td>
      <td>2.170000e+02</td>
      <td>2.200000e+02</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>1.040000e+02</td>
      <td>2.550000e+02</td>
      <td>220.000005</td>
      <td>241.000000</td>
      <td>2.430000e+02</td>
      <td>2.550000e+02</td>
      <td>251.000000</td>
      <td>2.540000e+02</td>
      <td>248.999996</td>
      <td>245.999998</td>
      <td>227.000003</td>
      <td>215.999995</td>
      <td>2.450000e+02</td>
      <td>249.000004</td>
      <td>252.000004</td>
      <td>249.999997</td>
      <td>2.510000e+02</td>
      <td>2.440000e+02</td>
      <td>245.999993</td>
      <td>224.999996</td>
      <td>217.000005</td>
      <td>226.999995</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>9.800000e+01</td>
      <td>236.999994</td>
      <td>211.000003</td>
      <td>244.999999</td>
      <td>2.050000e+02</td>
      <td>104.000001</td>
      <td>9.900000e+01</td>
      <td>87.000000</td>
      <td>8.600000e+01</td>
      <td>82.000001</td>
      <td>89.000000</td>
      <td>95.000000</td>
      <td>71.000000</td>
      <td>86.000000</td>
      <td>99.000000</td>
      <td>1.090000e+02</td>
      <td>91.000000</td>
      <td>179.999999</td>
      <td>251.000002</td>
      <td>2.210000e+02</td>
      <td>226.000007</td>
      <td>2.140000e+02</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>2.350000e+02</td>
      <td>2.430000e+02</td>
      <td>246.999997</td>
      <td>-7.577064e-07</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>0.000003</td>
      <td>0.000002</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>28.000000</td>
      <td>0.000001</td>
      <td>-0.000003</td>
      <td>0.000001</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>-0.000001</td>
      <td>250.999997</td>
      <td>242.999990</td>
      <td>255.000007</td>
      <td>3.700000e+01</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>1.560000e+02</td>
      <td>2.370000e+02</td>
      <td>2.210000e+02</td>
      <td>6.560168e-08</td>
      <td>2.924545e-07</td>
      <td>3.999999e+00</td>
      <td>2.000000e+00</td>
      <td>1.000002e+00</td>
      <td>-6.002562e-07</td>
      <td>9.999983e-01</td>
      <td>3.000000e+00</td>
      <td>-0.000002</td>
      <td>1.999999e+00</td>
      <td>3.000001e+00</td>
      <td>0.999999</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>1.710000e+02</td>
      <td>2.410000e+02</td>
      <td>2.270000e+02</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.628456e-11</td>
      <td>2.116344e-10</td>
      <td>-1.459979e-10</td>
      <td>2.426657e-09</td>
      <td>-2.665179e-11</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>8.642878e-08</td>
      <td>-4.371375e-07</td>
      <td>3.400000e+01</td>
      <td>89.999999</td>
      <td>1.280000e+02</td>
      <td>1.460000e+02</td>
      <td>1.560000e+02</td>
      <td>1.600000e+02</td>
      <td>1.310000e+02</td>
      <td>54.000000</td>
      <td>5.516976e-08</td>
      <td>-2.068072e-07</td>
      <td>4.000000e+00</td>
      <td>9.006170e-08</td>
      <td>-2.751660e-08</td>
      <td>-2.676011e-08</td>
      <td>1.534132e-09</td>
      <td>3.965276e-09</td>
      <td>4.171343e-09</td>
      <td>5.597892e-10</td>
      <td>-2.459002e-10</td>
      <td>1.018412e-09</td>
      <td>4.727461e-09</td>
      <td>4.497033e-09</td>
      <td>-4.694904e-09</td>
      <td>8.493365e-08</td>
      <td>2.000000e+00</td>
      <td>1.985495e-07</td>
      <td>4.999999e+00</td>
      <td>140.999998</td>
      <td>1.740000e+02</td>
      <td>213.999996</td>
      <td>211.000002</td>
      <td>2.240000e+02</td>
      <td>236.999998</td>
      <td>240.000004</td>
      <td>225.999999</td>
      <td>2.270000e+02</td>
      <td>177.000001</td>
      <td>-9.767781e-07</td>
      <td>-6.219337e-07</td>
      <td>9.999995e-01</td>
      <td>9.999998e-01</td>
      <td>-8.071513e-09</td>
      <td>-1.345314e-08</td>
      <td>-3.650504e-08</td>
      <td>3.064484e-09</td>
      <td>1.395653e-09</td>
      <td>-5.678459e-10</td>
      <td>-8.652231e-10</td>
      <td>6.095701e-09</td>
      <td>-2.268324e-08</td>
      <td>1.082306e-07</td>
      <td>1.000000e+00</td>
      <td>7.000001e+00</td>
      <td>3.908779e-07</td>
      <td>112.000002</td>
      <td>237.000000</td>
      <td>219.999999</td>
      <td>197.999999</td>
      <td>199.000000</td>
      <td>191.000000</td>
      <td>1.910000e+02</td>
      <td>180.000001</td>
      <td>205.999997</td>
      <td>231.000004</td>
      <td>2.500000e+02</td>
      <td>124.000000</td>
      <td>-0.000002</td>
      <td>5.000000e+00</td>
      <td>-1.318873e-07</td>
      <td>-1.672773e-07</td>
      <td>2.125031e-09</td>
      <td>-8.742525e-08</td>
      <td>-1.006425e-08</td>
      <td>-5.139340e-09</td>
      <td>2.001768e-09</td>
      <td>-9.364624e-10</td>
      <td>-1.518954e-08</td>
      <td>-4.168295e-08</td>
      <td>1.493545e-07</td>
      <td>9.999997e-01</td>
      <td>5.000001e+00</td>
      <td>-9.688781e-07</td>
      <td>2.600000e+01</td>
      <td>214.000000</td>
      <td>197.000001</td>
      <td>198.999997</td>
      <td>247.000001</td>
      <td>246.999999</td>
      <td>244.999999</td>
      <td>239.000005</td>
      <td>2.090000e+02</td>
      <td>2.010000e+02</td>
      <td>241.999999</td>
      <td>111.999999</td>
      <td>5.257434e-07</td>
      <td>5.000000e+00</td>
      <td>-1.805364e-07</td>
      <td>4.216997e-07</td>
      <td>-1.558985e-07</td>
      <td>-1.001893e-07</td>
      <td>-4.949160e-09</td>
      <td>1.981593e-08</td>
      <td>2.329049e-09</td>
      <td>-5.676910e-09</td>
      <td>-1.800060e-08</td>
      <td>-3.427830e-08</td>
      <td>-5.326159e-07</td>
      <td>9.999991e-01</td>
      <td>6.555034e-07</td>
      <td>-0.000001</td>
      <td>-0.000002</td>
      <td>0.000002</td>
      <td>199.000006</td>
      <td>217.000005</td>
      <td>247.999998</td>
      <td>2.120000e+02</td>
      <td>2.080000e+02</td>
      <td>248.999998</td>
      <td>225.000000</td>
      <td>177.999999</td>
      <td>6.400000e+01</td>
      <td>-0.000002</td>
      <td>5.942894e-07</td>
      <td>9.570876e-07</td>
      <td>7.753735e-07</td>
      <td>-8.642634e-07</td>
      <td>-4.548926e-07</td>
      <td>1.847391e-08</td>
      <td>-1.021340e-08</td>
      <td>1.077761e-08</td>
      <td>-2.561862e-09</td>
      <td>-4.695927e-08</td>
      <td>4.582200e-08</td>
      <td>-3.727119e-08</td>
      <td>1.100016e-07</td>
      <td>0.000002</td>
      <td>1.000002</td>
      <td>0.000001</td>
      <td>4.999999</td>
      <td>125.000001</td>
      <td>215.000005</td>
      <td>1.550000e+02</td>
      <td>219.000005</td>
      <td>235.999999</td>
      <td>242.999996</td>
      <td>255.000001</td>
      <td>188.000001</td>
      <td>161.000001</td>
      <td>148.000001</td>
      <td>91.000000</td>
      <td>2.100000e+01</td>
      <td>-0.000002</td>
      <td>9.999988e-01</td>
      <td>5.144796e-07</td>
      <td>-1.955981e-07</td>
      <td>1.345243e-07</td>
      <td>1.621610e-07</td>
      <td>1.430707e-08</td>
      <td>-4.618929e-09</td>
      <td>4.971934e-09</td>
      <td>6.497929e-08</td>
      <td>2.643760e-07</td>
      <td>-2.410009e-07</td>
      <td>2.999998</td>
      <td>0.000002</td>
      <td>30.000000</td>
      <td>1.950000e+02</td>
      <td>175.999999</td>
      <td>1.800000e+02</td>
      <td>205.999999</td>
      <td>199.999998</td>
      <td>227.000004</td>
      <td>253.999999</td>
      <td>185.000002</td>
      <td>168.000000</td>
      <td>138.000000</td>
      <td>181.000001</td>
      <td>200.000000</td>
      <td>204.000002</td>
      <td>3.800000e+01</td>
      <td>-0.000002</td>
      <td>2.999998e+00</td>
      <td>-6.656718e-07</td>
      <td>-5.450623e-07</td>
      <td>-4.032686e-07</td>
      <td>-1.347659e-08</td>
      <td>-2.890140e-08</td>
      <td>1.131168e-07</td>
      <td>-4.175154e-08</td>
      <td>-5.971974e-07</td>
      <td>3.715096e-07</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>170.999996</td>
      <td>159.000002</td>
      <td>144.000002</td>
      <td>1.490000e+02</td>
      <td>155.000000</td>
      <td>1.530000e+02</td>
      <td>190.000003</td>
      <td>172.999999</td>
      <td>135.000001</td>
      <td>159.000000</td>
      <td>153.000000</td>
      <td>146.000000</td>
      <td>145.999999</td>
      <td>155.999999</td>
      <td>140.000002</td>
      <td>0.000002</td>
      <td>0.000001</td>
      <td>-7.762443e-07</td>
      <td>-6.404821e-07</td>
      <td>-6.205741e-07</td>
      <td>-9.954905e-08</td>
      <td>2.976238e-08</td>
      <td>-9.164984e-08</td>
      <td>-1.933697e-07</td>
      <td>-3.981918e-07</td>
      <td>4.334051e-07</td>
      <td>-0.000002</td>
      <td>1.100000e+01</td>
      <td>166.000000</td>
      <td>1.530000e+02</td>
      <td>156.999999</td>
      <td>159.000001</td>
      <td>152.000001</td>
      <td>137.000001</td>
      <td>178.999998</td>
      <td>178.000002</td>
      <td>143.000000</td>
      <td>162.000001</td>
      <td>152.000000</td>
      <td>159.000000</td>
      <td>158.000001</td>
      <td>1.520000e+02</td>
      <td>165.000002</td>
      <td>40.000002</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>8.170825e-07</td>
      <td>-8.944622e-08</td>
      <td>9.377016e-08</td>
      <td>-2.152347e-08</td>
      <td>-9.805354e-08</td>
      <td>1.100760e-07</td>
      <td>-1.284166e-07</td>
      <td>-8.785177e-07</td>
      <td>-5.893633e-07</td>
      <td>5.400000e+01</td>
      <td>180.000005</td>
      <td>161.999997</td>
      <td>153.000000</td>
      <td>155.000002</td>
      <td>157.999998</td>
      <td>145.000000</td>
      <td>174.999998</td>
      <td>1.750000e+02</td>
      <td>143.000000</td>
      <td>157.000000</td>
      <td>153.000000</td>
      <td>152.000000</td>
      <td>155.000000</td>
      <td>161.000001</td>
      <td>168.000001</td>
      <td>55.000000</td>
      <td>0.000002</td>
      <td>-1.907800e-08</td>
      <td>-6.678754e-07</td>
      <td>3.626715e-07</td>
      <td>7.653776e-08</td>
      <td>3.391509e-08</td>
      <td>-1.837763e-07</td>
      <td>-1.002178e-07</td>
      <td>9.562319e-08</td>
      <td>5.449854e-07</td>
      <td>6.690875e-07</td>
      <td>94.000000</td>
      <td>184.000000</td>
      <td>170.000002</td>
      <td>162.999998</td>
      <td>152.000001</td>
      <td>156.000001</td>
      <td>144.000001</td>
      <td>166.999999</td>
      <td>179.999998</td>
      <td>145.000000</td>
      <td>157.000000</td>
      <td>153.000000</td>
      <td>147.000000</td>
      <td>178.999998</td>
      <td>1.660000e+02</td>
      <td>171.999999</td>
      <td>116.000000</td>
      <td>1.135173e-07</td>
      <td>-0.000001</td>
      <td>0.000001</td>
      <td>-7.649599e-07</td>
      <td>1.421396e-07</td>
      <td>-3.134657e-08</td>
      <td>-1.488990e-07</td>
      <td>-2.217518e-07</td>
      <td>-7.524475e-07</td>
      <td>-8.247956e-07</td>
      <td>2.124748e-07</td>
      <td>131.999997</td>
      <td>183.999999</td>
      <td>181.999997</td>
      <td>177.000001</td>
      <td>164.999999</td>
      <td>155.000000</td>
      <td>150.000001</td>
      <td>164.000001</td>
      <td>186.000000</td>
      <td>146.000000</td>
      <td>158.000000</td>
      <td>157.000000</td>
      <td>145.000000</td>
      <td>172.999999</td>
      <td>169.000001</td>
      <td>173.999998</td>
      <td>180.000001</td>
      <td>0.000002</td>
      <td>-0.000001</td>
      <td>-2.314611e-07</td>
      <td>-0.000002</td>
      <td>3.276677e-07</td>
      <td>-1.048992e-07</td>
      <td>-9.582211e-08</td>
      <td>-1.728624e-07</td>
      <td>4.348013e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>158.000000</td>
      <td>179.999998</td>
      <td>196.999997</td>
      <td>198.999996</td>
      <td>178.000001</td>
      <td>158.999999</td>
      <td>151.000000</td>
      <td>161.000001</td>
      <td>1.880000e+02</td>
      <td>146.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>157.000000</td>
      <td>187.000000</td>
      <td>169.000000</td>
      <td>169.999997</td>
      <td>215.999999</td>
      <td>0.000003</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-1.099796e-08</td>
      <td>-1.249524e-07</td>
      <td>8.047429e-10</td>
      <td>1.957992e-07</td>
      <td>-5.361049e-07</td>
      <td>0.000001</td>
      <td>0.000002</td>
      <td>1.710000e+02</td>
      <td>172.999998</td>
      <td>196.000004</td>
      <td>216.999997</td>
      <td>183.000002</td>
      <td>165.000001</td>
      <td>152.000000</td>
      <td>160.999999</td>
      <td>188.999998</td>
      <td>148.000000</td>
      <td>159.000000</td>
      <td>158.000000</td>
      <td>162.000000</td>
      <td>194.999999</td>
      <td>1.680000e+02</td>
      <td>164.000000</td>
      <td>212.000003</td>
      <td>14.999999</td>
      <td>0.000002</td>
      <td>3.101954e-07</td>
      <td>-0.000002</td>
      <td>-2.671779e-07</td>
      <td>5.450883e-08</td>
      <td>-1.799488e-07</td>
      <td>-2.672337e-07</td>
      <td>3.815783e-07</td>
      <td>0.000002</td>
      <td>-8.765240e-07</td>
      <td>185.000004</td>
      <td>167.999998</td>
      <td>188.999997</td>
      <td>164.999999</td>
      <td>186.999999</td>
      <td>160.000000</td>
      <td>154.000000</td>
      <td>1.560000e+02</td>
      <td>192.999999</td>
      <td>148.000000</td>
      <td>159.000000</td>
      <td>156.000000</td>
      <td>162.000000</td>
      <td>176.999999</td>
      <td>173.000002</td>
      <td>1.640000e+02</td>
      <td>1.920000e+02</td>
      <td>46.000002</td>
      <td>0.000002</td>
      <td>3.281200e-07</td>
      <td>-0.000002</td>
      <td>-1.307099e-07</td>
      <td>3.196277e-07</td>
      <td>5.418402e-08</td>
      <td>-2.917852e-07</td>
      <td>7.308653e-07</td>
      <td>-0.000002</td>
      <td>13.000001</td>
      <td>1.930000e+02</td>
      <td>167.999998</td>
      <td>184.000000</td>
      <td>1.170000e+02</td>
      <td>191.000002</td>
      <td>157.000000</td>
      <td>158.000000</td>
      <td>156.000000</td>
      <td>194.999998</td>
      <td>149.000000</td>
      <td>163.000000</td>
      <td>150.000000</td>
      <td>160.000000</td>
      <td>124.000001</td>
      <td>169.000001</td>
      <td>165.999999</td>
      <td>1.870000e+02</td>
      <td>70.000001</td>
      <td>0.000002</td>
      <td>3.892250e-07</td>
      <td>6.972123e-07</td>
      <td>-2.291508e-07</td>
      <td>-2.135472e-07</td>
      <td>-4.984975e-07</td>
      <td>-8.258425e-07</td>
      <td>0.000001</td>
      <td>-5.332055e-07</td>
      <td>3.300000e+01</td>
      <td>191.999997</td>
      <td>166.999997</td>
      <td>195.000000</td>
      <td>90.000001</td>
      <td>185.999999</td>
      <td>156.000000</td>
      <td>158.000000</td>
      <td>157.000000</td>
      <td>193.999999</td>
      <td>151.000000</td>
      <td>166.000000</td>
      <td>153.000000</td>
      <td>174.000000</td>
      <td>9.300000e+01</td>
      <td>158.000000</td>
      <td>1.880000e+02</td>
      <td>184.000000</td>
      <td>96.000000</td>
      <td>-0.000002</td>
      <td>0.000001</td>
      <td>-7.986391e-07</td>
      <td>-4.286650e-07</td>
      <td>-1.681014e-07</td>
      <td>6.363970e-07</td>
      <td>-6.975870e-07</td>
      <td>-5.423990e-07</td>
      <td>0.000001</td>
      <td>56.000001</td>
      <td>190.999996</td>
      <td>168.000003</td>
      <td>203.999996</td>
      <td>126.000000</td>
      <td>172.999999</td>
      <td>161.000000</td>
      <td>161.000000</td>
      <td>155.000000</td>
      <td>193.000001</td>
      <td>1.550000e+02</td>
      <td>163.000000</td>
      <td>156.000000</td>
      <td>170.000000</td>
      <td>1.090000e+02</td>
      <td>1.520000e+02</td>
      <td>2.060000e+02</td>
      <td>180.000000</td>
      <td>114.999999</td>
      <td>5.651978e-07</td>
      <td>-5.163183e-07</td>
      <td>0.000002</td>
      <td>-1.174552e-07</td>
      <td>-5.947502e-08</td>
      <td>-2.880678e-07</td>
      <td>-0.000001</td>
      <td>5.352320e-07</td>
      <td>0.000002</td>
      <td>77.000000</td>
      <td>189.000001</td>
      <td>173.000001</td>
      <td>208.000002</td>
      <td>157.000000</td>
      <td>174.000001</td>
      <td>162.000000</td>
      <td>163.000000</td>
      <td>154.000000</td>
      <td>192.999999</td>
      <td>157.000000</td>
      <td>163.000000</td>
      <td>159.000000</td>
      <td>172.000000</td>
      <td>130.000000</td>
      <td>130.000000</td>
      <td>194.999999</td>
      <td>177.000003</td>
      <td>1.270000e+02</td>
      <td>-3.174695e-07</td>
      <td>-0.000002</td>
      <td>-5.629809e-07</td>
      <td>-3.012789e-07</td>
      <td>-1.447285e-07</td>
      <td>-6.046403e-07</td>
      <td>1.118729e-07</td>
      <td>-9.513843e-07</td>
      <td>-2.911069e-07</td>
      <td>8.100000e+01</td>
      <td>186.000001</td>
      <td>176.000003</td>
      <td>1.970000e+02</td>
      <td>148.000000</td>
      <td>174.000000</td>
      <td>162.999999</td>
      <td>164.000000</td>
      <td>156.000000</td>
      <td>194.999999</td>
      <td>1.530000e+02</td>
      <td>164.000000</td>
      <td>161.000000</td>
      <td>161.000000</td>
      <td>155.000001</td>
      <td>87.999999</td>
      <td>192.999999</td>
      <td>176.000001</td>
      <td>139.000002</td>
      <td>-0.000002</td>
      <td>9.265024e-07</td>
      <td>0.000002</td>
      <td>4.011770e-07</td>
      <td>1.363091e-07</td>
      <td>3.289930e-07</td>
      <td>-2.624320e-07</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>90.000000</td>
      <td>1.830000e+02</td>
      <td>198.999996</td>
      <td>1.550000e+02</td>
      <td>1.290000e+02</td>
      <td>183.000000</td>
      <td>161.000000</td>
      <td>165.000000</td>
      <td>156.000000</td>
      <td>195.000000</td>
      <td>155.000000</td>
      <td>159.000000</td>
      <td>164.000001</td>
      <td>158.999999</td>
      <td>166.999998</td>
      <td>100.000000</td>
      <td>1.940000e+02</td>
      <td>175.000002</td>
      <td>149.999999</td>
      <td>-0.000002</td>
      <td>-0.000002</td>
      <td>7.528400e-07</td>
      <td>-1.202886e-08</td>
      <td>-1.592361e-07</td>
      <td>5.095788e-07</td>
      <td>-4.052591e-08</td>
      <td>-2.873870e-07</td>
      <td>5.500636e-07</td>
      <td>103.000001</td>
      <td>181.000005</td>
      <td>2.090000e+02</td>
      <td>1.010000e+02</td>
      <td>147.000000</td>
      <td>1.780000e+02</td>
      <td>1.630000e+02</td>
      <td>165.000000</td>
      <td>155.000000</td>
      <td>197.000001</td>
      <td>162.000000</td>
      <td>1.570000e+02</td>
      <td>165.000001</td>
      <td>161.000001</td>
      <td>161.000002</td>
      <td>1.520000e+02</td>
      <td>191.000005</td>
      <td>1.700000e+02</td>
      <td>1.600000e+02</td>
      <td>-1.849482e-07</td>
      <td>0.000002</td>
      <td>2.427885e-08</td>
      <td>-2.188508e-07</td>
      <td>1.379810e-07</td>
      <td>-1.391677e-07</td>
      <td>1.906360e-07</td>
      <td>4.654391e-07</td>
      <td>0.000001</td>
      <td>113.000001</td>
      <td>1.810000e+02</td>
      <td>2.100000e+02</td>
      <td>79.000001</td>
      <td>1.720000e+02</td>
      <td>173.000000</td>
      <td>166.000000</td>
      <td>169.000001</td>
      <td>1.570000e+02</td>
      <td>197.999999</td>
      <td>167.000000</td>
      <td>1.570000e+02</td>
      <td>161.999999</td>
      <td>161.999998</td>
      <td>157.000002</td>
      <td>1.520000e+02</td>
      <td>184.000001</td>
      <td>173.000003</td>
      <td>156.000003</td>
      <td>7.859119e-07</td>
      <td>2.466921e-07</td>
      <td>1.444405e-07</td>
      <td>5.295760e-08</td>
      <td>-1.867194e-08</td>
      <td>-2.868018e-07</td>
      <td>-7.280360e-07</td>
      <td>-6.257008e-07</td>
      <td>0.000001</td>
      <td>1.320000e+02</td>
      <td>179.000002</td>
      <td>2.110000e+02</td>
      <td>101.000000</td>
      <td>180.000001</td>
      <td>170.000001</td>
      <td>167.000000</td>
      <td>171.000001</td>
      <td>158.000001</td>
      <td>193.000000</td>
      <td>1.720000e+02</td>
      <td>163.000001</td>
      <td>155.000000</td>
      <td>165.000000</td>
      <td>159.999998</td>
      <td>150.000003</td>
      <td>165.000004</td>
      <td>181.000002</td>
      <td>1.570000e+02</td>
      <td>7.765040e-07</td>
      <td>-1.484936e-08</td>
      <td>5.969346e-07</td>
      <td>4.756745e-08</td>
      <td>-9.525822e-08</td>
      <td>-1.043257e-07</td>
      <td>-3.057516e-08</td>
      <td>-7.851076e-07</td>
      <td>-8.448512e-07</td>
      <td>148.000005</td>
      <td>180.000002</td>
      <td>2.020000e+02</td>
      <td>1.130000e+02</td>
      <td>179.000001</td>
      <td>1.660000e+02</td>
      <td>164.999998</td>
      <td>166.000001</td>
      <td>161.999999</td>
      <td>180.999998</td>
      <td>1.640000e+02</td>
      <td>162.999997</td>
      <td>160.000002</td>
      <td>157.000003</td>
      <td>1.560000e+02</td>
      <td>1.540000e+02</td>
      <td>162.000005</td>
      <td>169.999998</td>
      <td>156.000000</td>
      <td>0.000002</td>
      <td>5.335659e-07</td>
      <td>-5.266864e-07</td>
      <td>8.280237e-08</td>
      <td>3.033599e-08</td>
      <td>5.283691e-08</td>
      <td>1.380712e-07</td>
      <td>1.159922e-07</td>
      <td>-0.000001</td>
      <td>150.999996</td>
      <td>182.000001</td>
      <td>1.810000e+02</td>
      <td>136.000000</td>
      <td>2.090000e+02</td>
      <td>177.999998</td>
      <td>1.770000e+02</td>
      <td>177.999999</td>
      <td>171.999998</td>
      <td>193.999998</td>
      <td>176.999999</td>
      <td>168.000000</td>
      <td>171.000002</td>
      <td>1.690000e+02</td>
      <td>174.000004</td>
      <td>167.000003</td>
      <td>160.000002</td>
      <td>1.870000e+02</td>
      <td>166.999997</td>
      <td>-5.421757e-07</td>
      <td>6.163737e-07</td>
      <td>-5.578727e-08</td>
      <td>4.793912e-09</td>
      <td>-4.510589e-09</td>
      <td>-9.451520e-09</td>
      <td>-1.198971e-07</td>
      <td>-2.364927e-08</td>
      <td>-9.384266e-07</td>
      <td>1.520000e+02</td>
      <td>186.999998</td>
      <td>1.790000e+02</td>
      <td>6.000001</td>
      <td>92.000001</td>
      <td>108.000001</td>
      <td>119.000000</td>
      <td>119.000001</td>
      <td>117.000001</td>
      <td>154.999999</td>
      <td>147.999997</td>
      <td>140.000001</td>
      <td>144.000000</td>
      <td>130.000002</td>
      <td>77.999999</td>
      <td>-0.000001</td>
      <td>38.000000</td>
      <td>209.999999</td>
      <td>146.000004</td>
      <td>4.965935e-07</td>
      <td>-2.344399e-07</td>
      <td>-1.488622e-08</td>
      <td>1.608706e-08</td>
      <td>-3.893886e-11</td>
      <td>-3.305654e-09</td>
      <td>-5.438611e-09</td>
      <td>-4.320514e-08</td>
      <td>-2.151095e-07</td>
      <td>1.660000e+02</td>
      <td>2.090000e+02</td>
      <td>1.280000e+02</td>
      <td>2.924545e-07</td>
      <td>-2.304249e-08</td>
      <td>9.522988e-07</td>
      <td>-3.426212e-07</td>
      <td>-6.002562e-07</td>
      <td>-3.102456e-07</td>
      <td>8.776177e-07</td>
      <td>-0.000002</td>
      <td>6.155732e-07</td>
      <td>-3.593068e-07</td>
      <td>0.000002</td>
      <td>3.771737e-07</td>
      <td>5.207580e-07</td>
      <td>-1.732680e-07</td>
      <td>-6.984834e-07</td>
      <td>-2.398092e-07</td>
      <td>2.374313e-07</td>
      <td>-4.985816e-08</td>
      <td>-2.682765e-10</td>
      <td>-2.551856e-09</td>
      <td>4</td>
    </tr>
  </tbody>
</table>


Behind the scenes this also splits our data into a *training* dataset that we use to update the model parameters, and a *validation* dataset that we use to evaluate the model.
This is really important because neural networks are incredibly flexible and can often memorise the training data; the validation dataset is an exam with questions the model has never seen before.

We can access the individual dataloaders with `.train` and `.valid` respectively.


```python
dls.train, dls.valid
```




    (<fastai.tabular.core.TabDataLoader at 0x7f4c0e502d10>,
     <fastai.tabular.core.TabDataLoader at 0x7f4c0e502e90>)



These can be iterated on to get batches of examples to train or evaluate the model on.

This particular dataloader returns a tuple containing 3 items


```python
batch = next(iter(dls.train))
type(batch), len(batch)
```




    (tuple, 3)



The first is an empty array.
This would contain any categorical variables in our model, but since we are only using the continuous pixel values it's empty.


```python
batch[0]
```




    tensor([], size=(4096, 0), dtype=torch.int64)



The second is a 4096x784 array of numbers.
These correspond to 4096 of the rows from the initial training data.


```python
print(batch[1].shape)
batch[1]
```

    torch.Size([4096, 784])





    tensor([[-0.0104, -0.0225, -0.0271,  ..., -0.1582, -0.0908, -0.0321],
            [-0.0104, -0.0225, -0.0271,  ..., -0.1582, -0.0908, -0.0321],
            [-0.0104, -0.0225, -0.0271,  ..., -0.1582, -0.0908, -0.0321],
            ...,
            [-0.0104, -0.0225, -0.0271,  ...,  3.5075,  8.2715, -0.0321],
            [-0.0104, -0.0225, -0.0271,  ..., -0.1582, -0.0908, -0.0321],
            [-0.0104, -0.0225, -0.0271,  ..., -0.1582, -0.0908, -0.0321]])



We can see the image has been slightly whitened by the normalization.
This is because we normalized each pixel column *independently*; we may get better results if the normalize them all together.
But you can still tell it's some kind of top.


```python
plt.imshow(batch[1][0].reshape(28, 28), cmap='Greys')
```




    <matplotlib.image.AxesImage at 0x7f4c0e45fbd0>





![png](/post/peeling-fastai-layered-api-with-fashion-mnist/output_48_1.png)



The final part of the batch is the *labels* from 0-9 corresponding to the each row; what we are trying to predict.


```python
print(batch[2].shape)
batch[2]
```

    torch.Size([4096, 1])





    tensor([[2],
            [9],
            [5],
            ...,
            [7],
            [1],
            [9]], dtype=torch.int8)



Apparently the image above is a shirt (and not a pullover or t-shirt/top).


```python
labels[str(batch[2][0][0].item())]
```




    'Pullover'



We can iterate through the batches to see we have about 4500 labels from each category in the training dataloader


```python
from collections import Counter

train_label_count = Counter()
for batch in dls.train:
    train_label_count.update(batch[2].squeeze().numpy())

train_label_count
```




    Counter({1: 4507,
             9: 4567,
             3: 4449,
             7: 4554,
             6: 4517,
             0: 4447,
             4: 4486,
             5: 4540,
             2: 4496,
             8: 4493})



Similarly the validation data contains around 1200 rows each.


```python
valid_label_count = Counter()
for batch in dls.valid:
    valid_label_count.update(batch[2].squeeze().numpy())

valid_label_count
```




    Counter({9: 1137,
             5: 1171,
             0: 1271,
             4: 1205,
             3: 1231,
             1: 1213,
             7: 1157,
             2: 1217,
             8: 1210,
             6: 1188})



20% of the data has gone into the validation set, but only a little over 75% is in the validation set, we've dropped around 5% of the data.


```python
n_valid = sum(valid_label_count.values())
n_train = sum(train_label_count.values())

{'n_train': n_train,
 'pct_train': '{:.2%}'.format(n_train / len(df)),
 'n_valid': n_valid,
 'pct_valid': '{:.2%}'.format(n_valid / len(df))}
```




    {'n_train': 45056,
     'pct_train': '75.09%',
     'n_valid': 12000,
     'pct_valid': '20.00%'}



The reason for this is fastai has made all the batches equal length by dropping the extra examples.


```python
n_train / 4096
```




    11.0



### 5. Learner

Now we have our dataloaders to load the data for training and validation we need a way to learn from the data.
The fastai learner contains all the things we need to do that:

1. the dataloaders
2. a *model* consisting of an *architecture* and *parameters*, which can make *output predictions* from *inputs*
3. any *metrics* for quantitatively evaluating the system
4. a *loss function* for automatically evaluating the quality of *output predictions* against *labels*
5. an *optimiser* for updating the parameters to minimise the loss function

We do all this with a [tabular_learner](https://docs.fast.ai/tabular.learner.html#tabular_learnerhttps://docs.fast.ai/tabular.learner.html#tabular_learner), we specify:

* dls: the dataloaders
* layers: The *hidden* layers that define the *architecutre* of the model, we use a single layer of dimension 100
* opt_fun: The optimiser to use for updating parameters, here Stochastic Gradient Descent
* metrics: human interpretable metrics; accuracy is the proportion of labels the model correctly guesses
* config: model configuration; here we are turning off BatchNorm which is a technique to help train Deep Neural Networks. As we're trying to keep the model simple we leave them off.


```python
learn = tabular_learner(dls, layers=[100], opt_func=SGD, metrics=accuracy, config=dict(use_bn=False, bn_cont=False))
```

Let's step through the parts of the learner

## 5.1 Dataloaders

We access the dataloaders using `.dls`, and can use them just as before


```python
learn.dls
```




    <fastai.tabular.data.TabularDataLoaders at 0x7f4c0e5766d0>




```python
batch = next(iter(learn.dls.valid))
tuple(x.shape for x in batch)
```




    (torch.Size([4096, 0]), torch.Size([4096, 784]), torch.Size([4096, 1]))



## 5.2 Model

The model can take our input data and make predictions.

The `get_preds` function returns the model predictions and input labels from a dataloader (the validation dataloader by default).


```python
probs, actual = learn.get_preds()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







The probs is a bunch of numbers corresponding to the probability the image of the corresponding class


```python
print(probs.shape)
probs
```

    torch.Size([12000, 10])





    tensor([[0.1305, 0.0419, 0.0984,  ..., 0.1273, 0.0680, 0.0738],
            [0.1179, 0.0815, 0.1189,  ..., 0.0941, 0.0969, 0.1004],
            [0.1168, 0.0627, 0.1048,  ..., 0.0942, 0.1132, 0.0914],
            ...,
            [0.1110, 0.0944, 0.0760,  ..., 0.1031, 0.1014, 0.0918],
            [0.1116, 0.0645, 0.0845,  ..., 0.0651, 0.1352, 0.0923],
            [0.0975, 0.0764, 0.1297,  ..., 0.0754, 0.1255, 0.0889]])



The probabilities sum to 1


```python
probs.sum(axis=1)
```




    tensor([1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000])



The actual categories from the validation data is the second argument.


```python
print(actual.shape)
actual
```

    torch.Size([12000, 1])





    tensor([[9],
            [5],
            [0],
            ...,
            [8],
            [0],
            [2]], dtype=torch.int8)



We can check that the actuals match the labels from the first validation batch


```python
assert (actual[:len(batch[2])] == batch[2]).all().item()
```

The predictions come from the underlying model running a batch at a time


```python
batch_pred = learn.model(batch[0], batch[1])
batch_pred
```




    tensor([[ 0.3774, -0.7596,  0.0953,  ...,  0.3521, -0.2752, -0.1932],
            [ 0.1231, -0.2459,  0.1313,  ..., -0.1020, -0.0729, -0.0379],
            [ 0.1825, -0.4396,  0.0746,  ..., -0.0320,  0.1516, -0.0628],
            ...,
            [ 0.1052, -0.6275,  0.0955,  ..., -0.0851,  0.1674, -0.1369],
            [ 0.1469, -0.3495, -0.2195,  ..., -0.2075,  0.0509, -0.0478],
            [-0.2378, -0.6591, -0.1431,  ..., -0.0598,  0.1025,  0.2491]],
           grad_fn=<AddmmBackward0>)



You might notice these *aren't* probabilities; some of them are negative.

There's a trick to make numbers into probabilities, called the softmax function.


```python
batch_probs = F.softmax(batch_pred, dim=1)
batch_probs
```




    tensor([[0.1305, 0.0419, 0.0984,  ..., 0.1273, 0.0680, 0.0738],
            [0.1179, 0.0815, 0.1189,  ..., 0.0941, 0.0969, 0.1004],
            [0.1168, 0.0627, 0.1048,  ..., 0.0942, 0.1132, 0.0914],
            ...,
            [0.1077, 0.0518, 0.1067,  ..., 0.0891, 0.1146, 0.0846],
            [0.1224, 0.0745, 0.0849,  ..., 0.0859, 0.1112, 0.1007],
            [0.0739, 0.0485, 0.0813,  ..., 0.0883, 0.1039, 0.1203]],
           grad_fn=<SoftmaxBackward0>)



These give exactly the same predictions for the batch as before


```python
assert (probs[:len(batch_pred)] == batch_probs).all().item()
```

We can look at the underlying *model architecture*.

Ignore (embeds) and (emb_drop); the main part of the model is the (layers) consisting of a Sequential containing:

* Linear model that takes 28x28=784 features in, and output 100 features
* ReLU (which just means "set all negative values to 0")
* Linear model that takes in 100 features and outputs 10 features

That is it's just two linear functions with a "set negative values to 0" in between!


```python
learn.model
```




    TabularModel(
      (embeds): ModuleList()
      (emb_drop): Dropout(p=0.0, inplace=False)
      (layers): Sequential(
        (0): LinBnDrop(
          (0): Linear(in_features=784, out_features=100, bias=True)
          (1): ReLU(inplace=True)
        )
        (1): LinBnDrop(
          (0): Linear(in_features=100, out_features=10, bias=True)
        )
      )
    )



We can also look at the underlying parameters from the model:

* 100 x 784 parameters for the first linear function
* 100 parameters for the first bias
* 10 x 100 parameters for the second linear function
* 10 parameters for the second bias


```python
[x.shape for x in learn.parameters()]
```




    [torch.Size([100, 784]),
     torch.Size([100]),
     torch.Size([10, 100]),
     torch.Size([10])]



### 5.3 Metrics

The metrics are the human interpretable quantitative measures of the model; in this case we just used the accuracy.

We can get the loss and any metrics we passed in by calling `learn.validate()`.

The accuracy should be close to 10% because we have a randomly initialised model with 10 equally likely categories.


```python
learn.validate()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    (#2) [2.344938278198242,0.08091666549444199]



We can list out all the metrics


```python
learn.metrics
```




    (#1) [<fastai.learner.AvgMetric object at 0x7f4c0e492490>]



We can get the name of each metric


```python
learn.metrics[0].name
```




    'accuracy'



and call it on our predictions to get the accuracy


```python
learn.metrics[0].func(probs, actual)
```




    TensorBase(0.0809)



Our actual predictions are the categories with the highest probability


```python
preds = probs.argmax(axis=1)
preds
```




    tensor([4, 2, 0,  ..., 5, 8, 2])



Then the accuracy is just the proportion of predictions that are the same as the actuals


```python
sum(preds == actual.flatten()) / len(preds)
```




    tensor(0.0809)



### 5.4 Loss

Accuracy is a good easy to understand metric, but it's hard to optimise.
The accuracy only changes when the order of the probabilities change.
A small change in probabilities won't change accuracy most of the time so it's hard to tell which direction to move the parameters to make it better.

Instead for multicategory classification we use something called CrossEntropyLoss


```python
learn.loss_func
```




    FlattenedLoss of CrossEntropyLoss()



We can evaluate it on a single batch by passing the model predictions (*not* the probabilities) and the labels


```python
learn.loss_func(batch_pred, batch[2])
```




    TensorBase(2.3452, grad_fn=<AliasBackward0>)



What is CrossEntropyLoss?

* Find the probability of each *actual* category
* Take the negative logarithm of each
* Average over all entries

Since the logarithm is bigger the bigger the input (in mathematical jargon it's *strictly monotonic*) the higher the probability for the correct class the lower the CrossEntropyLoss.
If we bump up the probability for the correct class by x for all predictions, then the loss decreases by -log(x).


```python
actual_probs = torch.tensor([prob[idx] for prob, idx in zip(batch_probs, batch[2].flatten())])
-actual_probs.log().mean()
```




    tensor(2.3452)



Here's a way to do this with just indexing:

* pass `torch.arange(len(batch_probs))`, this generates the list `[0, 1, 2, ..., N]`
* pass the label index as a long `[0, 0, ... 9]`

This will extract the pairs of row 0 to N, and the corresponding label column.

This is faster and PyTorch knows how to differentiate it.


```python
actual_probs = batch_probs[torch.arange(len(batch_probs)), batch[2].flatten().long()]
loss = -actual_probs.log().mean()
loss
```




    tensor(2.3452, grad_fn=<NegBackward0>)



### 5.5 Optimizer

Once we have a loss we need a way to update the model parameters in a way that decreases the loss; we call this component an optimizer.

This isn't automatically created so we create it using `create_opt`


```python
learn.opt_func
```




    <function fastai.optimizer.SGD(params, lr, mom=0.0, wd=0.0, decouple_wd=True)>




```python
learn.opt = learn.opt_func(learn.parameters(), lr=0.1)
```

Let's create a copy of the old parameters for reference


```python
old_params = [p.detach().numpy().copy() for p in learn.parameters()]
```

We want to move the parameters in the direction that decreases the loss.
To do this we call `backward` to fill in all the derivatives with respect to the parameters


```python
*x, y = next(iter(dls.valid))
preds = learn.model(*x)
loss = learn.loss_func(preds, y)
```


```python
loss
```




    TensorBase(2.3452, grad_fn=<AliasBackward0>)




```python
with torch.no_grad():
    loss.backward()
    learn.opt.step()
    learn.zero_grad()
```


```python
new_params = [p.detach().numpy() for p in learn.parameters()]
```

And the weights have moved!


```python
old_params[-1] - new_params[-1]
```




    array([-0.00016104, -0.00301814,  0.00073723,  0.0009996 ,  0.00075129,
            0.00130542, -0.00203079, -0.00050665,  0.00146017,  0.00046292],
          dtype=float32)



And the loss on the batch has decreased


```python
preds = learn.model(*x)
loss = learn.loss_func(preds, y)
loss
```




    TensorBase(2.0883, grad_fn=<AliasBackward0>)



### 6. Fit

The fit function just runs the training loop above.
In each *epoch* for each batch in the training dataloader it:

* evaluates the model on the inputs
* calculates the loss against the outputs
* updates the parameters with the optimizer to reduce the loss

Then at the end of each epoch it reports the metrics on the validation set (as well as the losses).


The fit argument takes two parameters:

* n_epoch: Number of times to run throgh the training data
* lr: The learning rate to use in the optimizer


```python
learn.fit(n_epoch=4, lr=0.1)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.391490</td>
      <td>0.972408</td>
      <td>0.700417</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.085862</td>
      <td>0.760498</td>
      <td>0.741750</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.928742</td>
      <td>0.673375</td>
      <td>0.765333</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.828868</td>
      <td>0.620714</td>
      <td>0.781417</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


# Training a model with fastai's lower level APIs

Now that we know all the components of the high level API let's rewrite it using the lower level APIs

Lets start fresh by clearing all our previous imports from the python namespace (although I think they're still loaded into memory).


```python
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
```

## 1. Imports

This time we'll only use three fundamental things from fastai: the Learner, the SGD optimizer and the DataLoaders object


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import tensor
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.optimizer import SGD
```

## 2. Load Data

We'll do this with Pandas as before, but this time we won't worry about converting the label into a categorical datatype.


```python
df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
df_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59995</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>73</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59997</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>160</td>
      <td>162</td>
      <td>163</td>
      <td>135</td>
      <td>94</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59998</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59999</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>60000 rows × 785 columns</p>
</div>



### 3. Data Loaders

Previously we ran

```
dls = TabularDataLoaders.from_df(df, y_names='label', bs=4096, procs=[Normalize])
````

We will do the steps manually:

* create a random validation split
* create training and validation datasets
* wrap these datasets in dataloaders with batchsize of 4096
* normalize the data


```python
valid_pct = 0.2

valid_mask = np.random.choice([True, False], len(df), p=(valid_pct, 1-valid_pct))
valid_mask
```




    array([False, False, False, ..., False, False, False])




```python
np.mean(valid_mask)
```




    0.20161666666666667



We can create Datasets containing the pairs of (image, label) for each of the train, validation and test splits.

We normalize the pixels to be between 0 and 1. (This is slightly different to `Normalize` which performs a linear transform on each column so that it has mean 0 and standard deviation 1).


```python
ds_train = [(np.array(img, dtype=np.float32) / 255., label) for _idx, (label, *img) in df[~valid_mask].iterrows()]
ds_valid = [(np.array(img, dtype=np.float32) / 255., label) for _idx, (label, *img) in df[ valid_mask].iterrows()]
ds_test  = [(np.array(img, dtype=np.float32) / 255., label) for _idx, (label, *img) in df_test.iterrows()]
```

We can pick out an example


```python
x, y = ds_train[0]


plt.imshow(x.reshape(28,28), cmap='Greys')
y
```




    2





![png](/post/peeling-fastai-layered-api-with-fashion-mnist/output_139_1.png)



We then put these into a PyTorch DataLoaders to shuffle them and collate them into batches


```python
batch_size = 4096

dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_valid = DataLoader(ds_valid, batch_size)
dl_test = DataLoader(ds_test, batch_size)
```


```python
x, y = next(iter(dl_train))
x, y
```




    (tensor([[0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             ...,
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.]]),
     tensor([5, 0, 8,  ..., 8, 3, 9]))



We can then wrap these in a DataLoaders object


```python
dls = DataLoaders(dl_train, dl_valid, dl_test)
```


```python
dls.train, dls[0]
```




    (<torch.utils.data.dataloader.DataLoader at 0x7f4c0e6e0450>,
     <torch.utils.data.dataloader.DataLoader at 0x7f4c0e6e0450>)



### 4. Learner


Using the high level API did a lot of things:

```
learn = tabular_learner(dls, layers=[100], opt_func=SGD, metrics=accuracy, config=dict(use_bn=False, bn_cont=False))
```

1. build and initialise the model
2. set the metrics
3. set an appropriate loss function
4. register the optimizer

We'll do these parts manually and put them into a Learner.

#### 4.1 Model

Using PyTorch's Sequential we can easily rewrite the model manually


```python
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)
```

And run it over an example batch of data


```python
x.shape
```




    torch.Size([4096, 784])



We get 10 outputs for each item in the batch, as expected.


```python
pred = model(x)
print(pred.shape)
pred
```

    torch.Size([4096, 10])





    tensor([[ 0.0838,  0.1000,  0.0104,  ..., -0.0319, -0.1107,  0.0448],
            [ 0.0488,  0.0577,  0.0980,  ...,  0.0814, -0.1584,  0.1305],
            [ 0.1289,  0.0631,  0.0885,  ...,  0.0194, -0.0895,  0.0239],
            ...,
            [ 0.1703,  0.0883,  0.0789,  ..., -0.0812, -0.1138,  0.1045],
            [ 0.0145,  0.0626,  0.1440,  ..., -0.0434, -0.1266,  0.2100],
            [ 0.0092,  0.2286,  0.2602,  ...,  0.0141, -0.0817,  0.0991]],
           grad_fn=<AddmmBackward0>)



#### 4.2 Metrics

We can calculate accuracy as the number of predictions that are the same as the labels.
Since we have 10 equally likely classes for our randomly initialised model it should be about 10%.


```python
def accuracy(prob, actual):
    preds = prob.argmax(axis=-1)
    return sum(preds == actual.flatten()) / len(actual)
```


```python
accuracy(pred, y)
```




    tensor(0.0835)



### 4.3 Loss

The appropriate loss function for multiclass classification is CrossEntropy loss.


```python
loss_function = nn.CrossEntropyLoss()
```


```python
loss_function(pred, y)
```




    tensor(2.3009, grad_fn=<NllLossBackward0>)



### Optimizer

PyTorch provides `torch.optim.SGD` optimizer but we can't use it directly with a Learner; from the [docs](https://docs.fast.ai/learner.html#Learner)

> The most important is opt_func. If you are not using a fastai optimizer, you will need to write a function that wraps your PyTorch optimizer in an OptimWrapper. See the optimizer module for more details. This is to ensure the library's schedulers/freeze API work with your code.

We'll use fastai's SGD instead for now.

### Putting it into Learner



```python
learn = Learner(dls=dls, model=model, loss_func=nn.CrossEntropyLoss(), opt_func=SGD, metrics=[accuracy])
```

Note that this performs slightly worse than our original model which got to 82% accuracy and 0.5 validation loss in 5 epochs. It would be interesting to know what's changed!

With this kind of machine learning code a small change can make a big difference in how fast a model trains and how accurate it gets; this is why it's good to be able to dig into the detail!


```python
learn.fit(5, lr=0.2)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.773703</td>
      <td>1.286911</td>
      <td>0.647599</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.426255</td>
      <td>1.041697</td>
      <td>0.638919</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.228973</td>
      <td>0.891194</td>
      <td>0.687113</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.102247</td>
      <td>0.816285</td>
      <td>0.696784</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.006129</td>
      <td>0.768820</td>
      <td>0.736877</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


# Training with base PyTorch

Now we're going to train the model again, but this time just using the basic utilities from Pytorch.
We'll also use no classes and keep everything as simple low level functions so we can see the underlying mechanics.

First we clear our namespace to prevent cheating.


```python
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
```

## 1. Import

We'll import as few utilities as we can


```python
import numpy as np
from torch import tensor, randn, arange, no_grad, stack
```

## 2. Load data

This time we'll load the data in using pure Numpy; because the data is just numbers it's easy to do this.


```python
data = np.loadtxt('../input/fashionmnist/fashion-mnist_train.csv', skiprows=1, delimiter=',')
```

## 3. Dataloaders


```python
valid_mask = np.random.choice([True, False], len(data), p=(0.2, 0.8))
```


```python
X_train, y_train = tensor(data[~valid_mask, 1:].astype(np.float32) / 255.), tensor(data[~valid_mask,0].astype(np.int64))
X_valid, y_valid = tensor(data[ valid_mask, 1:].astype(np.float32) / 255.), tensor(data[ valid_mask,0].astype(np.int64))
```

### Learner

### 4.1 Model

Our model consists an *architecture* and *parameters*; we'll need a way to initialise those parameters.


```python
def init_params(size, std=1.0): return (randn(size)*std).requires_grad_()
```

Now, as before, we can set up 2 linear models with a ReLU between them.
In torch nn code this looks like:

```
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)
```

The first linear layer consists of 784 * 100 weights plus 100 biases.
The ReLU layer has no parameters; it's just a nonlinear activation.
The final layer consists of 100 * 10 weights to 10 biases.


```python
w1, b1 = init_params((784, 100)), init_params((100,))
w2, b2 = init_params((100, 10)), init_params((10,))

params = [w1, w2, b1, b2]
```

Our model then takes in the 784 pixels and performs:

* affine projection onto 100 dimensional space
* ReLU: Replace all the negative values by zero
* affine transformation onto 10 dimensional space

That looks like this:


```python
def model(x):
    act1 = x@w1 + b1
    act2 = act1 * (act1 > 0)
    act3 = act2@w2 + b2
    return act3
```

This can take the predictor from our dataloader


```python
x, y = X_train[:1024], y_train[:1024]

pred = model(x)

pred.shape
```




    torch.Size([1024, 10])



### Metrics




```python
def accuracy(pred, y): return sum(y.flatten() == pred.argmax(axis=1)) / len(y)

accuracy(pred, y)
```




    tensor(0.0869)




```python
accuracy(model(X_valid), y_valid)
```




    tensor(0.0910)



### 3. Loss function

Our loss function is the *negative log likelihood*; the likelihood is how probable the data is given the model, that is we average the probabilities for the correct label, and then take the negative log.

The first step in calculating this is getting the model probabilities.
We normalise th predictions with a softmax; expoentiate to make positive, and then divide by the sum to normalise to 1.

Unfortunately if we do this naively we end up getting infinity because of the limits of floating point arithmetic.


```python
pred.exp().sum(axis=1)
```




    tensor([inf, inf, inf,  ..., inf, inf, inf], grad_fn=<SumBackward1>)



Instead we use the log probabilities, which have a better range in floating point space, and use the [log-sum-exp trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) to make it stable (PyTorch has a [logsumexp function](https://pytorch.org/docs/stable/generated/torch.logsumexp.html), but it's easy to write.


```python
def logsumexp(x):
    c = x.max(axis=1).values
    x_shift = x - c[:, None]
    return c + x_shift.exp().sum(axis=1).log()
```

Check they are the same


```python
a = tensor([[1,2,3], [4,5,7]])
```


```python
a.exp().sum(axis=1).log(), logsumexp(a)
```




    (tensor([3.4076, 7.1698]), tensor([3.4076, 7.1698]))



We can then calculate the log probabilities using the softmax


```python
logprob = a - logsumexp(a)[:, None]
logprob
```




    tensor([[-2.4076, -1.4076, -0.4076],
            [-3.1698, -2.1698, -0.1698]])



And if we exponentiate them they sum to 1.


```python
logprob.exp().sum(axis=1)
```




    tensor([1., 1.])




```python
def pred_to_logprob(pred):
    return pred - logsumexp(pred)[:, None]
```


```python
pred_to_logprob(pred)[range(len(y)), y.long()]
```




    tensor([-2.9484e+02, -4.0163e+02, -8.3874e+01,  ..., -1.9576e+02,
            -3.4655e+02, -4.0436e-03], grad_fn=<IndexBackward0>)




```python
def loss_func(pred, y):
    logprob = pred_to_logprob(pred)

    true_prob = logprob[range(len(y)), y]

    return -true_prob.mean()
```

Our randomly initialised weights should on average give a ~1/10 probability to each class, and so the loss should be around -log(1/10) = 2.3.


```python
loss = loss_func(pred, y)
loss
```




    tensor(126.1162, grad_fn=<NegBackward0>)



### Optimizer

The SGD optimizer just moves each paramater a small step down the gradient to reduce the overall loss (and then we need to reset the gradients to zero).

We can easily run the whole training loop as follows (though note we get slightly worse accuracy than last time).


```python
batch_size = 2048
lr = 0.2

for epoch in range(5):
    for _batch in range(len(X_train) // batch_size):
        # Data loader
        idx = np.random.choice(len(X_train), batch_size, replace=False)
        X, y = X_train[idx], y_train[idx]

        pred = model(X)
        loss = loss_func(pred, y)
        loss.backward()
        with no_grad():
            for p in params:
                p -= lr * p.grad
                p.grad.zero_()


    print(epoch, accuracy(model(X_valid), y_valid))
```

    0 tensor(0.5943)
    1 tensor(0.6033)
    2 tensor(0.6119)
    3 tensor(0.6306)
    4 tensor(0.6387)


# What's the purpose of abstraction?

The high level tabular API is convenient for very quickly training a good model.
It has good defaults for things like the loss, model architecture and the optimiser, but it highly configurable if we want to change the defaults.
However this API is limited to the kinds of applications it's been built for; if we wanted a very different architecture or to work on a novel kind of input or output we can't use it.

The midlevel API exposes the Learner which is a *very* flexible training loop, and lets us use whatever kind of model, data, and optimiser we want.
This can be used for any number of tasks, but requires more work to implement the model, and choose the right hyperparameters.

Using the low level PyTorch API with minimal abstraction we can see how everything fits together.
There's no magic, and the training loop itself is rather simple.
However if we want to for example switch the optimiser, or the model, or the data type, it involves rewriting the training loop and it's a lot of work to maintain and debug.
The abstractions in the midlevel API often allow switching models, optimisers, data types, or metrics without changing anything else.

The next step would be to rebuild the abstractions to understand what they're actually doing.
Each time we went down a layer our model trained slower; with the right abstractions it's easy to focus on one piece at a time and work out *why* this happened (and maybe even discover a better way of training!)