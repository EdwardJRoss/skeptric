---
date: 2019-10-11 15:14:09+11:00
image: /images/lego.png
title: 'Data Blockless: A better way to create data'
---

Before you can do any machine learning you need to be able to read the data, create test and training splits and convert it into the right format.
Fastai has a generic [data block API](https://docs.fast.ai/data_block.html) for doing these tasks.
However it's quite hard to extend to new data types.
There's a few classes to implement; Items, ItemLists, LabelLists and the Preprocessors which are obfuscated through a complex inheritance and dispatch hierarchy.
I spent two weeks of evenings trying to implement a combined image and tabular data for the [Petfinder.my Adoption Prediction Kernel](https://www.kaggle.com/c/petfinder-adoption-prediction), and only [managed to combine the data sources](https://www.kaggle.com/edwardjross/neural-network-tabuler-vision-fastai-poc) because fastai's [Sylvain Guggers released a MixedItemList](https://forums.fast.ai/t/custom-itemlist-getting-forkingpickler-broken-pipe/39086/10).

However it's possible to make data preparation straightforward and easy to extend, using a generalisation of the `ListContainer` from [fastai part2 v3](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb#A-Hooks-class).

# Worked example

Creating a Pytorch DataLoader for Deep Learning requires:

1. Reading in the data
2. Apply transforms
3. Label the inputs
4. Split into training, validation and test sets
5. Wrap it in a DataLoader

Here's an extract from a [worked example](https://nbviewer.jupyter.org/github/EdwardJRoss/agilitai/blob/master/Training%20Prototype%20-%20Deep%20Learning%20with%20Images.ipynb) using [Imagenette](https://github.com/fastai/imagenette).


For Imagenette (like for [Imagenet](http://www.image-net.org/)) there are separate training and validation folders, each containing a folder with each object category.
It's easy to extract these labels with Pandas:

```python
df = (pd
   .DataFrame(image_paths, columns=['path'])
   .assign(label=lambda df: df.path.apply(lambda x: x.parent.name),
           split=lambda df: df.path.apply(lambda x: x.parent.parent.name),
           train=lambda df: df.split == 'train')
)
```

Data block provides a variety of common methods to load, label and split data, but in practice you have to tweak the datasets to make it work.
Using Pandas is very flexible and almost as easy.

We'll need to provide transforms that takes the image path, opens the file, resizes it and converts it to a tensor.
Similarly we need a transform to one-hot encode categories.
The transforms are applied lazily, so no images are opened until they're accessed by the dataloader.

```python
img_tfms = [img_open, partial(img_resize, size=128), img_rgb, img_to_float]
img_items = ListContainer(df.path, img_tfms)

cat_tfms = [cat_norm]
cat_items = ListContainer(df.label, cat_tfms)
```

We then label our items using the `combine` function:

```python
items = img_items.combine(cat_items)
```

Then we can split the training and validation set using the `split` method of ListContainers using the dataframe's train mask, and then create a [Pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for each split.

```python
train_items, valid_items = items.split(df.train)

train_dl = DataLoader(train_items, bs, shuffle=True)
valid_dl = DataLoader(valid_items, bs, shuffle=False)
```

The rest of this post will explore how all of this works.

# Transforming Items

To take a description of an item, like the path to an image, and converting it to a Tensor requires transformation.
We don't want to read the image into memory until we have to, so we only execute this transformation when we get a single item.

The `ListContainer` needs to store both the items and the transformations.

```python
class ListContainer(object)
    def __init__(self, items, tfms=None): self.items, self.tfms = list(items), list(tfms or [])
```

We have a method `get` to apply the transforms to an item:

```python
def get(idx): return comply(self.tfms, self.items[idx])
```

Where `comply`, a portmanteau of [`compose`](https://en.wikipedia.org/wiki/Function_composition_(computer_science)) and [`apply`](https://en.wikipedia.org/wiki/Apply).
If we have multiple items (like how a DataFrame has multiple series) then `tfms` is a list of lists of functions, where each list of functions is composed and applied separately to the items:

```python
def comply(functions, x):
    if len(functions) > 0 and isinstance(functions[0], Iterable):
        assert len(functions) == len(x)
        return [comply(f, xi) for (f, xi) in zip(functions, x)]
    for f in functions:
        x = f(x)
    return x
```


# Selecting Objects

ListContainers can be subsetted the same way numpy arrays can, using a list of indices, a boolean mask or a slice.
Subsetting creates another list container.
This can be used for example to extract the training set: `train_items = items[df.train]`

When a single integer index is passed, it calls `get` to apply the transformations.
This will be used by the dataloader.

```python
def __getitem__(self, idx):
    if isinstance(idx, int): return self.get(idx)
    elif isinstance(idx, slice): return self.__class__(self.items[idx], self.tfms)
    # Must be a list
    elif len(idx) > 0 and isinstance(idx[0], (bool, np.bool_)):
        if len(idx) != len(self.items):
            raise IndexError(f'Boolean index length {len(idx)} did not match collection length {len(self.items)}')
        assert len(idx) == len(self.items), "Boolean mask must have same length as object"
        return self.__class__([o for m,o in zip(idx, self.items) if m], self.tfms)
    else: return self.__class__([self.items[i] for i in idx], self.tfms)
```

# List Functionality

There are standard methods provided for the length, iteration, changing and removing items.

```python
    def __len__(self): return len(self.items)
    def __iter__(self): return (self[i] for i in range(len(self.items)))
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
```

# Splitting Test and Training

To be able to split a set into test and training we need to be able to pick out the complement of a selection with `exclude`.
Similar to `__getitem__` it must be able to handle a boolean mask, a slice or a list of indices.

```python
def exclude(self, idxs):
    if isinstance(idxs, slice): idxs = range(len(self))[idxs]
    if len(idxs) == 0: return self
    elif isinstance(idxs[0], (bool, np.bool_)):
        return self[[not x for x in idxs]]
    else:
        return self[[x for x in range(len(self)) if x not in idxs]]
```

Split just combines `__getitem__` and `exclude`.
Because each returns another `ListContainer` the set could be split again to produce separate train, validation and test sets.

```python
def split(self, by):
    return (self[by], self.exclude(by))
```

# Labelling Data

To label the data we just store an additional `LabelList`, similar to `Series` in a `DataFrame`.

```python
    def combine(self, *others):
        for other in others:
            assert len(self) == len(other)
        lists = (self,) + others
        items = zip(*[getattr(l, 'items', l) for l in lists])
        tfms = [getattr(l, 'tfms') for l in lists]
        return self.__class__(items, tfms)
```

There also need to be methods to extract the item from the label.

```python
    def separate_one(self, dim):
        return self.__class__([item[dim] for item in self.items], self.tfms[dim])

    def separate(self):
        dim = len(self.tfms)
        return [self.project_one(i) for i in range(dim)]
```

Note that these methods are very generic, the same idea could be used to combine text and image data.

# Displaying

The items can be displayed at the console, before transformation:

```python
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        if self.tfms: res += f'; Transformed by {self.tfms}'
        return res
```

Generally you also want a datatype specific way to display your data (e.g. show an image or play an audiofile).
This would need to be added by subclassing `ListContainer` for separate instances.

# Preprocessing

Fastai has a separate [preprocessing](https://docs.fast.ai/data_block.html#Invisible-step:-preprocessing) stage that is invisible, which has confused [me and others](https://forums.fast.ai/t/character-level-language-model/31379/2) when trying to build a [character level language model](https://nbviewer.jupyter.org/gist/EdwardJRoss/86b31848a7951411de56f10f55e9de4e).
Any preprocessing would be done explicitly before putting items into the ItemList; since they are to be applied once, to all the data.
It would be straightforward to create an explicit preprocessing function for each data type that returns a corresponding `ListContainer`.

# Transformations in the DataLoader

It's also possible to put [arbitrary transformations](https://nbviewer.jupyter.org/github/EdwardJRoss/agilitai/blob/master/Training%20Prototype%20-%20Deep%20Learning%20with%20Images.ipynb#4.1-We-could-do-the-mappings-in-Collate) in the DataLoader's `collate_fn` argument.

```python
class TransformCollate:
    def __init__(self, tfms=[], collate=torch.stack):
        self.tfms = tfms
        self.collate = collate

    def __call__(self, items):
        return self.collate([comply(self.tfms, item) for item in items])
```

Then you wouldn't need to lazily apply transformation in the `ListContainer`, they would be applied by the `DataLoader` as needed.

```python
img_collate = TransformCollate(img_tfms)
train_x = DataLoader(df.path[df.train], bs, collate_fn=img_collate, shuffle=True)
```

We could also provide a method for combining the collation function for the items with the collation function for the labels.

```python
class ProductCollate:
    def __init__(self, *collates):
        self.collates = collates

    def __call__(self, items):
        items = list(zip(*items))
        assert len(items) == len(self.collates)
        return tuple(collate(item_group) for collate, item_group in zip(self.collates, items))
```

This can be used to get a DataLoader for the labelled items:

```python
xy_collate = ProductCollate(img_collate, cat_collate)
train = DataLoader(list(zip(df.path[df.train], df.label[df.train])),
                   bs,
                   collate_fn=xy_collate,
                   shuffle=True
                  )
```

I personally prefer the transformation to occur in the `ListContainer` because you can easily interactively explore the output of transforms, and because the transforms are stored with the object they don't need to be passed around like the custom collation functions.
But lazy transformations used in `ListContainer` are unusual in Python and could cause confusion.

I'm going to continue working on improving deep learning workflows in my [agilitai](https://github.com/EdwardJRoss/agilitai) library based heavily on fastai v3 part 2.