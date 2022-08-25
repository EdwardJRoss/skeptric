+++
title = "Converting SentenceTransformers to Tensorflow"
tags = ["python", "nlp", "tensorflow"]
date = "2022-08-23T08:00:00+10:00"
feature_image = "/images/sentencetransformers-to-tensorflow.png"
+++




[SentenceTransformers](https://www.sbert.net/) provides a convenient interface for creating embeddings of text (and images) in PyTorch, which can be used for neural retrieval and ranking.
But what if you want to integrate a model trained in SentenceTransformers with other existing models in Tensorflow?
The best solution would be to rewrite the training in Tensorflow, but if you've already spent a lot of time training a model you may want to import it into Tensorflow.
This post will show you how.

This post was generated with a [Jupyter notebook](/notebooks/sentencetransformers-to-tensorflow.ipynb) which you can download if you want to run it yourself.

## Training a SentenceTransformers Model

Let's start with an example model from the [SentenceTransformers Training Examples](https://www.sbert.net/docs/training/overview.html) a bi-encoder consisting of a Transformer embedding, followed by mean pooling, and a single dense layer.
The transformer `model_name` can be almost any [model from HuggingFace](https://huggingface.co/models?library=pytorch&sort=downloads), but for this example we'll use one of the smaller [pre-trained SentenceTransformers models](https://www.sbert.net/docs/pretrained_models.html) tuned to sentence embeddings, [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).


```python
from sentence_transformers import SentenceTransformer, models
from torch import nn

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
max_seq_length = 512
output_dimension = 256

word_embedding_model = models.Transformer(model_name,
                                          max_seq_length=max_seq_length)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_cls_token=False,
                               pooling_mode_mean_tokens=True,
                               pooling_mode_max_tokens=False)

dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                           out_features=output_dimension,
                           activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

(word_embedding_model.get_word_embedding_dimension(),
 pooling_model.get_sentence_embedding_dimension(),
 output_dimension)
```




    (384, 384, 256)



Next we would finetune this model using our own data.
Here we'll just use some dummy sample data and put it into `InputExample`.
If we had more data than fit in memory we could use [memory mapping with PyArrow](https://skeptric.com/pyarrow-sentencetransformers).


```python
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

train_examples = [
    InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
```

Now we can train the model with one of the [SentenceTransformer losses](https://www.sbert.net/docs/package_reference/losses.html).
Using `use_amp` (Automatic Mixed Precision) means we'll get faster throughput and use less GPU memory on a GPU that supports it.


```python
from sentence_transformers import losses


num_epochs = 3
train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=int(len(train_examples) * num_epochs * 0.1),
          use_amp=True,
          show_progress_bar=False)
```

Now we'll save the model to import into Tensorflow; in this example we'll just use a temporary directory.


```python
from tempfile import TemporaryDirectory

output_dir = TemporaryDirectory()

model.save(output_dir.name)
```

## Converting to Tensorflow

This kind of model can be converted into a Keras model in the following steps:

1. Use Huggingface Transformers to load the model into Tensorflow using TFAutoModel
2. Pass the tokenized input and extract the hidden state
3. Mean Pool the Hidden State
4. Pass the output through the dense layer



```python
import sentence_transformers
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel 

def sentencetransformer_to_tensorflow(model_path: str) -> tf.keras.Model:
    """Convert SentenceTransformer model at model_path to TensorFlow Keras model"""
    # 1. Load the Transformer model
    tf_model = TFAutoModel.from_pretrained(model_path, from_pt=True)

    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32)
    
    token_type_ids = tf.keras.Input(shape=(None,), dtype=tf.int32)

    # 2. Get the Hidden State
    hidden_state = tf_model.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    ).last_hidden_state

    # 3. Mean pooling
    mean_pool = tf.keras.layers.GlobalAveragePooling1D()(
        hidden_state
    )
    
    # 4. Dense layer
    sentence_transformer_model = SentenceTransformer(model_path, device="cpu")
    dense_layer = sentence_transformer_model[-1]
    dense = pytorch_to_tensorflow_dense_layer(dense_model)(mean_pool)

    # Return the model
    model = tf.keras.Model(
        dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ),
        dense,
    )

    return model
```

We can convert the Dense model into Tensorflow with a simple mapping of the weights.


```python
TORCH_TO_KERAS_ACTIVATION = {"torch.nn.modules.activation.Tanh": "tanh"}

def pytorch_to_tensorflow_dense_layer(dense_model: sentence_transformers.models.Dense) -> tf.keras.layers.Dense:
    weight = dense_model.linear.get_parameter("weight").cpu().detach().numpy().T
    bias = dense_model.linear.get_parameter("bias").cpu().detach().numpy()

    dense_config = dense_model.get_config_dict()

    return tf.keras.layers.Dense(
        dense_config["out_features"],
        input_shape=(dense_config["in_features"],),
        activation=TORCH_TO_KERAS_ACTIVATION[dense_config["activation_function"]],
        use_bias=dense_config["bias"],
        weights=[weight, bias],
    )
```

Then we can load our Tensorflow model from the output directory.


```python
tf_model = sentencetransformer_to_tensorflow(output_dir.name)

tf_model
```




    <keras.engine.functional.Functional at 0x7f9105a35eb0>



To test it's the same we can check it on a small sample of input text.


```python
tokenizer = AutoTokenizer.from_pretrained(output_dir.name)

input_text = ['SentenceTransformers are groovy']

tf_tokens = tokenizer(input_text, return_tensors='tf')


import numpy as np
assert np.isclose(tf_model(tf_tokens).numpy(),
                  model.encode(input_text),
                  atol=1e-5).all()
```

The rest of this article goes through how this translation works step-by-step, which could be useful if you wanted to expand this to different SentenceTransformer models.

### Checking we can save and load the model

A final check is that we can save and load the model to get the same result.

You may have noticed in the code above for step 2 we called `tf_model.bert`, rather than just `tf_model.` This is [requried to save a TFBertModel](https://github.com/huggingface/transformers/issues/3627#issuecomment-646390974). If you're not using something bert based you may need to use a different method (such as `.transformer`)


```python
with TemporaryDirectory() as tf_output_dir:
    tf_model.save(tf_output_dir)
    tf_model_2 = tf.keras.models.load_model(tf_output_dir)
    
assert np.isclose(tf_model_2(tf_tokens).numpy(),
                  model.encode(input_text),
                  atol=1e-5).all()
```

    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.


    WARNING:absl:Found untraced functions such as embeddings_layer_call_fn, embeddings_layer_call_and_return_conditional_losses, encoder_layer_call_fn, encoder_layer_call_and_return_conditional_losses, pooler_layer_call_fn while saving (showing 5 of 545). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: /tmp/tmp416di2x1/assets


    INFO:tensorflow:Assets written to: /tmp/tmp416di2x1/assets


    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.


    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.


## Understanding the SentenceTransformers Model

Let's start by understanding this particular SentenceTransformer model.
It's made of 3 layers, the Transformer to embed the text, the pooling layer, and a dense layer.


```python
model
```




    SentenceTransformer(
      (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
      (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
      (2): Dense({'in_features': 384, 'out_features': 256, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    )



We can embed a text using the `encode` function


```python
input_text = ['SentenceTransformers are groovy']

embedding = model.encode(input_text)

print(embedding.shape)

embedding
```

    (1, 256)





    array([[-0.23693885, -0.00868655,  0.12708516, ...,  0.02389161,
            -0.03000948, -0.20219219]], dtype=float32)



We can build this up by going through each layer individually.
First we need to tokenize the input:


```python
tokens = model.tokenize(input_text)
tokens
```




    {'input_ids': tensor([[  101,  6251,  6494,  3619, 14192,  2545,  2024, 24665,  9541, 10736,
                102]]),
     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}



The `input_ids` correspond to the tokens of the input text; for this single text we can ignore the `token_type_ids` and `attention_mask`.


```python
model.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
```




    ['[CLS]',
     'sentence',
     '##tra',
     '##ns',
     '##form',
     '##ers',
     'are',
     'gr',
     '##oo',
     '##vy',
     '[SEP]']



We then pass it through the Transformer Layer which gives us a tensor of: `batch_size * seq_length * hidden_dimension`


```python
model[0]
```




    Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 




```python
transformer_embedding = model[0]({k:v.to(model.device) for k, v in tokens.items()})

print(transformer_embedding.keys())

transformer_embedding_array = transformer_embedding['token_embeddings'].detach().cpu().numpy()

print(transformer_embedding_array.shape)

transformer_embedding_array
```

    dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'token_embeddings'])
    (1, 11, 384)





    array([[[-0.10578819, -0.24982804, -0.25226122, ...,  0.07380735,
             -0.16002229, -0.10104472],
            [ 0.08560147, -0.56386817,  0.07879569, ...,  0.53842884,
              0.8103698 , -1.3633531 ],
            [-0.49132693, -0.2222665 , -0.6009016 , ..., -0.6793231 ,
              0.02079807,  0.19803165],
            ...,
            [-0.42034534, -0.35809648, -0.40514907, ...,  0.18629718,
              0.44449466,  0.21497107],
            [ 0.06977252, -0.3880473 , -0.6704632 , ...,  0.1784373 ,
              0.6243761 , -0.39589474],
            [-0.2825999 ,  0.20663676,  0.31645843, ...,  0.6711646 ,
              0.23843716,  0.08616418]]], dtype=float32)



We then use the pooling layer to remove the sequence dimension


```python
model[1]
```




    Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})




```python
pooled = model[1](transformer_embedding)

print(pooled.keys())

pooled_array = pooled['sentence_embedding'].cpu().detach().numpy()

print(pooled_array.shape)

pooled_array
```

    dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'token_embeddings', 'sentence_embedding'])
    (1, 384)





    array([[-0.3000789 , -0.20643215, -0.21641909, ...,  0.1395422 ,
             0.2712665 , -0.10013962]], dtype=float32)



Then finally we pass it through the dense layer to get the final result.

Note that the result overwrites the `sentence_embedding` key in the dictionary


```python
model[2]
```




    Dense({'in_features': 384, 'out_features': 256, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})




```python
dense_output = model[2](pooled)

print(dense_output.keys())

dense_output_array = dense_output['sentence_embedding'].cpu().detach().numpy()

print(dense_output_array.shape)
dense_output_array
```

    dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'token_embeddings', 'sentence_embedding'])
    (1, 256)





    array([[-0.23693885, -0.00868655,  0.12708516, ...,  0.02389161,
            -0.03000948, -0.20219219]], dtype=float32)



This is the same as we got when we called `encode`


```python
assert (model.encode(input_text) == dense_output_array).all()
```

## Importing the model into Tensorflow

We can load the Tokenizer and Tensorflow model using `transformers`.
Transformers converts the model weights from PyTorch to Transformers when we pass `from_pt=True` (and this may not work for some exotic architectures).


```python
from transformers import TFAutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(output_dir.name)

tf_model = TFAutoModel.from_pretrained(output_dir.name, from_pt=True)

del output_dir
```

We can use the Tokenizer to produce tokens as Tensorflow Tensors:


```python
tf_tokens = tokenizer(input_text, padding=True, truncation=True, return_tensors='tf')
tf_tokens
```




    {'input_ids': <tf.Tensor: shape=(1, 11), dtype=int32, numpy=array([[  101,  6251,  6494, ...,  9541, 10736,   102]], dtype=int32)>, 'token_type_ids': <tf.Tensor: shape=(1, 11), dtype=int32, numpy=array([[0, 0, 0, ..., 0, 0, 0]], dtype=int32)>, 'attention_mask': <tf.Tensor: shape=(1, 11), dtype=int32, numpy=array([[1, 1, 1, ..., 1, 1, 1]], dtype=int32)>}




```python
tf_embedding = tf_model(tf_tokens)

print(tf_embedding.keys())
```

    odict_keys(['last_hidden_state', 'pooler_output'])


The `last_hidden_state` is, up to floating point precision error, the same as the output of the first Transformer layer.


```python
assert np.isclose(tf_embedding.last_hidden_state.numpy(), transformer_embedding_array, atol=1e-5).all()
```

However the `pooler_output` is completely different


```python
np.abs(tf_embedding.pooler_output.numpy() - pooled_array).max()
```




    0.8287506



But we can produce a mean pooling manually.

If we used a different combination of pooling layers in SentenceTransformers (such as CLS or Max pooling) we would have to change this.


```python
import tensorflow as tf

tf_pooled = tf.keras.layers.GlobalAveragePooling1D()(tf_embedding.last_hidden_state)

assert np.isclose(tf_pooled.numpy(), pooled_array, atol=1e-5).all()
```

Finally we need to load the dense model into Tensorflow.

We can extract the weight, bias and configuration from the PyTorch model.


```python
dense_model = model[2]

weight = dense_model.linear.get_parameter("weight").cpu().detach().numpy().T
bias = dense_model.linear.get_parameter("bias").cpu().detach().numpy()

dense_config = dense_model.get_config_dict()

print(weight.shape, bias.shape)
dense_config
```

    (384, 256) (256,)





    {'in_features': 384,
     'out_features': 256,
     'bias': True,
     'activation_function': 'torch.nn.modules.activation.Tanh'}



Then we can use this to create a corresponding dense layer in Keras.
If we had more dense layers, or used a differente activation, we'd need to update accordingly.


```python
TORCH_TO_KERAS_ACTIVATION = {"torch.nn.modules.activation.Tanh": "tanh"}

tf_dense = tf.keras.layers.Dense(
        dense_config["out_features"],
        input_shape=(dense_config["in_features"],),
        activation=TORCH_TO_KERAS_ACTIVATION[dense_config["activation_function"]],
        use_bias=dense_config["bias"],
        weights=[weight, bias],
    )

```

Then passing the output through this layer gives us the same result as the original model.


```python
tf_output = tf_dense(tf_pooled)

assert np.isclose(tf_output.numpy(), model.encode(input_text), atol=1e-5).all()
```

This gets us the same result as our function previous, once we wrap it in Keras functional API.

We could in fact train this model, which if we have to use Tensorflow makes more sense than training the SentenceTransformer model.
But SentenceTransformers provides quite a conventient interface to prototype different losses and pooling layers that it may still be useful to use it before converting everything to Tensorflow.
