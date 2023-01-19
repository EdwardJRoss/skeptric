---
categories:
- nlp
date: '2023-01-19T20:23:44+10:00'
image: mlm_loss.jpg
title: Why pretrain your own language model?
---

I think now is a great time to pretrain your own language model from scratch.
This may be a strange statement when all the best performing models today are hundreds of billions of parameters trained on specialised systems with gigantic datasets run on large expensive cluster of GPU machines for very long periods of time.
However even with hundreds of millions of parameters it's possible to train models that were state of the art 4 years ago, and are very useful for fine-tuning for specific tasks and running in production.
Fine-tuning from scratch gives the ability to modify the tokenizer in ways more suitable for a task, to carefully select corpora appropriate to the task, use different model types (e.g. sparse attention), and gives a better intuition for domain adaptation.
Moreover scaling laws mean that we can experiment with small pre-training to find the best model that would scale up to many tokens.

Training a language model is easier than ever before; the [Cramming paper (Geiping and Goldstein)](https://arxiv.org/abs/2212.14034) trains a model with performance comparable to [BERT-Base (Devlin et al)](https://arxiv.org/abs/1810.04805) in one day on an A6000 GPU.
Here's a table from their paper comparing their Crammed BERT (with 24 hours of GPU pre-training, which would cost around $24 USD on a second tier cloud provider) to BERT and Roberta on the GLUE tasks (minus Winograd NLI).

|                           | MNLI      | SST-2 | STSB | RTE  | QNLI | QQP  | MRPC | CoLA | GLUE |
|---------------------------|-----------|-------|------|------|------|------|------|------|------|
| BERT-Base (Fully trained) | 83.2/83.4 | 91.9  | 86.7 | 59.2 | 90.6 | 87.7 | 89.3 | 56.5 | 80.9 |
| BERT-Base (No Pretrain)   | 34.1/34.1 | 79.9  | 17.8 | 47.3 | 50.0 | 68.6 | 77.9 | 0.0  | 45.5 |
| ROBERTA-Base              | 86.6/86.4 | 93.7  | 90.4 | 77.3 | 92.1 | 88.3 | 91.4 | 60.2 | 85.1 |
| Crammed BERT (A6000)      | 83.9/84.1 | 92.2  | 84.6 | 53.8 | 89.5 | 87.3 | 87.5 | 44.5 | 78.6 |

While Crammed BERT does significantly worse than BERT on RTE, MRPC, and CoLA, these are known to be highly variable and can likely be improved using the fine-tuning procedure advocated in [On the Stability of Fine-Tuning Bert (Mosbach, Andriushchenko, and Klakow)](https://arxiv.org/abs/2006.04884).
There's likely more improvements that can be made; for example the recent [NarrowBERT (Li, et al.)](https://arxiv.org/abs/2301.04761) only uses attention on masked tokens which increases pretraining speed by 1.3-2.5x with only a small reduction in performance.
These procedures could be used on different training sets more appropriate to a downstream task, and could help with further pretraining on domain specific data which is generally beneficial as shown in [Don't Stop Pretraining (Gururangan et al.)](https://arxiv.org/abs/2004.10964).

Decoder models can be trained from scratch as well; Andrej Karpathy has released [nanoGPT](https://github.com/karpathy/nanoGPT) which he claims can match the [base GPT-2 model](https://openai.com/blog/better-language-models/) with 1 day of training on 8 x A100 40GB (which is ~$250 USD on a second tier cloud provider).
The model and training are each 300 lines of code, and Karpathy released a detailed [video tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) which makes it easy to try ways to improve it.

These costs are relatively moderate but could add up when iterating on models.
However there are fairly robust [Scaling Laws for Neural Lanugage Models (Kaplan, et al.)](https://arxiv.org/abs/2001.08361) and [for Transfer (Hernandez et al.)](https://arxiv.org/abs/2102.01293) refined in [Training Compute-Optimal Large Language Models (Hoffman et al.)](https://arxiv.org/abs/2203.15556) and also seen in the Crammed BERT paper.
These mean that we can find optimal ways to train language models on much smaller compute budgets (iterating on the data, the model, and training procedure) and then scale them up when we have a good approach.

Most people don't have the resources to train a very large language model from scratch (and it wouldn't be good for the environment if they did).
While Google, OpenAI, and Microsoft are increasingly keeping their large language models private behind a paid API there are other initiatives like [EleutherAI](https://blog.eleuther.ai/why-release-a-large-language-model/) and [BigScience](https://bigscience.huggingface.co/) releasing very large language models.
But I think smaller language models still have a lot of value, and are much easier to use and adapt in a production setting; even keeping them up to date like the [Online Language Modelling initiative](https://github.com/huggingface/olm-datasets) is worthwhile.
They aren't as powerful as the very Large Language Models, but by combining them with search or logic I suspect they will be very effective in production.
