---
categories:
- nlp
- python
date: '2021-09-11T08:00:00+10:00'
image: /images/translate_samples.png
title: Offline Translation in Python
---

Suppose you want to translate text from one language to another.
Most people's first point of call is an online translation service from one of the big cloud providers, and most translation libraries in Python wrap Google translate.
However the free services have rate limits, the paid services can quickly get expensive, and sometimes you have private data you don't want to upload online.
An alternative is to run a machine translation model locally, and thanks to Hugging Face it's pretty simple to do.

There are some downsides to running these models locally.
The quality of the translations will be lower than the cloud providers, and even they produce very strange results [like translating gibberish into religious prophecies](https://www.vice.com/en/article/j5npeg/why-is-google-translate-spitting-out-sinister-religious-prophecies) and sometimes amusing, like in a [Google translation of Final Fantasy IV](https://legendsoflocalization.com/funky-fantasy-iv/).
The quality is often quite good, but sometimes the output is bizarre or wrong.
It can also be quite slow to run, especially if you don't run it on a GPU.

This article will compare two options; [Argos Translate](https://github.com/argosopentech/argos-translate) (a wrapper for [OpenNMT](https://opennmt.net/)) with [Marian Machine Translation](https://marian-nmt.github.io/).
Argos Translate is a more complete solution, is easier to get set up, and is substantially faster.
However Marian Machine Translation gives better translations, supports more languages, and better supports batch translations.

# Argos Machine Translation

[Argos Translate](https://github.com/argosopentech/argos-translate) is a wrapper around [OpenNMT's](https://opennmt.net/) [CTranslate2 Models](https://github.com/OpenNMT/CTranslate2).
It can provide a web API, a GUI, and a command line interface as well as a Python interface.

Installation is straightforward with `pip install argostranslate`.
You have to [download the model locally](https://www.argosopentech.com/argospm/index/) and then install it with `package.install_from_path`.
For example for a Russian to English model:

```python
# Download the file
import urllib.request
urllib.request.urlretrieve('https://argosopentech.nyc3.digitaloceanspaces.com/argospm/translate-ru_en-1_0.argosmodel', 'translate-ru_en-1_0.argosmodel')

# Install it
from argostranslate import package
package.install_from_path('translate-ru_en-1_0.argosmodel')
```

Following the documentation it seens it's quite hard to directly get a model and you have to iterate through the installed languages.
I had to implement this helper function (but surely there's an easier way??):


```python
from argostranslate import translate

def get_argos_model(source, target):
    lang = f'{source} -> {target}'
    source_lang = [model for model in translate.get_installed_languages() if lang in map(repr, model.translations_from)]
    target_lang = [model for model in translate.get_installed_languages() if lang in map(repr, model.translations_to)]
    
    return source_lang[0].get_translation(target_lang[0])

argos_ru_en = get_argos_model('Russian', 'English')
```

Then I could translate a string with:

```python
argos_ru_en.translate('что слишком сознавать — это болезнь, настоящая, полная болезнь.')
# "I think it's a disease, a real, complete disease."
```

# Marian Machine Translation

We can run a minimal example using the example from [Hugging Face's wrapper](https://huggingface.co/transformers/model_doc/marian.html) of [Marian Machine Translation](https://marian-nmt.github.io/), the engine behind Microsoft Translator.
First you need to [install PyTorch](https://pytorch.org/), then pip install `sentencepiece` and `huggingface`, the you can run:

```python
from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence

class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]
        
        
marian_ru_en = Translator('ru', 'en')
marian_ru_en.translate(['что слишком сознавать — это болезнь, настоящая, полная болезнь.'])
# Returns: ['That being too conscious is a disease, a real, complete disease.']
```

The `Translator` object takes a list of sentences from a source language to a destination language.
The languages are two letter [ISO 639 codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes), above we create a translator from Russian (`ru`) to English (`en`).
This will then get a model and a tokenizer, which will be downloaded if necessary (and weighs around 300MB).
Marian supports a very large list of languages, you can see all the models on [Hugging Face's model hub](https://huggingface.co/Helsinki-NLP).

It is incredible to me that this is so easy (thanks to Hugging Face), that there are so many languages (thanks to Neural Machine Translation), and that the models are freely available (thanks to Helsinki NLP).
I looked into this about 5 years ago and the best open solution was [Moses Machine Translation](http://www.statmt.org/moses/), which I couldn't get set up in a few hours, has worse translations (statistical methods can get quite good, but require a lot of work and tend to work poorly for informal text, such as web based data) and many fewer languages.
The space of language technology has come a remarkably long way in a few years.
However there are some limitations of these models, in terms of the length of text it can translate, how fast it runs and how it responds to unusual punctuation (relative to the data it was trained on).

## Handling Long Text

If you run it on a long text you will get `IndexError: index out of range in self`; this is a limitation of Transformer models where they have a maximum size input.
The model only supports up to 512 tokens (where a token is dependent on the [SentencePiece encoding](https://github.com/google/sentencepiece), most words are made up of at most a few tokens), and if you pass any more it fails.
The `tokens` returned by the tokenizer is a dictionary containing two tensors, the `input_ids` (which are the actual sentencepiece token ids) and `attention_mask` (which seems to be all 1s).
The tensors are rank 2 with shape the number of texts by the *maximum* number of tokens in any of the texts.
For long texts the best thing to do is to break the text apart at sentence boundaries and then paste it back together again.

A robust way to break a long text into sentences is with [Stanza](https://stanfordnlp.github.io/stanza/):

```python
import stanza
 
# First you will need to download the model
# stanza.download('ru')
nlp = stanza.Pipeline('ru', processors='tokenize')

for sentence in nlp.process('Сдается однокомнатная мебелированная квартира квартира. Ежемесячная плата 18 тыс.р. + свет.').sentences:
    print(sentence.text)
    
# Сдается однокомнатная мебелированная квартира квартира.
# Ежемесячная плата 18 тыс.р. + свет.
```

However we lose the space between sentences.
To be able to capture this we need to be able to get both the sentence and the boundary preceeding it.
Start with a generic container for the text and a prefix:

```python
from dataclassess import dataclass

@dataclass(frozen=True)
class SentenceBoundary:
    text: str
    prefix: str
        
    def __str__(self):
        return self.prefix + self.text
```

And then create a SentenceBoundary object that can extract these from a Stanza Document, appending an empty text to get the trailing characters of the document.
We also add methods for getting the non-empty sentences for translation, and for mapping the text through a dictionary of translations.

```python
from __future__ import annotations # For Python 3.7
from typing import List

@dataclass(frozen=True)
class SentenceBoundaries:
    sentence_boundaries: List[SentenceBoundary]
        
    @classmethod
    def from_doc(cls, doc: stanza.Document) -> SentenceBoundaries:
        sentence_boundaries = []
        start_idx = 0
        for sent in doc.sentences:
            sentence_boundaries.append(SentenceBoundary(text=sent.text, prefix=doc.text[start_idx:sent.tokens[0].start_char]))
            start_idx = sent.tokens[-1].end_char
        sentence_boundaries.append(SentenceBoundary(text='', prefix=doc.text[start_idx:]))
        return cls(sentence_boundaries)
    
    @property
    def nonempty_sentences(self) -> List[str]:
        return [item.text for item in self.sentence_boundaries if item.text]
    
    def map(self, d: Dict[str, str]) -> SentenceBoundaries:
        return SentenceBoundaries([SentenceBoundary(text=d.get(sb.text, sb.text),
                                                    prefix=sb.prefix) for sb in self.sentence_boundaries])
    
    def __str__(self) -> str:
        return ''.join(map(str, self.sentence_boundaries))
```

Because the all the texts are put into a single tensor there needs to be enough memory (CPU or GPU) available to store it all.
So we need to [minibatch](/python-minibatching) the sentences into smaller groups.
In fact since it needs to be processed in a rectangular block you should try to process all the shortest texts together and all the longest texts together for best efficiency.
Moreover it's worth caching any repeated texts to stop retranslating.

Putting this all together we get a more robust translator:

```python
class Translator:
    def __init__(self, source_lang: str, dest_lang: str, use_gpu: bool=False) -> None:
        self.use_gpu = use_gpu
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        if use_gpu:
            self.model = self.model.cuda()
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.sentencizer = stanza.Pipeline(source_lang, processors='tokenize', verbose=False, use_gpu=use_gpu)
        
    def sentencize(self, texts: Sequence[str]) -> List[SentenceBoundaries]:
        return [SentenceBoundaries.from_doc(self.sentencizer.process(text)) for text in texts]
                
    def translate(self, texts: Sequence[str], batch_size:int=10, truncation=True) -> Sequence[str]:
        if isinstance(texts, str):
            raise ValueError('Expected a sequence of texts')
        text_sentences = self.sentencize(texts)
        translations = {sent: None for text in text_sentences for sent in text.nonempty_sentences}
    
        for text_batch in minibatch(sorted(translations, key=len, reverse=True), batch_size):
            tokens = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=truncation)
            if self.use_gpu:
                tokens = {k:v.cuda() for k, v in tokens.items()}
            translate_tokens = self.model.generate(**tokens)
            translate_batch = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]
            for (text, translated) in zip(text_batch, translate_batch):
                translations[text] = translated
            
        return [str(text.map(translations)) for text in text_sentences]
```

Note that we set `truncation=True` in the tokenizer, so if a text is too long after breaking it into sentences we just drop the rest of the text rather than failing.

# Translation Quality

Finally sometimes punctuation causes some strange hallucinations from the model.
First consider some pure punctuation:

```python
for translation in marian_ru_en.translate(['', '.', '!', '-', '&']):
    print(translation)
```

MarianMT gives some rather creative translations:

```
It's okay. It's okay, it's okay, it's okay.
I don't know.
Hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey, hey.
- Yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah, yeah.
♪ I don't know ♪
```

Argos does much better with most of these translating them by copying, but still translates `'.'` to `♪` and `- -` to `[Grunts]`.
It also seems to truncate repeated punctuation.

For some examples of how it performs I tested them on text from the [Avito Demand Prediction competition](https://www.kaggle.com/c/avito-demand-prediction), which is in Russian (not a language I know well).
Here's some examples from titles (which are difficult because they're fragments).
Generally Google Translate is pretty close to the actual meaning (verified by the images), then Marian captures less of the actual meaning, and Argos less again.

| Text                                          | MarianMT Translation                             | ArgosMT Translation                  | Google Translate                             |
|-----------------------------------------------|--------------------------------------------------|--------------------------------------|----------------------------------------------|
| Кокоби(кокон для сна)                         | Cocoby (sleep cocoon)                            | Cocobi(s)                            | Cocobi (sleeping cocoon)                     |
| Стойка для Одежды                             | Clothes stand                                    | Clothing for clothing                | Clothes Rack                                 |
| Philips bluray                                | Philips bluray                                   | Philips bluray                       | Philips bluray                               |
| Автокресло                                    | Truck, car seat                                  | Road                                 | Car seat                                     |
| ВАЗ 2110, 2003                                | VAZ 2110, 2003                                   | WAZ 2110, 2003                       | VAZ 2110, 2003                               |
| Авто люлька                                   | Car bulwark                                      | Truck                                | Car carrycot                                 |
| Водонагреватель 100 литров нержавейка плоский | The water absorber is 100 litres stainless flat. | 100 litres of dynamometer flat       | Water heater 100 liters stainless steel flat |
| Бойфренды colins                              | Collins Boyfriends                               | Colins                               | boyfriends colins                            |
| Платье                                        | Clothes                                          | Plato.                               | Dress                                        |
| Полу ботиночки замш натур.Бамбини             | Half the boots are for nature. Bambini.          | Half the boots are straight. Bambini | Semi boots suede nature Bambini              |
| 1-к квартира, 25 м², 2/2 эт.                  | One-to-one apartment, 25 m2, 2/2 ot.             | 1st apartment, 25 m2, 2/2.           | 1-room apartment, 25 m², 2/2 fl.             |
| Джинсы                                        | Jeans.                                           | Jeans                                | Jeans                                        |
| Атласы и Контурныя карты за 8 класс           | Atlass and end-of-grade maps                     | Atlas and Contourna Cards 8 class    | Atlases and Contour maps for grade 8         |
| Монитор acer 18.5                             | Monitor acer 18.5                                | Monitor acer 18.5                    | acer 18.5 monitor                            |
| Продаются щенки немецкой овчарки              | German shepherd's puppies are for sale.          | They sell German sheep               | German Shepherd puppies for sale             |
| Платье женское новое                          | A woman's new dress.                             | Women's new dress                    | Women's dress new                            |
| Chevrolet Lanos, 2008                         | Chevrolet Lanos, 2008                            | Chevrolet Lanos, 2008                | Chevrolet Lanos, 2008                        |
| Объемная цифра 2                              | Volume 2                                         | Volume 2                             | 3D figure 2                                  |
| Куртка весенняя(осенняя)                      | Spring jacket (Spring)                           | Spring(s)                            | Jacket spring (autumn)                       |
| Сниму коттедж                                 | I'll take the cottage off.                       | I'll take the cattage.               | Cottage for rent                             |


## Effect of punctuation

The MarianMT model is very sensitive to punctuation and capitalisation, the OpenNMT model used by Argos is less sensitive to punctuation but is sensitive to tokenisation.
This is likely because the underlying sentencepiece tokenizer doesn't treat these specially; this makes it incredibly flexible for languages with non-European punctuation and tokenisation (for example Thai, Arabic and Mandarin Chinese).
However it means it performs less well with strange punctuation and capitalisation.
Consider the following description fragment from the Avito competition:

> Чтобы посмотреть весь ассортимент нашего магазина перейдите по ссылке в блоке справа ⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒/
>
> НАЛИЧИЕ ТОВАРА УТОЧНЯЙТЕ ПО КОТАКТНОМУ ТЕЛЕФОНУ./
>
> Продам Кулер для компьютера COOLER MASTER/
>
> /
>
> В НАШЕМ МАГАЗИНЕ НА ТЕХНИКУ ДАЕТСЯ ГАРАНТИЯ!!!!!!/
> ========================================/

Google Translate gives a plausible translation:

> To view the entire range of our store, follow the link in the block on the right ⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒⇒ /
>
> CHECK OUT THE AVAILABILITY OF THE GOODS BY CONTACT PHONE. /
>
> Selling Cooler for computer COOLER MASTER /
>
> /
>
> IN OUR STORE IT IS GIVEN A WARRANTY !!!!!! /
>
> ======================================== / 

### MarianMT

MarianMT (without breaking into sentences) loses a lot of the punctuation information, it drops the middle sentence about contacting by telephone, and misses some translations (МАГАЗИНЕ should be shop or store like in the Google translate, but it's been just transliterated to magasine)

> To look at the entire range of our store, you can cross-reference in the right-hand-hand box to sell the Cooler Master/// in our magasine to technic to produce garantium.

It's very sensitive to the punctuation, if we just remove the last forward slash we get a bunch of hallucinated punctuation:

> To look at the entire range of our store, I want you to cross-reference to the right-hand box in our computer, COLER MASTER// IN TECHNOLOGY, GARANTIA!!!!!!!!!!!!/=======================================================================================)============)========================================)=========================================== )))))))))))))))))) )))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

This could be due to sentencepiece's encoding being impacted by the punctuation (you can see Google translated broke apart the forward slashes from the spaces). What if we remove the extra punctuation?

> Чтобы посмотреть весь ассортимент нашего магазина перейдите по ссылке в блоке справа. 
>
> НАЛИЧИЕ ТОВАРА УТОЧНЯЙТЕ ПО КОТАКТНОМУ ТЕЛЕФОНУ.
>
> Продам Кулер для компьютера COOLER MASTER.
>
> В НАШЕМ МАГАЗИНЕ НА ТЕХНИКУ ДАЕТСЯ ГАРАНТИЯ!!!!!!

Strangely MarianMT drops all but the first line (and it's not because of the newlines):

> To look at the entire range of our store, you can cross the link in the block on the right.

Translating it line by line gives a better result, but it's more literal:

> To see the full range of our store, cross the link in the block on the right.
>
> Cash the product, get it off the phone.
>
> I'll sell Cooler for COOLER MASTER.
>
> THE TECHNOLOGY GUARANTIES IN OUR MAGAZINE!!!!!!!

If we first lowercase it all (using `.lower()` in Python) the line-by-line translation gets a little better:

> To see the entire range of our store, cross the link in the block on the right.
>
> Check whether the product is available on the kitty phone.
>
> I'm gonna sell a cooler master cooler.
>
> in our hardware store there's a guarantee!!!!!!!!!!!

### Argos Translate

Argos Translate gives a readable translation that drops some punctuation, and after the first sentence is mostly nonsense, and hallucinates a final line.
Curiously translating `========` with MarianMT also gives a similar translation `== sync, corrected by elderman == @elder_man'`.

> In order to look at the entire range of our stores, go through the reference to the right of the right, the State states that it states:
>
> I'd like you to take a look at the TEST. /
>
> I'll sell Couler for COOLER MASTER/
>
> /
>
> I've got a lot of guitar! /
>
> == sync, corrected by elderman ==

Removing the extra punctuation doesn't change the translation content much, except removing the hallucination.

> To see our store's entire range, you're gonna have to go to the right block. I'd like you to take a look at the TEST.
>
> I'll sell Couler for the COOLER MASTER computer.
>
> I'm in the middle of a guitar!

Passing the sentences separately gives the same result.
First lower-casing all the words gives a very slightly better translation:

> To've seen the store's all sorts of stuff, you're gonna have to go to the right block.
>
> If there's a merchandise, please click on the cable phone.
>
> I'll sell the cooler master computer.
>
> We've got security at our hardware store!


# Speed

The speed of translation can be slow for a large amount of text, but it is trivially paralellisable.
Running MarianMT on my CPU for titles (typically a few words/tokens) I could get through about 1000 per 15 minutes of CPU time (on my laptop).
Descriptions, which average around 27 words, would take about 25 CPU minutes per 100.
So to process on a single CPU the whole 1.5 million Avito titles would take about 2 weeks, and the descriptions almost a year.

It automatically runs on multiple cores, and you could distribute it between multiple machines.
Running on a GPU is around 50 times faster; on a Kaggle GPU I could get through 1000 titles in under 30 seconds, and 100 descriptions in 40 seconds (tuning the mini-batch size appropriately), so you could translate all 1.5 million descriptions in about a week.

Argos is around 3 times faster on CPU than MarianMT, however it doesn't automatically parallelise and the object can't be pickled so it's hard to use with multiprocessing.
It can be used [with GPU acceleration](https://github.com/argosopentech/argos-translate/#gpu-acceleration), but since objects can only be passed one string at a time I can't see a way to process batches at once.
So for batch processing large amounts of text with OpenNMT it may be better to use the underlying [CTranlate2](https://github.com/OpenNMT/CTranslate2) model directly.

# Conclusion

I think it's amazing that in a relatively short amount of time you can get reasonable machine translation on a commodity PC.
There's some hoops you have to jump through to get it to work, and it's a little bit slow on CPU, but it's great that it's even possible and that these models are in the open.
You couldn't use it to replace a professional machine translation service (without at least some fine tuning on a more specific dataset), but it's definitely good enough to be useful.
It would be interesting to test it out on some low resource languages to see how well the models perform.