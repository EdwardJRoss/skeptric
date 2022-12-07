---
categories:
- nlp
- python
- ner
date: '2022-04-09T22:47:00+10:00'
image: /images/ner_ingredient_example.png
title: Training Recipe Ingredient NER with Transformers
---

I trained a Transformer model to predict the components of an ingredient, such as the name of the ingredient, the quantity and the unit.
It performed better than the benchmark CRF model when using fewer data examples, and on the full dataset performed similarly but was useful for identifying issues in the data annotations.
It also has some success even on languages that it wasn't trained on such as French, Hungarian, and Russian.
It took several hours to put together with no prior experience, and minutes to train for free on a Kaggle notebook.
You can [try it online](https://huggingface.co/edwardjross/xlm-roberta-base-finetuned-recipe-all) or see the [training notebook](https://github.com/EdwardJRoss/nlp_transformers_exercises/blob/master/notebooks/ch4-ner-recipe-stanford-crf.ipynb)<a href="https://kaggle.com/kernels/welcome?src=https://github.com/EdwardJRoss/nlp_transformers_exercises/blob/master/notebooks/ch4-ner-recipe-xlm-roberta.ipynb"><img style="display: inline;" src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>.

The underlying training and test data is from [A Named Entity Based Approach to Model Recipes](https://arxiv.org/abs/2004.12184), by Diwan, Batra, and Bagler.
They manually annotated a large number of ingredients from AllRecipes.com and FOOD.com with the tags below.

| Tag       | Significance                          | Example        |
|-----------|---------------------------------------|----------------|
| NAME      | Name of Ingredient                    | salt, pepper   |
| STATE     | Processing State of Ingredient.       | ground, thawed |
| UNIT      | Measuring unit(s).                    | gram, cup      |
| QUANTITY  | Quantity associated with the unit(s). | 1, 1 1/2 , 2-4 |
| SIZE      | Portion sizes mentioned.              | small, large   |
| TEMP      | Temperature applied prior to cooking. | hot, frozen    |
| DRY/FRESH | Fresh otherwise as mentioned.         | dry, fresh     |

I have [previously replicated their benchmark](/stanford-ner-python) using Stanford NER, a Conditional Random Fields model.
Here are the f1-scores reported in the paper (columns are training set, rows are testing set).

| Benchmark - Paper | AllRecipes | FOOD.com | BOTH   |
|-------------------|------------|----------|--------|
| Testing Set       |            |          |        |
| AllRecipes        | 96.82%     | 93.17%   | 97.09% |
| FOOD.com          | 86.72%     | 95.19%   | 98.48% |
| BOTH              | 89.72%     | 94.98%   | 96.11% |

While these may look impressive, using the model of predicting the most common label for each token, and O for out of vocabulary tokens, gets an f1-score over 92%.
This is actually quite a simple problem because most tokens have a label and ambiguity is rare.

I followed the process of training an NER with transformers from Chapter 4 of [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) by Tunstall, von Werra, and Wolf (using their [public notebooks](https://github.com/nlp-with-transformers/notebooks) as a guide).
There was marked improvement on using the smaller AllRecipes dataset (1470 training samples).
However on the larger FOOD.com dataset (5142 training samples) the increase in performance was smaller, and on the combined dataset it was very marginal.
Note this is keeping a validation set, and using the default hyperparameters from the text; I haven't tried to optimise it at all or use every data point.

| Transformer (XLM Roberta) | AllRecipes | FOOD.com | BOTH   |
|---------------------------|------------|----------|--------|
| Testing Set               |            |          |        |
| AllRecipes                | 96.94%     | 95.73%   | 97.34% |
| FOOD.com                  | 91.64%     | 96.04%   | 95.77% |
| BOTH                      | 92.9%      | 95.96%   | 96.15% |

I suspect the reasons it doesn't do much better are because of inconsistencies in the annotation and hitting the limits.
Running an error analysis, as per the NLP with transformers text, showed some issues.
Often only the first of multiple ingredients is annotated.

| token | 1        | teaspoon | orange | zest | or | 1 | teaspoon | lemon | zest |
|-------|----------|----------|--------|------|----|---|----------|-------|------|
| label | QUANTITY | UNIT     | NAME   | NAME | O  | O | O        | O     | O    |

In this case all but the last ingredient name is annotated.

| token  | 1/4      | cup  | sugar | , | to | taste | ( | can  | use | honey | , | agave | syrup | , | or | stevia | ) |
|--------|----------|------|-------|---|----|-------|---|------|-----|-------|---|-------|-------|---|----|--------|---|
| labels | QUANTITY | UNIT | NAME  | O | O  | O     | O | UNIT | O   | NAME  | O | NAME  | NAME  | O | O  | O      | O |

The inconsistencies confused both me and the model.
There are instances in both "firm tofu" and "firm tomatoes" where firm is considered part of the name, and others where it is part of the state.
Similarly in "stewing beef", stewing is sometimes a state and sometimes part of the name.
Though there were real issues in the model; it couldn't distinguish "clove" in "garlic clove" (a unit), from "ground cloves" (a name).

An amazing thing about using a multilingual transformer model like XLM Roberta is it has some zero-shot cross-language generalisation.
Even though all the examples are English it does better than random on other languages.
Admittedly the pattern of ingredients makes it easier (e.g. a numerical quantity, followed by a unit, followed by a name), but it picked up some other things.
I didn't have a dataset to test on, but tried it on a few examples I could find.
If you want to try more you can try it in the [Huggingface model hub](https://huggingface.co/edwardjross/xlm-roberta-base-finetuned-recipe-all) and share what you find.

As you might expect it does well on a French example, where there's a lot of similar vocabulary.
However any model relying on token lookups would not be able to learn this from the training data.

| token       | 1          | petit  | oignon | rouge  |
|-------------|------------|--------|--------|--------|
| translation | 1          | small  | onion  | red    |
| actual      | I-QUANTITY | I-SIZE | I-NAME | I-NAME |
| prediction  | I-QUANTITY | I-SIZE | I-NAME | I-NAME |


Going a bit further afield to Hungarian it certainly does better than random.
Here's an example where it only makes a mistake on one entity; but picks up that fagyasztott is not part of the name.

| token       | 1        | csomag | fagyasztott | kukorica |
|-------------|----------|--------|-------------|----------|
| translation | 1        | packet | frozen      | corn     |
| actual      | QUANTITY | UNIT   | TEMP        | NAME     |
| prediction  | QUANTITY | UNIT   | STATE       | NAME     |

Here's another Hungarian example where it gets the name wrong because it missed the unit (konzerv).

| token       | 50       | dkg       | kukorica | konzerv |
|-------------|----------|-----------|----------|---------|
| translation | 50       | dkg (10g) | corn     | canned  |
| actual      | QUANTITY | UNIT      | NAME     | UNIT    |
| prediction  | QUANTITY | UNIT      | NAME     | NAME    |

However here's a harder example that it gets precisely right.

| token       | őrölt  | fehér | bor    |
|-------------|--------|-------|--------|
| translation | ground | white | pepper |
| actual      | STATE  | NAME  | NAME   |
| prediction  | STATE  | NAME  | NAME   |

Russian should be even harder since it's a different script, although is straightforward to transliterate.
However here's an example that it gets exactly right:

| token       | Сало | свиное | свежее | - | 50 | г      |
|-------------|------|--------|--------|---|----|--------|
| translation | fat  | port   | fresh  | - | 50 | g      |
| actual      | NAME | NAME   | DF     | O | O  | I-UNIT |
| prediction  | NAME | NAME   | DF     | O | O  | I-UNIT |

If one wanted to extend this model to one of these other languages the existing predictions would be a good way to start.
Then annotators could correct the mistakes, especially where the model is unsure, which is much faster than manually labelling every token.
In this way a good training set could be constructed relatively quickly by bootstrapping from another language.
For more ideas on dealing with few to no labels, see Chapter 9 of the [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) book.

To take the model further we could fix the annotation errors, in particular multiple annotations within an ingredient, and retrain the model.
We could also annotate more diverse ingredient sets; the NY Times released a similar [ingredient phrase tagger](https://github.com/NYTimes/ingredient-phrase-tagger) along with training data (and the corresponding [blog post](https://open.blogs.nytimes.com/2015/04/09/extracting-structured-data-from-recipes-using-conditional-random-fields/) is informative).
However the tagger is already really very good.

Though really the model is really good and a better thing to do would be to run it over a large number of recipe ingredients to extract information.
There are many recipes that can be extracted from the internet; for example using [Web Data Commons](http://webdatacommons.org) extracts of Recipes, [recipes subreddit](https://www.reddit.com/r/recipes/) (via [pushshift](https://pushshift.io/)), or [exporting](https://en.wikibooks.org/wiki/Special:Export) the [Cookbook wikibook](https://en.wikibooks.org/wiki/Cookbook:Table_of_Contents), or using [OpenRecipes](https://github.com/fictivekin/openrecipes) (or their [latest export](https://s3.amazonaws.com/openrecipes/20170107-061401-recipeitems.json.gz)).
Practically the CRF model is likely a better choice since it works roughly as well and would run much more efficiently.
Then you could look at which ingredient names occur together, estimate nutritional content, convert quantities by region or do more complex tasks like suggest ingredient substitutes or generate recipes.
