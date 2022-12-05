---
categories:
- general
date: '2020-12-01T19:54:00+11:00'
image: /images/alphafold.webp
title: 'AlphaFold: Predicting protein shape from its composition'
---

The [Critical Assessment of protein Structure Prediction (CASP)](https://predictioncenter.org/) runs every two years to predict the shape of a protein, the building blocks of life, from its sequence of amino acids.
We know the shape of a bunch (around 170,000) of proteins from techniques like X-ray crystallography and Magnetic Resonance Imaging, but it's a big experimental job to actually measure this.
However we know the sequence of millions of proteins due to cheap DNA sequencing and the [DNA to protein translation](https://www.nature.com/scitable/topicpage/translation-dna-to-mrna-to-protein-393/). 
Understanding the shape of a protein is one step towards understanding its function; how it interacts and binds with other substances.

Over the past 2 CASP competitions there have been big leaps forward from DeepMind's AlphaFold and in the recent competition [they produced a model that gets very close to experimental error](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology).
I'm normally sceptical when there's a big press release about Deep Learning solving something, but in this case it's a really hard problem and a big breakthrough.
There's commentary in [science magazine](https://www.sciencemag.org/news/2020/11/game-has-changed-ai-triumphs-solving-protein-structures) that comments on the impact.

This could be a precursor for understanding other complex systems where we understand some aspect of the structure but not the emergent properties.
For example in materials science a challenge is to understand the properties of a material (such as its malleability, heat resistance and conductivity) given its chemical structure.
Or trying to predict specific modes of brain activity from messy MRIs (or even EEGs).
Perhaps we could even improve our already very good predictions of weather patterns.

It would be really interesting to understand more about their approach and get an indication about whether it would generalise to other domains.
DeepMind weren't the only team to use deep neural networks; what did they do differently?
How did they incorporated the physical and chemical knowledge into the framing of the problem?