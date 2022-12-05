---
categories:
- data
- whatcar
date: '2020-11-07T16:24:07+11:00'
image: /images/whatcar_annotate.png
title: Importance of Collecting You Own Training Data
---

A couple years ago I built [whatcar.xyz](http://whatcar.xyz) which predicts the make and model of Australian cars.
It was built mainly with externally sourced data and so only works sometimes, under good conditions.
To make it better I've started collecting my own training data.

External data sources are extremely convenient for training a model as they can often be obtained much more cheaply than curating your own data.
But the data will almost always be different to what you are actually performing inference on, and so you're relying on a certain amount of generalisation.
While techniques like data augmentation can reduce the gap ultimately the best way to train a very good model is to also add your own labelled data.

Using external data gets a good start, but it doesn't work as well on out of domain data.
For Whatcar I used a Resnet50 model that was pretrained on Imagenet, and then added a custom classification head and fine tuned it on a few thousand example images of Australian car makes and models (for the top 400 makes and models).
This definitely works reasonably well; if I'm careful to frame the car properly in a landscape image it correctly predicts the model about half the time.
However if there's some objects in the background, or I take the photo in portrait then it does pretty poorly.
This is to be expected since most labelled images of cars will be well framed in landscape; these photos are out of domain.

Further fine tuning the model on real images I've taken should get better results.
In the [Don't Stop Pretraining](/dont-stop-pretraining) paper it was shown for NLP models that the best outcomes came from training on a massive corpus of language data, then further training it on a large amount of in domain data and then further training again on the specific data to be classified.
What I'm proposing here is a very close analogy; with Imagenet being the massive dataset, external data like VMMRdb being the in domain data and finally using the images taken directly.
The difference with NLP is they use language models all the way through; here we're using different classification models.
It would be interesting to try [self supervised computer vision methods](https://www.fast.ai/2020/01/13/self_supervised/) to see if they could make it even more data efficient.

So to this end I needed to make it easy to collect data for further use.
I made a simple endpoint to my server where an image can be taken and annotated with a make and a model from dropdowns and then uploaded.
The image is uploaded to a blob store and the annotation and metadata, such as image blob store location and uploader information, is stored in a database.
I can then manually collect and label data on my phone; it's ideal to do this at the same time because I can manually spot the badge which may not be visible from the photograph.

I'm going to continue collecting training data and test whether it can make the model better.
One limitation of collecting all the data on my mobile phone is I may overfit to the lens of my camera; it may not generalise well to other cameras.
It may also overfit to cars in my survey area, I may inadvertently capture those cars or types of cars, many times over.
Ideally I'd get lots of annotations from different people on different devices in different locations, and if I had a commercial application I would pay for these.

This model definitely works in cases other than car classification.
The benefit of having a server is you can get domain experts in the field to annotate the data on the spot.

There could be more time efficient methods of obtaining lots of training data.
For example taking a video walking around each car would get a lot of different angles.
But it would require a lot more storage, transfer and processing.