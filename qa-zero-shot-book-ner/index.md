---
categories:
- nlp
- ner
- hnbooks
date: '2022-06-28T13:40:15+10:00'
image: /images/qa_book_ner.png
title: Question Answeeing as Zero Shot NER for Books
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline).
I've [previously found](/ask-hn-book-recommendations) book recommendations for Ask HN Books, and [have used](/book-ner-work-of-art) the `Work of Art` named entity from Ontonotes to detect the titles.
Another approach is to use extractive question answering as a sort of zero-shot NER.
This works amazingly well, at least providing that there is an actual book title there.

The code is simple using Transformers high level [Question Answering Pipeline](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline).
I picked the first question that came to mind; some prompt engineering may produce better results.

```python
from transformers import pipeline

pipe = pipeline("question-answering")

pipe(context=books[0],
     question="What book is this about?",
     topk=5,
     handle_impossible_answer=True)
     
pipe(context=books[0],
     question="Who is the author?",
     topk=5,
     handle_impossible_answer=True)
```

These work very well when there is just one answer, the book answer is typically reliable above 0.5.
When there's more than one answer it often only captures one, and when there's no answers it will sometimes force one (especially for author).
Often the secondary answers will overlap with the top answer; we'd need to do some more filtering to get distinct answers.

I found contextualising the author could help; first find a book name then ask `"Who is the author of <book>?"`.
Subjectively this worked better than the Work of Art NER when there was a book.
If you want to see some examples check out the [example notebook](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0021-question-answering-for-title-extraction.ipynb).