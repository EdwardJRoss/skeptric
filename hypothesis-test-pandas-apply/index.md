---
categories:
- pandas
- python
- testing
date: '2021-07-03T08:00:00+10:00'
image: /images/hypothesis_pandas_series.png
title: Testing Pandas transformations with Hypothesis
---

Pandas and numpy let you perform fast transformations on large datasets by executing optimised low-level code.
However the syntax is very terse and it can quickly become hard to see what it's doing.
Often it's clearer in pure Python code, but Pandas `apply` function is much slower.
Hypothesis gives a way to check they are doing the same thing.

For example I've got some code where I've got a salary, but I don't know whether the rate is hourly, daily or annual.
I want to infer it from the code from some rules and return the number of hours it refers to.
I can compare a version that works on Pandas series `series_infer_salary_period_hours` with one that works on individual salaries `infer_salary_period_hours` as follows:

```python
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.pandas import series
from pandas.testing import assert_series_equal

@given(series(elements=(floats(0, 500_000))))
def test_infer_salary_period_hours_apply(s):
    assert_series_equal(series_infer_salary_period_hours(s),
                        s.apply(infer_salary_period_hours).astype('Int64'))
```

Things to note are that we restrict the elements to a reasonable range, and have to be careful with the types.

We can also check that it works the same on a one-element series.
In this case it's easy to check special edge cases at the boundaries using the `@example` decorator.

```python
from hypothesis import example, given
from hypothesis.strategies import floats
import pandas as pd

@given(floats(0, 500_000))
@example(15)
@example(100)
@example(300)
@example(1000)
@example(20_000)
def test_infer_salary_period_hours_element(s):
    s_series = pd.Series([s])
    series_ans = series_infer_salary_period_hours(s_series).iloc[0]
    ans = infer_salary_period_hours(s)
    assert ans == series_ans or (ans is None and pd.isna(series_ans))
```

Note that we have to be a bit careful about how we check `None` which is converted to `nan` by Pandas, which is not equal to any other `nan`.

In general it can be useful to check a Numpy or Pandas row level function against a scalar function written in vanilla Python.

Here's a full extract of this example:


```python
from hypothesis import example, given
from hypothesis.strategies import floats
from hypothesis.extra.pandas import series
from typing import Optional
import pandas as pd
from pandas.testing import assert_series_equal

def infer_salary_period_hours(salary: float) -> Optional[int]:
    """Infer salary period from a salary.

    Returns None if can't infer a period.
    """
    if 15 <= salary <= 100:
        # Likely hourly rate
        return 1
    elif 300 <= salary <= 1000:
        # Likely daily rate
        return 40
    elif salary >= 20_000:
        # Likely annual
        return 2_000

def series_infer_salary_period_hours(s: pd.Series) -> pd.Series:
    ans = pd.Series(None, s.index, dtype='Int64')
    ans[s.between(15, 100)] = 1
    ans[s.between(300, 1000)] = 40
    ans[s >= 20_000] = 2_000
    return ans

@given(series(elements=(floats(0, 500_000))))
def test_infer_salary_period_hours_apply(s):
    assert_series_equal(series_infer_salary_period_hours(s),
                        s.apply(infer_salary_period_hours).astype('Int64'))

@given(floats(0, 500_000))
@example(15)
@example(100)
@example(300)
@example(1000)
@example(20_000)
def test_infer_salary_period_hours_element(s):
    s_series = pd.Series([s])
    series_ans = series_infer_salary_period_hours(s_series).iloc[0]
    ans = infer_salary_period_hours(s)
    assert ans == series_ans or (ans is None and pd.isna(series_ans))
```
