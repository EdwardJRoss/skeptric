---
categories:
- sql
- data
date: '2022-05-11T19:06:58+10:00'
image: /images/point_in_time_join.png
title: Point-in-time joins and real time feature stores
---

Going from batch processing to near-real time applications is a big conceptual leap for data scientists.
Data scientists are often familiar with big SQL analytics databases and can run a batch process weekly or daily.
However to get good performance in some applications requires aggregating information that changes more quickly than batch processes can handle.
There is normally some lag time between an event being processed and being ingested into an analytics database (and this lag time can vary across the data) which limits the batch approach.
A near-real time approach, where the data is up to date between milliseconds and minutes, requries a suitable way to store and retrieve the data in a low latency way.
This data store can be called an online feature store; it contains the *features*, or inputs, for the model and is updated live.

When we've got an online feature store we need to be careful that we don't use it to train and evaluate new models.
We want the data *as it would have been when the prediction was made*, this means we need to record all the results in an offline analytics store.
However if we change the features of the model we want to use the current version of the features, not the version at the time the prediction was made.
Let's illustrate this with an example.

Consider a recommendation system for articles on this website.
On each article we are going to recommend 3 other articles to suggest, to help the reader find other relevant content.
Using our historical logs we see that looking at the previous page viewed can lead to substantially better recommendations than just using the current page viewed.
In particular we build a model that extracts key terms from the previous page viewed and the current page viewed, then these are used to form a query to return relevant pages from a database.
However our logs are only updated daily and people tend to view pages within minutes of each other, so we need a way of storing the information.

We create a PostgreSQL database to track all our state; we could use any other number of key-value stores but as it's a standard database it may be more familiar to Data Scientists.
We have a table `articles` for articles and their keywords which we update whenever a new article is published.

| article_id | article_name        | update_time | keywords                            |
|------------|---------------------|-------------|-------------------------------------|
| 1          | Point-in-time joins | 2022-04-01  | ["sql","python","machine-learning"] |
| 2          | Recipe NER          | 2022-03-01  | ["python","nlp"]                    |

And another table `activity` for tracking user activity which collects events from the frontend:

| user_id | event_time           | article_id |
|---------|----------------------|------------|
| 1       | 2022-05-11T11:00:00Z | 1          |
| 1       | 2022-05-11T11:05:00Z | 2          |
| 2       | 2022-05-11T11:00:00Z | 1          |

Then we can get get out the keywords with an SQL query, which we use to recommend articles:

```sql
SELECT history.*,
       articles.keywords || last_articles.keywords AS keywords
FROM
  (SELECT user_id,
          event_time,
          activity.article_id,
          lag(article_id) OVER (PARTITION BY user_id
                                ORDER BY event_time) AS last_article_id,
                               row_number() OVER (PARTITION BY user_id
                                                  ORDER BY event_time DESC) AS user_row_number
   FROM activity) AS history
LEFT JOIN articles ON history.article_id = articles.article_id
LEFT JOIN articles AS last_articles ON history.last_article_id = last_articles.article_id
WHERE user_row_number = 1;
```

| user_id | event_time           | article_id | last_article_id | user_row_number | keywords                            |
|---------|----------------------|------------|-----------------|-----------------|-------------------------------------|
| 1       | 2022-05-11T11:07:00Z | 3          | 2               | 1               | ["python","nlp","sql","python","machine-learning"] |
| 2       | 2022-05-11T11:00:00Z | 1          | null            | 1               | ["sql","python","machine-learning"] |


This works, but we find as the activity table grows the queries are getting slow and the recommendations are taking minutes to load.
Also when we are training a model on this database the recommendations get even slower again.
Instead we create a `current_activity` table that only carries the current snapshot of the data needed to generate the recommendations.
To keep it very fast we send and update the keywords on the fly:

| user_id | event_time           | article_id | last_article_id | current_keywords                    | last_keywords                       |
|---------|----------------------|------------|-----------------|-------------------------------------|-------------------------------------|
| 1       | 2022-05-11T11:07:00Z | 3          | 2               | ["python","nlp"]                    | ["sql","python","machine-learning"] |
| 2       | 2022-05-11T11:00:00Z | 1          | null            | ["sql","python","machine-learning"] | null                                |

But from the `current_activity` table we can't evaluate the historical recommendations.
Whenever a new page is viewed it UPDATEs the row for that user and we lose the history.
So we set up [Change Data Capture](https://datacater.io/blog/2021-09-02/postgresql-cdc-complete-guide.html) to track all the changes and store it in a separate analytics database.
We then have a `historical_activity` table that contains the state of the activity every time.

We think that a better keyword extraction strategy could produce better recommendations.
When training and evaluating on historical data we want to use our new keyword feature, not the one that was available at the time.
Otherwise we're not testing the model on the same data we will have at inference time, we have training-serving skew.
This means we can't use the keywords from the `historical_activity`, but have to lookup the new keywords using `article_id` and `last_article_id`.
So we add a new `keywords_v2` column and use a variation of the SQL query above to get the new features.
An alternative would be to add `keywords_v2` to the live feature store and wait for historical data to accumulate, but this is a very slow way to iterate on features.

However we actually rewrote the point-in-time joins article to use Julia instead of Python.
When recommending to people who viewed the old version we should use the keywords based on the text of that article; presumably they were more interested in Python.
To do this we need to do a *point-in-time join*; we join with the `article` table containing all versions and get the most recent version available at that time.
We do that by joining on `event_time >= update_time` and then pick the row with the most recent `update_time`.
It's a bit more complex because we need to do this for the previous version as well; here's an SQL query that may do it.

```sql
SELECT *
FROM
  (SELECT history.*,
          articles.keywords_v2 || last_articles.keywords_v2 AS keywords,
          row_number() OVER (PARTITION BY user_id,
                                          event_time,
                                          history.article_id,
                                          history.last_article_id
                             ORDER BY articles.update_time DESC, last_articles.update_time DESC) AS rn
   FROM
     (SELECT user_id,
             event_time,
             activity.article_id,
             lag(article_id) OVER (PARTITION BY user_id
                                   ORDER BY event_time) AS last_article_id,
                                  lag(event_time) OVER (PARTITION BY user_id
                                                        ORDER BY event_time) AS last_article_event_time,
                                                       row_number() OVER (PARTITION BY user_id
                                                                          ORDER BY event_time DESC) AS user_row_number
      FROM activity) AS history
   LEFT JOIN articles ON history.article_id = articles.article_id
   AND history.event_time >= articles.update_time
   LEFT JOIN articles AS last_articles ON history.last_article_id = last_articles.article_id
   AND history.last_article_event_time >= last_articles.update_time
   WHERE user_row_number = 1 ) all_versions
WHERE rn = 1;
```

This substantially improves our offline metrics.
We take the latest version of `keyword_v2` for all users, insert it into our online feature store, and run an A/B experiment to see whether it is really better.

In summary we want to get the data as it would have been if the new features were used at that time.
For the *raw data*, such as the interaction events and article text, we want to get the data as it was at interaction time.
If this involves combining data between historical tables we need to do a *point-in-time join* to get the correct data.
For the *features* that are built on the raw data, such as keywords, we want to calculate them as they are now.
This becomes very clear when you think about brand new features, such as using a the location of the user or a dense embedding of the articles instead of keywords.
This lets us do offline evaluation of new features in a faithful way.

In practice it may make sense to use various approximations to make this process computationally tractable.
We can truncate the update times (e.g. to the hour or the day) which can create fewer potential join points if there are multiple updates.
We can also limit the updates to some window if articles become stale and expire.
Any offline evaluation itself requries approximation, we don't know how a user would really react to a different set of recommendations, so these are generally acceptable.

# How do feature stores do it?

This hopefully illustrates the kinds of problems that feature stores try to solve.
Here are a few examples of point-in-time joins in different feature stores:

* [Feast](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) looks like it snapshots features at regular intervals and only retains them over some window (in the diagram feature_hourly_stats only has aggregate stats on the hour they set a `ttl=timedelta(hours=2)`)
* [Hopsworks](https://examples.hopsworks.ai/master/featurestore/hsfs/time_travel/point_in_time_join_python/) looks like it generates SparkSQL to do the point-in-time join (join on a.key = b.key and a.time <= b.time), I think over a limited time window
* [Sagemaker](https://aws.amazon.com/blogs/machine-learning/build-accurate-ml-training-datasets-using-point-in-time-queries-with-amazon-sagemaker-feature-store-and-apache-spark/) makes you roll your own point-in-time join and windowing manually
* [Featuretools](https://featuretools.alteryx.com/en/stable/getting_started/handling_time.html) (not a feature store) has an informative discussion of the different approaches (windowing, approximating, windowing)

Doing this manually, especially from SQL, can be very error prone (I wouldn't be surprised if the query above is wrong).
In general the features may need to represent different kinds of joins and aggregations (for example maybe we want the last 3 referral sources for a user).
If you need to do this kind of point-in-time join you should try to automate it, and make the process of generating features for model training/evaluation and at inference time as similar as possible.
This can be tricky because the performance characteristics are quite different, at training time you want to generate features on a very large amount of data with high throughput, but at inference time you want to generate features on a single datum with low latency.
Having a strategy that lets you do both is important for making these kinds of features practicable.