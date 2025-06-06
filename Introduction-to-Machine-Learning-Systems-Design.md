# **Introduction to Machine Learning Systems Design**

ML systems design takes a system approach to `MLOps`, which means
that we’ll consider an ML system holistically to ensure that all the components—the
`business requirements`, the `data stack`, `infrastructure`, `deployment`, `monitoring`, etc.—
and their stakeholders can work together to satisfy the specified objectives and
requirements. Before we develop an ML system, we must understand why this system is needed. If this system is built for a
business, it must be driven by `business objectives`, which will need to be translated
into ML objectives to guide the development of ML models.


## **Business and ML Objectives**
When working on an ML project, data scientists tend to care about the ML objectives: the metrics
they can measure about the performance of their ML models such as `accuracy`, `F1-score`, `inference latency`, etc. They get excited about improving their model’s accuracy from `94%` to `94.2%` and might spend a ton of resources—`data`, `compute`, and `engineering` time—to achieve that.

But the truth is: most companies don’t care about the `fancy ML metrics`. They don’t care about increasing a model’s accuracy from `94%` to `94.2%` unless it moves some business metrics. A pattern I see in many short-lived ML projects is that the data
scientists become too focused on hacking ML metrics without paying attention to business metrics. Their managers, however, only care about business metrics and, after failing to see how an ML project can help push their business metrics, kill the
projects prematurely (and possibly let go of the data science team involved).

So what metrics do companies care about? While most companies want to convince you otherwise, the sole purpose of businesses, according to the Nobel-winning economist `Milton Friedman`, is to maximize profits for shareholders. The ultimate goal of any project within a business is, therefore, to increase profits, either directly or indirectly: directly such as increasing sales (conversion rates) and cutting costs; indirectly such as higher customer satisfaction and increasing time spent on a website.

For an ML project to succeed within a business organization, it’s crucial to tie the
performance of an ML system to the overall business performance. What business
performance metrics is the new ML system supposed to influence, e.g., `the amount of ads revenue`, `the number of monthly active users?`

- Imagine that you work for an ecommerce site that cares about purchase-through rate
and you want to move your recommender system from `batch prediction` to `online
prediction`. You might reason that `online prediction` will enable recommendations
more relevant to users right now, which can lead to a `higher purchase-through rate`.
You can even do an experiment to show that online prediction can improve your
recommender system’s predictive accuracy by `X%` and, historically on your site, each
percent increase in the recommender system’s predictive accuracy led to a certain
increase in purchase-through rate.


- One of the reasons why predicting `ad click-through rates` and `fraud detection` are
among the most popular use cases for ML today is that it’s easy to map ML models’
performance to business metrics: every increase in click-through rate results in actual
ad revenue, and every fraudulent transaction stopped results in actual money saved.



- Many companies create their own metrics to map business metrics to ML metrics.
For example, `Netflix` measures the performance of their recommender system using
take-rate: the number of quality plays divided by the number of recommendations
a user sees. The higher the take-rate, the better the recommender system. Netflix
also put a recommender system’s take-rate in the context of their other business
metrics like `total streaming hours` and `subscription cancellation rate`. They found that
a higher take-rate also results in higher total streaming hours and lower subscription
cancellation rates.



- The effect of an ML project on business objectives can be hard to reason about. For
example, an ML model that gives customers more personalized solutions can make
them happier, which makes them spend more money on your services. The same ML
model can also solve their problems faster, which makes them spend less money on
your services.


- To gain a definite answer on the question of how ML metrics influence business
metrics, experiments are often needed. Many companies do that with experiments
like `A/B testing` and choose the model that leads to better business metrics, regardless
of whether this model has better ML metrics.



When evaluating ML solutions through the business lens, it’s important to be realistic about the expected returns. Due to all the hype surrounding ML, generated both by the media and by practitioners with a vested interest in ML adoption, some
companies might have the notion that ML can magically transform their businesses overnight.



## **Requirements for ML Systems**
The specified requirements for an ML system
vary from use case to use case. However, most systems should have these four characteristics: `reliability`, `scalability`, `maintainability`, and `adaptability`.


1. **Reliability**

The system should continue to perform the correct function at the desired level of
performance even in the face of adversity (hardware or software faults, and even
human error).

“Correctness” might be difficult to determine for ML systems. For example, your
system might call the predict function—e.g., `model.predict()` correctly, but the
predictions are wrong. How do we know if a prediction is wrong if we don’t have
ground truth labels to compare it with?

With traditional software systems, you often get a warning, such as a system crash
or runtime error or 404. However, ML systems can fail silently. End users don’t even
know that the system has failed and might have kept on using it as if it were working.



2. **Scalability**

There are multiple ways an ML system can grow. It can grow in complexity. Last year
you used a `logistic regression model` that fit into an Amazon Web Services (AWS)
free tier instance with `1 GB` of RAM, but this year, you switched to a 100-million-
parameter neural network that requires `16 GB` of RAM to generate predictions.

Your ML system can grow in traffic volume. When you started deploying an ML
system, you only served `10,000 prediction` requests daily. However, as your company’s
user base grows, the number of prediction requests your ML system serves daily
fluctuates between 1 million and 10 million.


An ML system might grow in ML model count. Initially, you might have only one
model for one use case, such as detecting the trending hashtags on a social network
site like `Twitter`. However, over time, you want to add more features to this use
case, so you’ll add one more to filter out `NSFW` (not safe for work) content and
another model to filter out tweets generated by bots. This growth pattern is especially common in ML systems that target enterprise use cases. Initially, a startup might serve only one enterprise customer, which means this startup only has one model. However, as this startup gains more customers, they might have one model for each
customer. A startup I worked with had `8,000` models in production for their `8,000` enterprise customers.


Whichever way your system grows, there should be reasonable ways of dealing with that growth. When talking about scalability most people think of resource scaling `(horizontal scaling)`, which consists of up-scaling (expanding the resources to handle growth) and downscaling (reducing the resources when not needed).


For example, at peak, your system might require `100 GPUs` (graphics processing units). However, most of the time, it needs only `10 GPUs`. Keeping `100 GPUs` up all the time can be costly, so your system should be able to scale down to `10 GPUs`.



3. **Maintainability**

There are many people who will work on an ML system. They are `ML engineers`, `DevOps engineers`, and subject matter experts (SMEs). They might come from very different backgrounds, with very different programming languages and tools, and
might own different parts of the process.

It’s important to structure your workloads and set up your infrastructure in such a way that different contributors can work using tools that they are comfortable with, instead of one group of contributors forcing their tools onto other groups.
Code should be documented. Code, data, and artifacts should be versioned. Models should be sufficiently reproducible so that even when the original authors are not around, other contributors can have sufficient contexts to build on their work. When
a problem occurs, different contributors should be able to work together to identify the problem and implement a solution without finger-pointing.



4. **Adaptability**

To adapt to shifting data distributions and business requirements, the system should
have some capacity for both discovering aspects for performance improvement and allowing updates without service interruption.
Because ML systems are part code, part data, and data can change quickly, ML systems need to be able to evolve quickly. This is tightly linked to maintainability.













































