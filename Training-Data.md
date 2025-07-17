# **Training Data**

`Training data still forms the foundation of modern ML algorithms. No matter how clever your algorithms might be, if your training data is bad, your algorithms won’t be able to perform well. It’s worth it to invest time and effort to curate and create training data that will enable your algorithms to learn something meaningful`.

`Most ML algorithms in use today are supervised ML algorithms, so obtaining labels is an integral part of creating training data. Many tasks, such as delivery time estimation or recommender systems, have natural labels. Natural labels are usually delayed, and the time it takes from when a prediction is served until when the feedback on it is provided is the feedback loop length. Tasks with natural labels are fairly common in the industry, which might mean that companies prefer to start on tasks that have natural labels over tasks without natural labels`.

`For tasks that don’t have natural labels, companies tend to rely on human annotators to annotate their data. However, hand labeling comes with many drawbacks. For example, hand labels can be expensive and slow. To combat the lack of hand labels, we consider alternatives including weak supervision, semi-supervision, transfer learning, and active learning`.

`ML algorithms work well in situations when the data distribution is more balanced, and not so well when the classes are heavily imbalanced. Unfortunately, problems with class imbalance are the norm in the real world.`


Building a state-of-the-art model is interesting. Spending days wrangling with a massive amount of malformatted data that doesn’t even fit into your machine’s memory is frustrating. Data is messy, complex, unpredictable, and potentially treacherous. If not handled properly, it can easily sink your entire ML operation. But this is precisely the reason why data scientists and ML engineers should learn how to handle data well, saving us time and headache down the road. Like other steps in building ML systems, creating training data is an iterative process. As your model evolves through a project lifecycle, your training data will likely also evolve.



## Sampling
Sampling is an integral part of the ML workflow that is, unfortunately, often overlooked in typical ML coursework. Sampling happens in many steps of an ML project lifecycle, such as sampling from all possible real-world data to create training data; sampling from a given dataset to create splits for `training`, `validation`, and `testing`; or sampling from all possible events that happen within your ML system for monitoring purposes. 

There are two families of sampling: nonprobability sampling and random sampling.

1. **Nonprobability Sampling**
Nonprobability sampling is when the selection of data isn’t based on any probability criteria. Here are some of the criteria for nonprobability sampling:

- Convenience sampling: Samples of data are selected based on their availability. This sampling method is popular because, well, it’s convenient.

- Snowball sampling: Future samples are selected based on existing samples. For example, to scrape legitimate Twitter accounts without having access to Twitter databases, you start with a small number of accounts, then you scrape all the accounts they follow, and so on.

- Judgment sampling: Experts decide what samples to include.

- Quota sampling: You select samples based on quotas for certain slices of data without any randomization. For example, when doing a survey, you might want `100 responses` from each of the age groups: under `30 years` old, between `30` and `60` years old, and above `60` years old, regardless of the actual age distribution.


2. **Simple Random Sampling**
In the simplest form of random sampling, you give all samples in the population equal probabilities of being selected. For example, you randomly select `10%` of the population, giving all members of this population an equal `10%` chance of being selected.

- The advantage of this method is that it’s easy to implement. 

- The drawback is that rare categories of data might not appear in your selection. Consider the case where a class appears only in `0.01%` of your data population. If you randomly select `1%` of your data, samples of this rare class will unlikely be selected. Models trained on this selection might think that this rare class doesn’t exist.


3. **Stratified Sampling**
To avoid the drawback of simple random sampling, you can first divide your population into the groups that you care about and sample from each group separately. For example, to sample `1%` of data that has two classes, `A` and `B`, you can sample `1%` of `class A` and `1%` of `class B`. This way, no matter how rare class A or B is, you’ll ensure that samples from it will be included in the selection. Each group is called a `stratum`, and this method is called `stratified sampling`.


4. **Weighted Sampling**
In weighted sampling, each sample is given a weight, which determines the probability of it being selected. For example, if you have three samples, `A`, `B`, and `C`, and want them to be selected with the probabilities of `50%`, `30%`, and `20%` respectively, you can give them the weights `0.5`, `0.3`, and `0.2`.

This method allows you to leverage domain expertise. For example, if you know that a certain subpopulation of data, such as more recent data, is more valuable to your model and want it to have a higher chance of being selected, you can give it a higher weight.


This also helps with the case when the data you have comes from a different distribution compared to the true data. For example, if in your data, `red samples` account for `25%` and `blue samples` account for `75%`, but you know that in the real world, red and blue have equal probability to happen, you can give red samples weights three times higher than blue samples.


5. **Reservoir Sampling**
Reservoir sampling is a fascinating algorithm that is especially useful when you have to deal with streaming data, which is usually what you have in production.

Imagine you have an incoming stream of tweets and you want to sample a certain number, `k`, of tweets to do analysis or train a model on. You don’t know how many tweets there are, but you know you can’t fit them all in memory, which means you don’t know in advance the probability at which a tweet should be selected. You want to ensure that: 

- Every tweet has an equal probability of being selected.

- You can stop the algorithm at any time and the tweets are sampled with the correct probability.


One solution for this problem is reservoir sampling. The algorithm involves a reservoir, which can be an array, and consists of three steps:

1. Put the first `k` elements into the reservoir.

2. For each incoming `nth` element, generate a random number `i` such that `1 ≤ i ≤ n`.

3. If `1 ≤ i ≤ k`: replace the `ith` element in the reservoir with the `nth` element. Else, do nothing.



6. **Importance Sampling**
Importance sampling is one of the most important sampling methods, not just in ML. It allows us to sample from a distribution when we only have access to another distribution.





## Labeling
Despite the promise of unsupervised ML, most ML models in production today are supervised, which means that they need labeled data to learn from. The performance of an ML model still depends heavily on the quality and quantity of the labeled data it’s trained on.


### **Hand Labels**
Anyone who has ever had to work with data in production has probably felt this at a visceral level: acquiring hand labels for your data is difficult for many, many reasons. First, hand-labeling data can be expensive, especially if subject matter expertise is required. To classify whether a comment is spam, you might be able to find `20` annotators on a crowdsourcing platform and train them in 15 minutes to label your
data. However, if you want to label `chest X-rays`, you’d need to find board-certified radiologists, whose time is limited and expensive.

Second, hand labeling poses a threat to data privacy. Hand labeling means that someone has to look at your data, which isn’t always possible if your data has strict privacy requirements. For example, you can’t just ship your patients’ medical records or your company’s confidential financial information to a third-party service for labeling. In many cases, your data might not even be allowed to leave your organization, and you
might have to hire or contract annotators to label your data on premises.

Slow labeling leads to slow iteration speed and makes your model less adaptive to changing environments and requirements. If the task changes or data changes, you’ll have to wait for your data to be relabeled before updating your model.


1. **Label multiplicity**
Often, to obtain enough labeled data, companies have to use data from multiple sources and rely on multiple annotators who have different levels of expertise. These different data sources and annotators also have different levels of accuracy. This leads to the problem of label ambiguity or label multiplicity: what to do when there are multiple conflicting labels for a data instance.

Disagreements among annotators are extremely common. The higher the level of domain expertise required, the higher the potential for annotating disagreement. If one human expert thinks the label should be `A` while another believes it should be `B`, how do we resolve this conflict to obtain one single ground truth? If human experts can’t agree on a label, what does human-level performance even mean?


2. **Data lineage**
Indiscriminately using data from multiple sources, generated with different annotators, without examining their quality can cause your model to fail mysteriously. Consider a case when you’ve trained a moderately good model with 100K data samples. Your ML engineers are confident that more data will improve the model performance, so you spend a lot of money to hire annotators to label another million data samples.


However, the model performance actually decreases after being trained on the new data. The reason is that the new million samples were crowdsourced to annotators who labeled data with much less accuracy than the original data. It can be especially difficult to remedy this if you’ve already mixed your data and can’t differentiate new data from old data.


It’s good practice to keep track of the origin of each of your data samples as well as its labels, a technique known as `data lineage.` Data lineage helps you both flag potential biases in your data and debug your models.




### **Natural Labels**
Hand-labeling isn’t the only source for labels. You might be lucky enough to work on tasks with natural ground truth labels. Tasks with natural labels are tasks where the model’s predictions can be automatically evaluated or partially evaluated by the system. An example is the model that estimates time of arrival for a certain route on `Google Maps`. If you take that route, by the end of your trip, Google Maps knows how
long the trip actually took, and thus can evaluate the accuracy of the predicted time of arrival. Another example is `stock price` prediction. If your model predicts a stock’s price in the next two minutes, then after two minutes, you can compare the predicted price with the actual price.


The canonical example of tasks with natural labels is `recommender systems`. The goal of a recommender system is to recommend to users items relevant to them. Whether a user clicks on the recommended item or not can be seen as the feedback for that recommendation. A recommendation that gets clicked on can be presumed to be good (i.e., the label is `POSITIVE`) and a recommendation that doesn’t get clicked on after a period of time, say 10 minutes, can be presumed to be bad (i.e., the label is `NEGATIVE`).


**Feedback loop length**
For tasks with natural ground truth labels, the time it takes from when a prediction is served until when the feedback on it is provided is the feedback loop length. Tasks with short feedback loops are tasks where labels are generally available within minutes. Many recommender systems have short feedback loops.



- `Different Types of User Feedback`
`If you want to extract labels from user feedback, it’s important to note that there are different types of user feedback. They can occur at different stages during a user journey on your app and differ by volume, strength of signal, and feedback loop length.`


`For example, consider an ecommerce application similar to what Amazon has. Types of feedback a user on this application can provide might include clicking on a product recommendation, adding a product to cart, buying a product, rating, leaving a review, and returning a previously bought product.`


`Clicking on a product happens much faster and more frequently (and therefore incurs a higher volume) than purchasing a product. However, buying a product is a much stronger signal on whether a user likes that product compared to just clicking on it.`


`When building a product recommender system, many companies focus on optimizing for clicks, which give them a higher volume of feedback to evaluate their models. However, some companies focus on purchases, which gives them a stronger signal that is also more correlated to their business metrics (e.g., revenue from product sales). Both approaches are valid. There’s no definite answer to what type of feedback you should optimize for your use case, and it merits serious discussions between all stakeholders involved.`


Choosing the right window length requires thorough consideration, as it involves the speed and accuracy trade-off. A short window length means that you can capture labels faster, which allows you to use these labels to detect issues with your model and address those issues as soon as possible. However, a short window length also means that you might prematurely label a recommendation as bad before it’s clicked on.


For tasks with long feedback loops, natural labels might not arrive for weeks or even months. `Fraud detection` is an example of a task with long feedback loops. For a certain period of time after a transaction, users can dispute whether that transaction is fraudulent or not. For example, when a customer read their credit card statement and saw a transaction they didn’t recognize, they might dispute it with their bank, giving the bank the feedback to label that transaction as fraudulent. A typical dispute window is one to three months. After the dispute window has passed, if there’s no dispute from the user, you might presume the transaction to be legitimate.


Labels with long feedback loops are helpful for reporting a model’s performance on quarterly or yearly business reports. However, they are not very helpful if you want to detect issues with your models as soon as possible. If there’s a problem with your fraud detection model and it takes you months to catch, by the time the problem is fixed, all the fraudulent transactions your faulty model let through might have caused
a small business to go bankrupt.



### **Handling the Lack of Labels**
Because of the challenges in acquiring sufficient high-quality labels, many techniques have been developed to address the problems that result. 

Four techniques for handling the lack of hand-labeled data
- weak supervision, 
- semi-supervision, 
- transfer learning, and 
- active learning.


**Weak Supervision**
If hand labeling is so problematic, what if we don’t use hand labels altogether? One approach that has gained popularity is weak supervision. One of the most popular open source tools for weak supervision is `Snorkel`, developed at the Stanford AI Lab. The insight behind weak supervision is that people rely on `heuristics`, which can be developed with subject matter expertise, to label data.

In theory, you don’t need any hand labels for weak supervision. However, to get a sense of how accurate your `LFs (labeling functions)` are, a small number of hand labels is recommended. These hand labels can help you discover patterns in your data to write better LFs.

Weak supervision can be especially useful when your data has strict privacy requirements. You only need to see a small, cleared subset of data to write LFs, which can be applied to the rest of your data without anyone looking at it.



**Semi-supervision**
If weak supervision leverages heuristics to obtain noisy labels, semi-supervision leverages structural assumptions to generate new labels based on a small set of initial labels. Unlike weak supervision, semi-supervision requires an initial set of labels.

A classic semi-supervision method is self-training. You start by training a model on your existing set of labeled data and use this model to make predictions for unlabeled samples. 

Semi-supervision is the most useful when the number of training labels is limited. One thing to consider when doing semi-supervision with limited data is how much of this limited data should be used to evaluate multiple candidate models and select the best one. 


**Transfer Learning**
Transfer learning refers to the family of methods where a model developed for a task is reused as the starting point for a model on a second task. First, the base model is trained for a base task. The base task is usually a task that has cheap and abundant training data. Language modeling is a great candidate because it doesn’t require labeled data. Language models can be trained on any body of text—books, `Wikipedia articles`, `chat histories` and the task is: given a sequence of tokens, predict the next token. When given the sequence “I bought NVIDIA shares because I believe in the importance of,” a language model might output “hardware” or “GPU” as the next token.


Transfer learning is especially appealing for tasks that don’t have a lot of labeled data. Even for tasks that have a lot of labeled data, using a pretrained model as the starting point can often boost the performance significantly compared to training from scratch.

Many have hypothesized that in the future only a handful of companies will be able to afford to train large
pretrained models. The rest of the industry will use these pretrained models directly or fine-tune them for their specific needs.



**Active Learning**
Active learning is a method for improving the efficiency of data labels. The hope here is that ML models can achieve greater accuracy with fewer training labels if they can choose which data samples to learn from.



## Class Imbalance
Class imbalance typically refers to a problem in classification tasks where there is a substantial difference in the number of samples in each class of the training data. For example, in a training dataset for the task of detecting lung cancer from `X-ray images`, `99.99%` of the `X-rays` might be of `normal lungs`, and only `0.01%` might contain `cancerous cells`.

Class imbalance can also happen with regression tasks where the labels are continuous.


### **Challenges of Class Imbalance
ML, especially deep learning, works well in situations when the data distribution is more balanced, and usually not so well when the classes are heavily imbalanced.

- The first reason is that class imbalance often means there’s insufficient signal for your model to learn to detect the minority classes.

- The second reason is that class imbalance makes it easier for your model to get stuck in a nonoptimal solution by exploiting a simple heuristic instead of learning anything useful about the underlying pattern of the data.

- The third reason is that class imbalance leads to asymmetric costs of error—the cost of a wrong prediction on a sample of the rare class might be much higher than a wrong prediction on a sample of the majority class.

For example, misclassification on an `X-ray` with `cancerous cells` is much more dangerous than misclassification on an `X-ray` of a `normal lung`. If your loss function isn’t configured to address this asymmetry, your model will treat all samples the same way. As a result, you might obtain a model that performs equally well on both majority and minority classes, while you much prefer a model that performs less well on the majority class but much better on the minority one.



### **Handling Class Imbalance**
Class imbalance affects tasks differently based on the level of imbalance. Some tasks are more sensitive to class imbalance than others.

Three approaches to handling class imbalance:
- Choosing the right metric for your problem
- data-level methods, which means changing the data distribution to make it less imbalanced
- Algorithm-level methods, which means changing your learning method to make it more robust to class imbalance.



1. **Using the right evaluation metrics**
The most important thing to do when facing a task with class imbalance is to choose the appropriate evaluation metrics. Wrong metrics will give you the wrong ideas of how your models are doing and, subsequently, won’t be able to help you develop or choose models good enough for your task.


The overall `accuracy` and `error rate` are the most frequently used metrics to report the performance of ML models. However, these are insufficient metrics for tasks with class imbalance because they treat all classes equally, which means the performance of your model on the majority class will dominate these metrics. This is especially bad when the majority class isn’t what you care about.


`F1`, `precision`, and `recall` are metrics that measure your model’s performance with respect to the positive class in binary classification problems, as they rely on true positive—an outcome where the model correctly predicts the positive class.


- `Precision = True Positive / (True Positive + False Positive)`
- `Recall = True Positive / (True Positive + False Negative)`
- `F1 = 2 × Precision × Recall / (Precision + Recall)`
 
Like F1 and recall, the `ROC` curve focuses only on the positive class and doesn’t show how well your model does on the negative class. Davis and Goadrich suggested that we should plot precision against recall instead, in what they termed the `Precision-Recall Curve`. They argued that this curve gives a more informative picture of an algorithm’s performance on tasks with heavy class imbalance.


2. **Data-level methods: Resampling**
Data-level methods modify the distribution of the training data to reduce the level of imbalance to make it easier for the model to learn. A common family of techniques is resampling. Resampling includes oversampling, adding more instances from the minority classes, and undersampling, removing instances of the majority classes.

The simplest way to undersample is to randomly remove instances from the majority class, whereas the simplest way to oversample is to randomly make copies of the minority class until you have a ratio that you’re happy with. 


A popular method of oversampling low-dimensional data is `SMOTE` (synthetic minority oversampling technique).
It synthesizes novel samples of the minority class through sampling convex combinations of existing data points within the minority class.


Undersampling runs the risk of losing important data from removing data. Oversampling runs the risk of overfitting on training data, especially if the added copies of the minority class are replicas of existing data. Many sophisticated sampling techniques have been developed to mitigate these risks.


One such technique is `two-phase learning`. You first train your model on the resampled data. This resampled data can be achieved by randomly undersampling large classes until each class has only N instances. You then fine-tune your model on the original data.


Another technique is dynamic sampling: oversample the low-performing classes and undersample the high-performing classes during the training process. Introduced by `Pouyanfar et al`., the method aims to show the model less of what it has already learned and more of what it has not.




3. **Algorithm-level methods**
If data-level methods mitigate the challenge of class imbalance by altering the distribution of your training data, algorithm-level methods keep the training data distribution intact but alter the algorithm to make it more robust to class imbalance.


Because the `loss function` (or the cost function) guides the learning process, many algorithm-level methods involve `adjustment` to the loss function. The key idea is that if there are two instances, `x1` and `x2`, and the loss resulting from making the wrong prediction on `x1` is higher than `x2`, the model will prioritize making the correct prediction on `x1` over making the correct prediction on `x2`. By giving the training instances we care about higher weight, we can make the model focus more on learning these instances.




## Data Augmentation
Data augmentation is a family of techniques that are used to increase the amount of training data. Traditionally, these techniques are used for tasks that have limited training data, such as in medical imaging. However, data—augmented data can make our models more robust to noise and even adversarial attacks.


Data augmentation has become a standard step in many computer vision tasks and is finding its way into natural language processing (NLP) tasks. The techniques depend heavily on the data format, as image manipulation is different from text manipulation. 


**Three main types of Data Augmentation**
1. **simple label-preserving transformations**
In computer vision, the simplest data augmentation technique is to randomly modify an image while preserving its label. You can modify the image by `cropping`, `flipping`, `rotating`, `inverting` (horizontally or vertically), `erasing part of the image`, and more. This makes sense because a rotated image of a dog is still a dog. Common ML frameworks like `PyTorch`, `TensorFlow`, and `Keras` all have support for image augmentation.


In NLP, you can randomly replace a word with a similar word, assuming that this replacement wouldn’t change the meaning or the sentiment of the sentence.

This type of data augmentation is a quick way to double or triple your training data.



2. **Perturbation**
Perturbation is also a label-preserving operation, but because sometimes it’s used to trick models into making wrong predictions.

Neural networks, in general, are sensitive to noise. In the case of computer vision, this means that adding a small amount of noise to an image can cause a neural network to misclassify it. 

Using deceptive data to trick a neural network into making wrong predictions is called `adversarial attacks`. Adding noise to samples is a common technique to create `adversarial samples`. The success of adversarial attacks is especially exaggerated as the resolution of images increases.



3. **Data Synthesis**
Since collecting data is expensive and slow, with many potential privacy concerns, it’d be a dream if we could sidestep it altogether and train our models with synthesized data. Even though we’re still far from being able to synthesize all training data, it’s possible to synthesize some training data to boost a model’s performance.


In NLP, templates can be a cheap way to bootstrap your model. (In NLP, a `template` is a pre-defined structure, often a string of words or a more complex format, that serves as a guide for generating or processing text. It provides a framework with placeholders that can be filled with specific information or content to create meaningful text outputs or to structure and analyze input text. )


## Summary
Training data still forms the foundation of modern ML algorithms. No matter how clever your algorithms might be, if your training data is bad, your algorithms won’t be able to perform well. It’s worth it to invest time and effort to curate and create training data that will enable your algorithms to learn something meaningful.






























































