# The Human Side of Machine Learning

ML systems aren't just technical. They involve business decision makers, users, and, of course, developers of the systems. 


## User Experience

ML systems behave differently from traditional systems. 

First, ML systems are probabilistic instead of deterministic. Usually, if you run the same software on the same input twice at different times, you can expect the same result. However, if you run the same ML system twice at different times on the exact same input, you might get different results.

Second, due to this probabilistic nature, ML systems’ predictions are mostly correct, and the hard part is we usually don’t know for what inputs the system will be correct!

Third, ML systems can also be large and might take an unexpectedly long time to produce a prediction.


These differences mean that ML systems can affect user experience differently, especially for users that have so far been used to traditional software. Due to the relatively new usage of ML in the real world, how ML systems affect user experience is still not well studied. 



**Three challenges that ML systems pose to good user experience and how to address them.**

1. **Ensuring User Experience Consistency**

ML predictions are probabilistic and inconsistent, which means that predictions generated for one user today might be different from what will be generated for the same user the next day, depending on the context of the predictions. For tasks that want to leverage ML to improve users’ experience, the inconsistency in ML predictions can be a hindrance.


2. **Combatting “Mostly Correct” Predictions**

This approach is very common and is sometimes called “human-in-the-loop” AI, as it involves humans to pick the best predictions or to improve on the machine-generated predictions.


3. **Smooth Failing**

This is related to the speed–accuracy trade-off: a model might have worse performance than another model but can do inference much faster. This less-optimal but fast model might give users worse predictions but might still be preferred in situations where latency is crucial. Many companies have to choose one model over another, but with a backup system, you can do both.




## Team Structure

An ML project involves not only data scientists and ML engineers, but also other types of engineers such as DevOps engineers and platform engineers as well as nondeveloper stakeholders like subject matter experts (SMEs). Given a diverse set of stakeholders, the question is what is the optimal structure when organizing ML teams. 



1. **Cross-functional Teams Collaboration**

SMEs (doctors, lawyers, bankers, farmers, stylists, etc.) are often overlooked in the design of ML systems, but many ML systems wouldn’t work without subject matter expertise. They’re not only users but also developers of ML systems.

An ML system would benefit a lot to have SMEs involved in the rest of the lifecycle, such as problem formulation, feature engineering, error
analysis, model evaluation, reranking predictions, and user interface: how to best present results to users and/or to other parts of the system.


There are many challenges that arise from having multiple different profiles working on a project. For example, how do you explain ML algorithms’ limitations and capacities to SMEs who might not have engineering or statistical backgrounds? To build an ML system, we want everything to be versioned, but how do you translate domain expertise (e.g., if there’s a small dot in this region between `X` and `Y` then it might be a sign of cancer) into code and version that?


It’s important to involve SMEs early on in the project planning phase and empower them to make contributions without having to burden engineers to give them access.


Most of the no-code ML solutions for SMEs are currently at the labeling, quality assurance, and feedback stages, but more platforms are being developed to aid in other critical junctions such as dataset creation and views for investigating issues that require SME input.




2. **End-to-End Data Scientists**

To do MLOps, we need not only `ML expertise` but also `Ops (operational) expertise`, especially around `deployment`, `containerization`, `job orchestration`, and `workflow management`.

To be able to bring all these areas of expertise into an ML project, companies tend to follow one of the two following approaches: have a separate team to manage all the `Ops` aspects or include `data scientists` on the team and have them own the entire process.



**Approach 1: Have a separate team to manage production**

In this approach, the data science/ML team develops models in the dev environment. Then a separate team, usually the Ops/platform/ML engineering team, productionizes the models in prod. This approach makes hiring easier as it’s easier to hire people with one set of skills instead of people with multiple sets of skills. It might also make life easier for each person involved, as they only have to focus on one concern (e.g., developing models or deploying models). However, this approach has many drawbacks:


- Communication and coordination overhead
- Debugging challenges
- Finger-pointing
- Narrow context



**Approach 2: Data scientists own the entire process**

In this approach, the data science team also has to worry about productionizing models. Data scientists become grumpy unicorns, expected to know everything about the process, and they might end up writing more boilerplate code than data science.

The success of a full-stack data scientist relies on the tools they have. They need tools that “abstract the data scientists from the complexities of `containerization`, `distributed processing`, `automatic failover`, and other advanced computer science concepts.



## Responsible AI

Responsible AI is the practice of designing, developing, and deploying AI systems with good intention and sufficient awareness to empower users, to engender trust, and to ensure fair and positive impact to society. It consists of areas like fairness, privacy, transparency, and accountability.

These terms are no longer just philosophical musings, but serious considerations for both policy makers and everyday practitioners. Given ML is being deployed into almost every aspect of our lives, failing to make our ML systems fair and ethical can lead to catastrophic consequences.

As developers of ML systems, you have the responsibility not only to think about how your systems will impact users and society at large, but also to help all stakeholders better realize their responsibilities toward the users by concretely implementing `ethics`, `safety`, and `inclusivity` into your ML systems.



## Irresponsible AI: Case Studies

Irresponsible AI systems can reproduce social inequalities, creating a cycle of biased data and unfair outcomes.


**Case study I: Automated grader’s biases**

Coarse-grained accuracy alone is nowhere close to being sufficient to evaluate a model’s performance, especially for a model whose performance can influence the future of so many students. A closer look into this algorithm reveals at least three major failures along the process of designing and developing this automated grading system:


• `Failure to set the right objective`
• `Failure to perform fine-grained evaluation to discover potential biases`
• `Failure to make the model transparent`



**Case study II: The danger of “anonymized” data**

Since the development of ML systems relies heavily on the quality of data, it’s important for user data to be collected. The research community needs access to high-quality datasets to develop new techniques. Practitioners and companies require access to data to discover new use cases and develop new AI-powered products.


However, collecting and sharing datasets might violate the privacy and security of the users whose data is part of these datasets. To protect users, there have been calls for anonymization of `personally identifiable information (PII)`. 


Collecting and sharing data is essential for the development of data-driven technologies like AI. Developers of applications that gather user data must understand that their users might not have the technical know-how and privacy awareness to choose the right privacy settings for themselves, and so developers must proactively work to make the right settings the default, even at the cost of gathering less data.




## A Framework for Responsible AI

As an ML practitioner, to audit model behavior and set out guidelines that best help you meet the needs of
your projects. This framework is not sufficient for every use case. There are certain applications where the use of AI might altogether be inappropriate or unethical (e.g., criminal sentencing decisions, predictive policing), regardless of which framework you follow.


1. **Discover sources for model biases**

Biases can creep in your system through the entire ML workflow. Your first step is to discover how these biases can creep in. The following are some examples of the sources of data, but keep in mind that this list is far from being exhaustive. One of the reasons why biases are so hard to combat is that biases can come from any step during a project lifecycle.


- `Training data:` Is the data used for developing your model representative of the data your model will handle in the real world? If not, your model might be biased against the groups of users with less data represented in the training data.


- `Labeling:` If you use human annotators to label your data, how do you measure the quality of these labels? How do you ensure that annotators follow standard guidelines instead of relying on subjective experience to label your data? The more annotators have to rely on their subjective experience, the more room for human biases.


- `Feature Engineering:` Does your model use any feature that contains sensitive information? Does your model cause a disparate impact on a subgroup of people? Disparate impact occurs “when a selection process has widely different outcomes for different groups, even as it appears to be neutral.”


This can happen when a model’s decision relies on information correlated with legally protected classes (e.g., ethnicity, gender, religious practice) even when this information isn’t used in training the model directly. 


- `Model's Objective:` Are you optimizing your model using an objective that enables fairness to all users? For example, are you prioritizing your model’s performance on all users, which skews your model toward the majority group of users?


- `Evaluation:` Are you performing adequate, fine-grained evaluation to understand your model’s performance on different groups of users? Fair, adequate evaluation depends on the existence of fair, adequate evaluation data.



2. **Understand the limitations of the data-driven approach**

ML is a `data-driven` approach to solving problems. However, it’s important to understand that data isn’t enough. Data concerns people in the real world, with socioeconomic and cultural aspects to consider. We need to gain a better understanding of the blind spots caused by too much reliance on data. This often means crossing over `disciplinary` and `functional` boundaries, both within and outside the organization, so that we can account for the lived experiences of those who will be impacted by the systems that we build.


As an example, to build an equitable automated grading system, it’s essential to work with domain experts to understand the demographic distribution of the student population and how socioeconomic factors get reflected in the historical performance data.



3. **Understand the trade-offs between different desiderata**

When building an ML system, there are different properties you might want this system to have. For example, you might want your system to have low inference latency, which could be obtained by model compression techniques like pruning.

You might also want your model to have high predictive accuracy, which could be achieved by adding more data. You might also want your model to be fair and transparent, which could require the model and the data used to develop this model to be made accessible for public scrutiny.


Often, ML literature makes the unrealistic assumption that optimizing for one property, like model accuracy, holds all others static. People might discuss techniques to improve a model’s fairness with the assumption that this model’s accuracy or latency will remain the same. However, in reality, improving one property can cause other properties to degrade. Here are two examples of these trade-offs:


- Privacy versus accuracy trade-off
- Compactness versus fairness trade-off


Similar trade-offs continue to be discovered. It’s important to be aware of these trade-offs so that we can make informed design decisions for our ML systems. If you are working with a system that is compressed or differentially private, allocating more resources to auditing model behavior is recommended to avoid unintended harm.


4. **Act early**

You might encounter this narrative often in ML systems. Companies might decide to
bypass ethical issues in ML models to save cost and time, only to discover risks in
the future when they end up costing a lot more. 

The earlier in the development cycle of an ML system that you can start thinking
about how this system will affect the life of users and what biases your system might
have, the cheaper it will be to address these biases.






5. **Create model cards**

Model cards are short documents accompanying trained ML models that provide information on how these models were trained and evaluated. Model cards also disclose the context in which models are intended to be used, as well as their limitations.


“The goal of model cards is to standardize ethical practice and reporting by allowing stakeholders to compare
candidate models for deployment across not only traditional evaluation metrics but also along the axes of ethical, inclusive, and fair considerations.”


`Model cards are documents that provide structured information about a machine learning model, promoting transparency and responsible AI development. They detail the model's intended use, training data, performance characteristics, and ethical considerations, enabling users to understand its capabilities and limitations.` 


**Key aspects of Model Cards:**

- `Transparency:` Model cards enhance transparency by providing a comprehensive overview of the model's development and intended use. 

- `Intended Use:` They clearly define the specific tasks the model is designed for and the target audience. 

- `Training Data:` Information about the data used to train the model, including its characteristics and potential biases, is included. 

- `Performance Metrics:` Model cards report the model's performance across various metrics, including breakdowns by demographic groups to identify potential biases. 

- `Ethical Considerations:` They address potential ethical implications and risks associated with the model's use. 


The following list has been adapted from content in the paper `“Model Cards for Model Reporting”` to show the information you might want to report for your models:

a. `Model details:` Basic information about the model.

    - Person or organization developing model
    - Model date
    - Model version
    - Model type
    - Information about training algorithms, parameters, fairness constraints or other applied approaches, and features
    - Paper or other resource for more information
    - Citation details
    - License
    - Where to send questions or comments


b. `Intended use:` Use cases that were envisioned during development.

    — Primary intended uses
    — Primary intended users
    — Out-of-scope use cases


c. `Factors:` Factors could include demographic or phenotypic groups, environmental conditions, technical attributes, or others.

    — Relevant factors
    — Evaluation factors


d. `Metrics:` Metrics should be chosen to reflect potential real-world impacts of the model.

    - Model performance measures
    - Decision thresholds
    - variation approaches


e. `Evaluation data:` Details on the dataset(s) used for the quantitative analyses in the card.

    - Datasets
    - Motivation
    - Preprocessing


f. `Training data:` May not be possible to provide in practice. When possible, this section should mirror Evaluation Data. If such detail is not possible, minimal allowable information should be provided here, such as details of the distribution over various factors in the training datasets.



g. `Quantitative analyses`

    - Unitary results
    — Intersectional results


h. `Ethical considerations`


i. `Caveats and recommendations`


Model cards are a step toward increasing transparency into the development of ML models. They are especially important in cases where people who use a model aren’t the same people who developed this model.




6. **Establish processes for mitigating biases**

Building responsible AI is a complex process, and the more ad hoc the process is, the more room there is for errors. It’s important for businesses to establish systematic processes for making their ML systems responsible.



7. **Stay up-to-date on responsible AI**

AI is a fast-moving field. New sources of biases in AI are constantly being discovered, and new challenges for responsible AI constantly emerge. Novel techniques to combat these biases and challenges are actively being developed. It’s important to stay up-to-date with the latest research in responsible AI.




## Summary

Despite the technical nature of ML solutions, designing ML systems can’t be confined in the technical domain. They are developed by humans, used by humans, and leave their marks in society.

Building an ML system often requires multiple skill sets, and an organization might wonder how to distribute these required skill sets: to involve different teams with different skill sets or to expect the same team (e.g., data scientists) to have all the skills. We explored the pros and cons of both approaches. The main cons of the first approach is overhead in communication. The main cons of the second approach is that it’s difficult to hire data scientists who can own the process of developing an ML system end-to-end. Even if they can, they might not be happy doing it. However, the second approach might be possible if these end-to-end data scientists are provided with sufficient tools and infrastructure.


We ended the chapter with what I believe to be the most important topic of this book: responsible AI. Responsible AI is no longer just an abstraction, but an essential practice in today’s ML industry that merits urgent actions. Incorporating ethics principles into your modeling and organizational practices will not only help you distinguish yourself as a professional and cutting-edge data scientist and ML engineer but also help your organization gain trust from your customers and users. It will also help your organization obtain a competitive edge in the market as more and more customers and users emphasize their need for responsible AI products and services.


















































































