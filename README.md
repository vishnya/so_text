# Introduction
To install all the dependencies required to execute the code, clone the repo
locally, `cd` to the root, and run 

```
pip install -e .
```

 In this README, I discuss my solution to the
 [problem statement](docs/problem_statement.pdf).
 The data comprises the text of questions on StackOverflow, with labels that
 indicate whether the question was accepted or closed. The set-up is that
 currently the acceptance/rejection of question posts is done
 manually, but we can offload some manual work through
 automation and predictive modeling. Here is the type of the data that is
 provided:
 
```
Body	Title	label
0	<p>Hi I'm new to <code>Ruby on Rails</code>. I...	RUBY: most common number for Users	1
1	<p>I know that StringBuffer class is synchroni...	What exactly does it mean when they say that S...	0
2	<p>I have a search engine on PHP that have ind...	Scan a webpage and get the video embed url only	1

```

 Additionally, columns that process the text were also provided.
 
 ## Summary
 
- In the course of the analysis, I created a predictive model (XGBoost) with a
 threshold set to `.5` to create a classifier. I also trained multiple Naive
  Bayes models, and a Keras Sequential neural network. 
- I chose the `roc_auc` as a performance metric.
   - The test set is held out from an artificially balanced set. Therefore
    the performance metrics only indicate how well the models perform on data
    from the artificially balanced distribution.
   - I chose the `roc_auc` because it is threshold-agnostic, and there is not
    enough information provided to determine a threshold otherwise.
- The repo code exists solely to capture some of the analysis below.
 It is in the style of a data science notebook, not software. I chose to not
  use a Jupyter notebook because it does not play as well with `git` as
   does markdown,
   and I
  knew that I would not be visualizing much data.
- The below analysis is data science exploration rather than exploitation.
    - The question is analysis focused, so the organization of the repo is like
     the report about playing
     in a sandbox. That is, I try various approaches to lay
   to get a sense of what works, to lay groundwork to choose one path to
    optimize on performance.

 ## Additional detail

 My approach was to build a predictive model that is a classifier
 once a threshold is applied. The limitations of my approach for this problem
 context are
 addressed in the analysis below. For one, without the cost of false
 positives and negatives, and an without understanding of which call to make in
  the
 trade-off between better classifying one category over another, one cannot
  systematically set a
  threshold. In classifying I therefore chose a 
 threshold of `.5`, more for a concrete *illustration* of how the classifier
 classifies points at *some* threshold.
 
 For the performance metric, I chose
 `roc_auc` because it shows performance across all thresholds on the test
 set. However, because the test set is held out from an artificially
 balanced set, it is itself artificially balanced, which means that **if we
 optimize around performance metrics, we are not optimizing
 around how the model would perform in the wild,** but rather the conditions of
 the test set. For example, it is not obvious to me that with the
 imbalanced nature of the test set, the rank ordering of models is
 preserved with respect to performance metrics. That is, just because one ML
  approach has a better or worse
 outcome in terms of performance metrics on the test set, it is unclear to me
 that it should be chosen over a different one.

 The problem statement emphasizes exploration and analysis over performance
 fine-tuning and deployment. Thus, given the caveats above, I experimented
 with approaches throughout the initial modeling process, employing various
 preprocessing, feature engineering, and algorithm techniques, which
 needed to be then represented in code in the repo. Because most of the below
 explores possible approaches, and the repo itself
 is intended to merely capture some of the code behind the analysis, the repo
 does not follow the usual standards of a Python library. Nevertheless, with
 more time I could  organize the code
 into a coherent whole that is focused on automating some experimentation to
 facilitate performance fine-tuning. Specifically, I would have a config file
 that populates the hardcoded hyper-parameters with different combinations,
 applies different algorithms specified in the config, and returns the optimal
 result (based on a
 performance metric).

 Finally, note that the choice of one modeling step over another depends on
 the outcome one wants, on the context in which the model is deployed. A
 general theme of the analysis is that different modeling steps lead to
 different values of different performances metrics, and **the choice of
 performance metric around which to optimize depends on how one is intending
 to take action based on the results of the model.** Overall, I did not find
 significant differences among the values performance metrics among different
 approaches I tried in the initial analysis.

# Full pipeline code
 Although the problem solution is exploratory, the statement implies that I
 have some final method in mind. As such, code in the form of an `SkLearn
  Pipeline`
 can be found
  [here](so_text/main.py).
 The main steps of the `Pipeline` are:
 - Read the data into a data frame.
 - Encode stemmed word features, and perform `tf-idf`.
 - Feed the `tf-idf` matrix, and also an additional feature (length of doc), 
 into `XGBoost`.
 - Generate a performance report, comprising the `roc_auc`, a classification
  report and a
  confusion matrix.
While I used Naive Bayes to test various approaches against each other, in
 the end I used XGBoost because I wanted to take a classifier that handles
 mixed types well (length is numerical/ordinal) s, and because it trains
  rapidly. 

The performance report for the pipeline is is:
```
>>> from so_text.main import main
>>> main()
roc_auc score: 0.8438393894051489
              precision    recall  f1-score   support

           0       0.75      0.78      0.77     12497
           1       0.77      0.74      0.75     12503

    accuracy                           0.76     25000
   macro avg       0.76      0.76      0.76     25000
weighted avg       0.76      0.76      0.76     25000
```

# Main analysis

 In the below, I compare various techniques. The code for the analysis is
 mostly found in
 [readme_analysis_runs.py](so_text/readme_analysis_runs.py).

 
## On text processing
 ### Summary
 We compare text processing approaches to see how it affects the model
 performance, and ultimately select the one provided by the data. 
 ### Additional detail
 When we apply various text parsing and processing techniques to training text
 data, we change the data. Thus the overall model estimator trained on that is
 different as well. There is a trade-off between scraping the text too much,
 so that signal is lost, and scraping the text too little, so that
 there is de-normalization, i.e. two words that should be treated as the same
 are not. For example, in the course of text processing, one might remove a
 word that would have been signal that separated the `0` and `1` class.
 In particular, the provided parsed text removes some `html`, but some of
 StackOverflow’s articles involve `html` code intentionally, so that means
 some of that signal is lost. Conversely, if one does not pre-process the
 text at all, one would treat the words `Cluster` and `cluster` differently,
 even though from the point of view of word frequency they should not be. 
  
 For illustration, we can compare the performance of a `MNNaiveBayes
 ` classifier trained on unprocessed vs  processed data (see
 [`run_no_preprocess_example`](so_text/readme_analysis_runs.py)):
   
```
>>> run.run_no_preprocess_example()
roc_auc score: 0.8400422115864314
              precision    recall  f1-score   support

           0       0.74      0.78      0.76     12497
           1       0.77      0.73      0.75     12503

    accuracy                           0.75     25000
   macro avg       0.75      0.75      0.75     25000
weighted avg       0.75      0.75      0.75     25000

confusion matrix:
             0  1  <-  Predicted
         0 [9764 2733]
Actual   1 [3435 9068]
```
It appears the processed text
had very slightly worse results in this setup (see
 [`run_title_and_body_example_concat`](so_text/readme_analysis_runs.py
 )); this likely is the a result of an essentially even trade-off between text
  normalization and signal, as mentioned above:
```
>>> run.run_title_and_body_example_concat()
roc_auc score: 0.8397354019687591
              precision    recall  f1-score   support

           0       0.74      0.77      0.76     12497
           1       0.76      0.73      0.75     12503

    accuracy                           0.75     25000
   macro avg       0.75      0.75      0.75     25000
weighted avg       0.75      0.75      0.75     25000

confusion matrix:
             0  1  <-  Predicted
         0 [9650 2847]
Actual   1 [3338 9165]
```

Now let's try another text processor
[`process_text`](so_text/utils.py). The performance dips
 slightly more. This could be because I included the stop words from the
 English language, which may have too aggressively removed some words.
 A customized list of stop words could improve performance.

```
>>> run.run_alternate_parser_example()
roc_auc score: 0.8378418466596904
              precision    recall  f1-score   support

           0       0.74      0.77      0.76     12497
           1       0.76      0.73      0.75     12503

    accuracy                           0.75     25000
   macro avg       0.75      0.75      0.75     25000
weighted avg       0.75      0.75      0.75     25000

confusion matrix:
               0  1  <-  Predicted
         0 [9621 2876]
Actual   1 [3365 9138]

```
Overall, from this preliminary analysis it appears that at least for Naive
Bayes, text processing does not significantly improve results for these
metrics. Going forward, I will stick with the the preprocessed columns that
were provided in the data set.

## On feature engineering
 ### Summary
 Recall from the problem statement that standards for posts specify they
 should be: “*on topic*, *detailed*, and *not a duplicate of another question*”.
 - *On topic* posts can be represented through word frequency (`tf-idf`), as
  well as word similarity (word embeddings). Topic modeling could also
   provide a feature.
 - *Detailed* posts can be represented through:
    - length,
    - unique word count,
    - parts of speech analysis,
    - code richness.
 - *Duplicate* posts can be represented via a decision layer outside the
  model, or an word similarity approach as a feature.
  ### Additional detail
#### "On topic":
We can represent on-topic-ness in at least 3 ways:
- Word frequency similarity
- Word embeddings similarity
- Topic modeling

##### Word frequency similarity

 One way of feature engineering the concept of "on topic" is to represent word
 frequency and compare to previously accepted documents on that basis. Given an
  unseen text, the model
 would give a score close to `0` if the text was similar in word frequency to a
  document labeled `0` (i.e., accepted).
 Word similarity measure is achieved with tf-idf vectorization in the
 [final Pipeline](so_text/main.py):
  
```
    classifier = Pipeline([
        ('features', FeatureUnion([
...
                ('tfidf',
                 TfidfVectorizer(tokenizer=tokenizer,
                                 min_df=TFIDF_MIN_DF,
                                 max_df=TFIDF_MAX_DF,
                                 ngram_range=TF_IDF_NGRAM_RANGE)),

```

 However, there is some nuance here. If two documents have very close tf-idf
 encodings, they can potentially be duplicates of each other, and we would
 want to reject all except the original document. De-duplication can happen
 outside the predictive component of the model on test data/new data, or can
 be feature engineered into the model.
  
  As an aside, I pumped up the ngram range to (1,3) in the final pipeline.
  The previous Naive bayes approaches (e.g. `run_title_and_body_example_concat`)
  used unigrams, and so had I tried bigrams
  for comparison; the result is a slightly higher `roc_auc`:
  ```
  >>> run_bigrams_comparison()
roc_auc score: 0.8530721963369584
              precision    recall  f1-score   support

           0       0.75      0.79      0.77     12497
           1       0.78      0.74      0.76     12503

    accuracy                           0.77     25000
   macro avg       0.77      0.77      0.77     25000
weighted avg       0.77      0.77      0.77     25000
 ```
 
 ##### Word embeddings similarity
 Another way to see document similarity is embed the documents in a high
 dimensional pre-trained feature space. I did a quick and dirty Keras
 Sequential model (sigmoid activation function, after GloVe embedding) on the
  processed
  data to
  see if that would provide
  comparable
 performance off the
 bat. In that model, each word is embedded as a 100-dim vector.
 See [run_nn_word_similarity_example](so_text/readme_analysis_runs.py).
 The result I obtained was:
 ```
 >>> print(f"accuracy: {accuracy}, f1_score: {f1_score}")
      accuracy: 0.6299999952316284, f1_score: 0.6307042837142944
```
 As a starting point, this fared worse than taking the bag-of-words tf-idf Naive
 Bayes approach (`run_title_and_body_example_concat`). Although the results
 appear much worse than the Naive Bayes approach, There is still the
 potential to achieve better performance with the deep learning approach, by
 including other features.
##### Topic modeling
 Topic modeling can help us obtain some signal as well, or can help us to
 break down the classifier into multiple classifiers across topics. In
 particular, if we perform topic modeling and see that the texts are in
 distinct clusters, we can create a feature. If a new document is in a
 topic that tends to have more acceptance, for instance, that document
 would be more likely to be accepted. Alternatively, we could create
 multiple classifiers across topics and see if that ensemble of
 classifiers performs better.
#### "Detailed":
 *Detailed* can refer to  the richness, sophistication, complexity of
  the text.
-   Document *length* (word count) is a simple proxy for detail
  . For example:
```
>>> df['Body_length'] = df.Body_processed.apply(lambda x: len(x.split()))
>>> df['Title_length'] = df.Title_processed.apply(lambda x: len(x.split()))
>>> df[['Body_length', 'Title_length', 'label']].groupby('label').agg('mean')

  	  Body_length	Title_length
label		
0	  284.056282	9.722529
1	  137.723620	9.099180
```
 We see that accepted texts have both longer bodies and titles.
 Let's get some more statistics about length, combining titles and bodies:
```
>>> def q25(x):
>>>     return x.quantile(0.75)
>>> def q75(x):
>>>     return x.quantile(0.75)
>>> def q99(x):
>>>     return x.quantile(0.99)
>>> df.groupby('label').agg({'Length': [q25, q75, q99, 'mean','min', 'max']})

	q25	q75	q99	mean	        min	max
label						
0	332	332	1796.03	292.778811	13	11319
1	156	156	950.01	145.822800	7	27014
``` 
  Overall, the length appears to be correlated with acceptance, except
  perhaps on the extreme end, since the maximum length document was in fact
  rejected. This could
  indicate that the document was "too wordy". Moreover, this suggests there
  might not be a linear relationship between the length and the likelihood
  that the document is accepted. To this end, to see a view of the extreme
  cases, we take the top few :
 ```
>>> df[df.Length>10000].groupby('label').Length.describe()
	count	mean	std	min	25%	50%	75%	max
label								
0	2.0	11243.0	107.480231	11167.0	11205.0	11243.0	11281.0	11319.0
1	1.0	27014.0	NaN	27014.0	27014.0	27014.0	27014.0	27014.0
```
 Thus, although the lengthiest one is rejected, the next two were still
 accepted. Inspecting those by hand, the reason for the length appears to be
 typically that code output was pasted. The longest one was likely not
 rejected
 for this reason, but rather because the question was not well-posed:
 ```
 i downloaded a free web design template and while perusing the include files
  i found the following in a php include for template . php ... this appears
 to  be an attempt to encode something ? i tried to decode the $ m variable
 that  looks encoded but it decodes as junk ... what is this ? $ m =" j  | h
 | z | f..."
```
At the other end, the shortest posts (all rejected) had the bodies:

```
>>> df[df.Length < 10]['Body_processed']
36706    does the tablelayoutpanel exist in vs 2005 ?
44229    how would you subclass an nsoutlineview ?   
59260    any inbuilt jstree themes available ?       
69692    does anyone know of a html formatter ?      
85035    javac is giving errors while compiling      
90914    ..............................              
```
 It is clear that the lack of explanation/specificity in the bodies doomed the
 posts to
 close.

 The length of
 the document is not represented by the `tf-idf` features themselves, so I have
 encoded this as a feature, which is present in the main Pipeline:
 
```
    classifier = Pipeline([
        ('features', FeatureUnion([
        ...
            ('words', Pipeline([
                ('wordext', NumberSelector('Body_length')),
                ('wscaler', StandardScaler()),
            ])),
        ])),

```
 
 - Another possible feature is
 *unique word count*. The corpus of accepted posts has more than twice as many
 unique words as the corpus of closed posts:

```
>>> results_0, results_1 = set(), set()
>>> df[df['label'] == 1]['Text'].str.lower().str.split().apply(results_1.update)
>>> df[df['label'] == 0]['Text'].str.lower().str.split().apply(results_0.update)
>>> print("label 1 unique words: %d," % len(list(results_1)),
      "label 0 unique words: %d" % len(list(results_0)))

label 1 unique words: 179796, label 0 unique words: 376782
```
 However, as one would expect, document length and unique word count per
 document are strongly correlated:
 ```
>>> df['Unique']= df.Text.apply(lambda x: len(set(x.split(" "))))
>>> print(df.corr()['Length']['Unique'])
0.8069517864895855
```
 
- Ratios of *parts of speech* could also be an indication of how
 "detailed" the text is. That is, perhaps posts that
 are accepted tend to have some sort of distribution among different parts of
 speech -- there are fewer adjectives proportionally, etc. Such a feature
 could also provide signal for the post being "on topic".
 (Implemented via [`pos_ratio`](so_text/utils.py).)
 (However on my MacBook Air the code took too long to run, and to test
 this in the future, I'd parallelize this in a cluster.) 
- Similar to parts of speech, another feature that I would try is *code
 richness*. A feature that indicates which percentage of the document is English
 words -- perhaps
 accepted posts tend to have more code in the document proportionally than
 English words (since for StackOverflow, when posted errors, it is good
 practice to post the output of the error and the code that lead up to it).


#### "Not a duplicate":
##### Summary
 
 Because the sample provided is a sample of the original set, and no
 column is provided to indicate whether or not the post is a duplicate,
 there is no way to know from the data whether or not a post is a
 duplicate. Depending on how many actual duplicates there are, this becomes
 an adversarial component of the modeling process. In particular a document is
 likely to be
 accepted in the modeling setup if it is similar to a previously accepted
 document; however, if it is "too" similar, it may be a duplicate, and hence
 rejected.
 Two ways of dealing with this using unsupervised learning:
 - A decision layer with unsupervised clustering,
 - The results of unsupervised clustering as a feature in the model.
 
##### Additional detail
 
 Suppose we were given the full corpus, and the model was trained on the full
 data in the StackOverflow database, and was retrained each time a new doc and
 label came in. Then, suppose that a new document
 comes in and we want to score it using the trained model. Suppose the previous
 training set included just 1 document which the new document duplicates,
 and that the new document is very similar to the one in the training
 set, by whatever measure the model used -- embedding similarity, etc. Then
 if the document in the training set was accepted (i.e., `label 0`), the model is
 more likely to predict that the new document will be accepted as well. Thus,
 on the new datum, the model would be inaccurate.
 
 On the other hand, if the training set already contained the original
 accepted document, *as well as* some duplicates of it that were rejected
  because they were duplicates, the model would be more likely to reject the new
 document, because it was similar to the duplicates that were rejected.

 Now, in our case, there is a limited corpus (limited number of rows) for the
 training set, because it was sampled from a larger one. Given a model trained
 on the limited set, if a new document
 comes in, it may well be a duplicate. However, because the training set
 is just a sample, it may not contain the similar duplicates that were
 rejected on
 grounds of their duplication, and so the document may be more likely to be
 labelled `0` by the model. 

 There are at least two ways of dealing with the duplication issue using
  unsupervised learning:
 - Create a decision layer on top of the predictive process. (This would be
  my approach.)
  In particular, there is a database of text documents that StackOverflow
  has, presumably from which the training was sampled in the first place. As
  a new document comes in, we can measure the cosine similarity of its
  encoded vector to the other encoded documents in the database. We can set
  a threshold, based on business logic (say, the cost of rejecting posts). If
  the new document is within the threshold to another document in the
  database, it can be marked as a duplicate and not accepted.
 - Add a column based on similarity.  Cluster texts based on embedded
 similarity, selecting
 the threshold by hand
 so that it appears that most of the documents in each cluster are duplicates of
 each other. Now, for each cluster of texts, choose one text at random at put
 a `1`. The obvious disadvantage of this approach is that we cannot typically
 manually inspect each cluster to see that it only contains duplicates, so
 this is equally ad hoc as the decision layer before, and in my opinion is a
 worse approach because by including this as a feature we are reducing the
 transparency the explicit decision layer would have.
 
 
#### Title and body

 Strictly speaking, we should encode/do tf-idf for "title" and "body" separately.
 The reason is that there is some *signal to be gained from keeping subject
  lines
 and body separate* as features; the vocabulary world of the subject lines and
  body are
 potentially different and the document might be rejected based on the failure
 to adhere to standards for the subject line specifically, say.

 In practice, I tried both:
  - vectorizing a concatenated column,
 - vectorizing both separately.
 Then I ran these through Naive Bayes. 
 For vectorizing a concatenated column, I obtained the result (from above):

```
>>> run.run_title_and_body_example_concat()
roc_auc score: 0.8397354019687591
              precision    recall  f1-score   support

           0       0.74      0.77      0.76     12497
           1       0.76      0.73      0.75     12503

    accuracy                           0.75     25000
   macro avg       0.75      0.75      0.75     25000
weighted avg       0.75      0.75      0.75     25000

              0  1  <-  Predicted
         0 [9650 2847]
Actual   1 [3338 9165]
```
 For vectorizing both separately, I obtained the result:

```
>>> run.run_title_and_body_example_sep()
roc_auc score: 0.8482461288589771
              precision    recall  f1-score   support

           0       0.74      0.80      0.77     12497
           1       0.78      0.72      0.75     12503

    accuracy                           0.76     25000
   macro avg       0.76      0.76      0.76     25000
weighted avg       0.76      0.76      0.76     25000

confusion matrix:
             0  1  <-  Predicted
        0 [9971 2526]
Actual  1 [3546 8957]
```
 Overall, the methods perform comparatively well, though there's a small
 bump obtained by separating title and body as expected.


## On Algorithms
 I tried a few algorithms, both of which had similar performance of `roc_auc`:
 - Naive Bayes: 
     I tried the algorithm on the tf-idf alone. I chose Naive Bayes
     because it was fast to train, which was helpful for trying various
      approaches in limited time. It is also  the natural first
      choice, considering the
     analogies one can make between this problem and detecting spam.
     The result was already presented in the "Title and Body" section above
      or in the function
[`run_title_and_body_example_concat`](so_text/readme_analysis_runs.py).
 (Note that is the baseline code for Naive Bayes in this situation,
  incorporating no
 additional preprocessing or features.)
     The downside of the Naive Bayes approach is
     that it assumes independence across different words which is an
     oversimplification. (As an aside, I  also tried with n
     -grams, but the improvement across metrics was not significant.) Another
      issue with Naive Bayes is that order of words is not preserved, whereas
     the order of the words could provide some signal in terms of whether the
     document is accepted or not. For example, documents that are accepted could
     have more introductory words up front. Given more time for this reason I
     would try training an `LSTM`, which does consider word order.
 - Keras Sequential neural network:
    I trained a neural network with the GloVe embedding to see how well a 
    embedding document's similarity (distance) to an accepted post
    corresponds to acceptance or closure of posts.
- `XGBoost`: 
    I tried `XGBoost` on the `tf-idf` encoded features, and an additional
    numerical one of length of the document. The advantage of XGBoost is
    that it is relatively fast to train and allows for features of mixed types.

## Questions from the problem statement
For completeness, I return to the questions.

*1. What metric did you use to judge your approach’s performance, and how
did it perform? Why did you choose that metric?*
#### Summary
 I chose the `roc_auc`, and the best performance tended to be about
 `.84` depending on the method. I chose it mainly because it is
  threshold agnostic. Overall, I generate a report that has `roc_auc`,
 and Sklearn's classification report, which breaks down precision and recall
 by class (and also includes accuracy), because it's better to have multiple
 performance metrics at hand if one does not know what the focus of the
 performance metrics is (e.g. if we want to focus on the minority class
 capture rate, we'd choose recall). 

#### Additional detail
The advantage of the classification report is that it breaks down how well the
model is performing in different buckets for a particular threshold. However,
providing the classification report without further explanation can be
misleading, because the classification report is just for one threshold.
If we do not know what the threshold is going to be for our model, the
classification report has limited significance. The threshold should be
optimized around the outcome one wants to achieve with the model. For example,
if StackOverflow wants to use the model to automate the approval or declining of
posts, it should calculate the cost to the business of false positives and
negatives, and thereby
determine the threshold for the classifier accordingly. (Threshold selection is
normally done in such an ad-hoc fashion.)  Then the classification report will
have more meaning.

 `Roc-auc` can be a good measure of the model performance because it does not take
 into account the choice of threshold, which as explained, cannot be
 determined without additional information; it is a way of
 understanding *how well a model is performing across all thresholds*. (The
 probabilistic interpretation of the roc auc is that it is
 probability the model will score a randomly chosen positive class higher
 than a randomly chosen negative class. So this metric is useful insofar as
 the test set is reflective of the real distribution.) 
 If the real distribution is highly skewed, and one wants to optimize on the
 capture rates of the minority class, precision-recall
 curves can be better in this situation
 (see http://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf).
 Therefore, if the artificially balanced training dataset provided was sampled
 from a highly imbalanced dataset, I may look instead to the AUC-PR.



*2. The dataset we’ve given you is artificially balanced such that there’s an 
even split of closed posts to accepted posts. Should this influence the
metrics you measure?*
#### Summary
 Yes, although really, the fact that it is artificially balanced limits the
 usefulness of any performance metrics at all.
#### Additional detail
 Typically in practice the choice of performance metrics for the classifier
 depends on what outcome one is trying to achieve, not the data itself. That
 is, given extraneous business reasons -- such as if there is a greater cost to
 the business of miscategorizing one category or another, or if we interpret
 business needs as requiring evenly accurate categorization across categories
  -- we then choose a performance metric. For example, in the case of credit
   card fraud,
 the cost to the business of accepting a fraudulent transaction is much
 higher than rejecting an honest one, so the metric that is typically chosen
 is recall.
    
 Furthermore, the performance metrics we calculate above tell us how well the
  model
 performs on the held-out test set, not on real production data. 
 Regardless of which
 performance metric we choose for this problem set, we are still just measuring
 how it performs in the artificial world where there's an even split. In
  particular, since the test
 set has been split out
 from the artificially balanced set, the test set is
 itself artificially balanced. The predictive aspect of the model is only
 accurate on real production data if the
 distribution of the production data and the distribution of the training data
  are
 similar. In the case when the original
 data are skewed, but both classes are equally sampled, there is no reliable
 frequency information for the model to be trained on, and therefore the
 probability scores it outputs are unreliable on unseen data. If we then set a
 threshold to produce a classifier, the
 performance of the classifier still just indicates how well the classifier
 would perform in a world in which there is an even split between closed
 posts and accepted posts, not the real world, not on the original data from
  which this was sampled. 
 
 
 Still, it may be illuminating to explore why an artificially balanced training
  set
  may
 have been sought in the first place. One case is: suppose the cost to the
 business of rejecting a post is high. Then the purpose of our classification
 model could be to classify as many rejected posts well as possible while
 still maintaining reasonable performance overall. If the original data has
 very few rejected posts and many accepted posts, then evenly sampling from
 each class will introduce an implicit cost to the accepted posts, so that we
 may be classifying the rejected posts better at the expense of
 misclassifying more accepted posts. In such situations, of highly skewed
 data where we are more concerned with the correct classifications of
 the minority
 class (but still want to be reasonable), we then want to look at precision
 -recall curves
 to see how well our overall model is doing and choose a threshold based on
 the trade-off between precision and recall. Overall though, as long as there
 is sufficient data in the minority class to capture the patterns that
 define the minority class, artificially balancing training data is not an
 approach that seems to be supported by statisticians, though it's often
 done as a quick-and-dirty solution for the reasons above.

 
*3. How generalizable is your method? If you were given a different (disjoint)
random sample of posts with the same labeling scheme, would you expect it to
perform well? Why or why not? Do you have evidence for your reasoning?*
 - The limitations depend on how the random sampling is done, and how
  much of it is done. The closer the distribution is to the real distribution,
  the more reflective the performance metrics are of how well the model would
  behave in the real world, which could in theory mean that the model
  performs worse, because it was not that effective to begin with.
 - The more data we sample, the more duplicates we will catch and the more
  likely we are to capture the original document that later documents
   duplicated. That is, suppose a new post is a duplicate of an old one. If a
   different random sample misses the original post, the new post may be
   accepted mistakenly. The more we avoid this situation, the better.
 - I do not have real evidence to support my reasoning, because I do not have
  the original set from which the dataset provided was sampled. 

*4. How well would this method work on an entirely new close reason, e.g.
duplicate or spam posts?*
 - For duplicates, duplicates are an aspect of this problem statement and the
  method is already limited in the way I have previously described.
 - I believe the methods I have used would apply well to spam posts. In
  particular, the
  overall approach of using  `tf-idf` to measure word frequency on spam posts works,
 with the “title” analogous to email subject line, and
 the “body” analogous to the body of the email. 
 *"How well"* the method works depends on what we are trying to achieve
 with the classifier as it is deployed to StackOverflow. 
 
*5. Are there edge cases that your method tends to do worse with? Better? E.g.,
How well does it handle really long posts or titles?*
I created an analysis of the long and short edge cases below. Overall, the
 `roc_auc` falls on the edge cases.
 ```
>>> df = run_long_short_analysis()
Performance on docs of word length > 1700, comprising 199 documents
roc_auc score: 0.6801360544217687
              precision    recall  f1-score   support

           0       0.77      0.98      0.86       150
           1       0.67      0.12      0.21        49

    accuracy                           0.77       199
   macro avg       0.72      0.55      0.54       199
weighted avg       0.75      0.77      0.70       199

confusion matrix: 

             0  1  <-  Predicted
          0 [147   3]
Actual   1 [43  6]
None
Performance on docs of word length < 20, comprising 81 documents
roc_auc score: 0.7022222222222223
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.92      0.88      0.90        75

    accuracy                           0.81        81
   macro avg       0.46      0.44      0.45        81
weighted avg       0.85      0.81      0.83        81

confusion matrix: 

            0  1  <-  Predicted
          0 [0 6]
Actual   1 [ 9 66]
```
*6. If you got to work on this again, what would you do differently (if
 anything)?*

- There is one thing I would do differently if I could roll back time to a
 a few days ago. I wanted to try out `SkLearn`'s
 `Pipeline` for the first time (because I love thinking about pipelines in ML
 ! :) ) but could not obtain feature importances from it for the inner
  components of FeatureUnion (this
   corresponds to an 
   [issue in GitHub](https://github.com/scikit-learn/scikit-learn/issues/6425
   ). Therefore as of this writing I do not know whether length is worth
    keeping in the model. 

Otherwise, here's what I would do with more time (most of these I have
 mentioned above):
 - **Obtain the original dataset** and train the model on that to avoid issues
  from artificial balancing.
 - Reorganize the code into an **auto experimentation repo**. The approach of
  focusing on the analysis
  first means
  that the code is too disorganized for my tastes  -- there's no overarching
  design because it's
  not software. For example, in
  [readme_analysis_runs](so_text/readme_analysis_runs.py),
  I broke out the code from the main Pipeline, so there is code duplication. I
  did this to have more control over the steps, as well as to iterate faster
  for the sake of quick analysis of whether that approach works.
  However, I believe this was necessary to quickly try out a bunch of
  approaches. In future versions I could reorganize code to facilitate
  automatic experimentation of a few approaches (different parameters in a
  config file, etc.)
- Perform **topic modeling** to check if there are specific topics that dominate
 the
 dataset. If there are, include those as features.
- Perform **unsupervised clustering of similarity** for documents to catch
 duplicates,
 either as a decision layer on top of the predictive model, or as a feature.
- Perform **anomaly detection removal** to remove data with
 issues. In particular, when we embed
 the words in the embedding space (say via GloVe or the like), there will be
 some points (documents) that fall a great distance away from the others. I
 would try removing those and retraining the model.
- **Contribute to the `sklearn` code** to fix the feature importances issue
 to perform feature importance; otherwise, I would discard the Pipeline
 approach.
- Improve the **deep learning** approach, including **more features** than just
 embedded documents, and
 try an **LSTM** to see if preserving word ordering helps classify better. 

*7. If you found any issues with the dataset, what are they?*

 - The fact the data is **artificially balanced and subsampled** means that the
  model is not predictively useful in the real world (as explained in 2). The
   fact that it
  is subsampled also means that duplicate posts lower the performance of the
  model, since if a document is similar to an accepted one in
  terms of word frequency similarity or word embedding similarity, it is
  impossible to know from the data provided whether it is a duplicate or not,
  and hence whether it is more likely to be accepted (because it's similar)
  or rejected (because it's a duplicate).
 - Missing data:
 There were two missing values because of misapplied parser:
  ```

Title	                                            label	Title_processed
<script async> not working in rails	                0	NaN
<rich:popupPanel buttons not working in JSF Pr...	0       NaN

```
 Since there were only two, it was not an issue to simply drop those values.
 This suggests that there could be other issues at the parsing stage. However,
 applying my own parser did not lead to a significant difference in
 performance, as discussed above.
- With more time, as discussed, I would try anomaly detection, because hand
 inspection did not lead me to find other issues with the dataset.
 
 # Thank you for reading! :)

 
 