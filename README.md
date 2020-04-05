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
 automation and predictive modeling. Here is a row of the data that was
 provided:
 
```
                                Title:  \
RUBY: most common number for Users   

                                                Body  label:  \
<p>Hi I'm new to <code>Ruby on Rails</code>. I...      1   

                                      Body_processed:  \
hi i'm new to ruby on rails . i created users ...   

                       Title_processed: 
ruby : most common number for users  

```
 My approach was to build a predictive model that is a classifier
 once a threshold is applied. The limitations of my approach for this problem
 context are
 addressed in the analysis below. For one, without the cost of false
 positives and negatives, and an understanding of which call to make in the
 trade-off between better classifying one category over another, it is not
 clear how to set the threshold. In classifying I provisionally went with a
 threshold of `.5`, more for a concrete *illustration* of how the classifier
 classifies points at *some* threshold. For the performance metric, I chose
 `roc_auc` because it shows performance across all thresholds on the test
 set. However, because the test set is held out from an artificially
 balanced set, it is itself artificially balanced, which means that if we
 optimize around performance metrics, we are not optimizing
 around how the model would perform in the wild, but rather the conditions of
 the test set. For example, it is unclear to me that the
 imbalanced nature of the test set means that the rank ordering of models is
 preserved. That is, just because one ML approach has a better or worse
 outcome in terms of performance metrics on the test set, it is unclear to me
 that it
 should be chosen over a different one. See the sections "On
 Evaluating Performance" and the answer to Question 2 below for a more
 detailed discussion.

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
 different values of different performances metrics, and the choice of
 performance metric around which to optimize depends on how one is intending
 to take action based on the results of the model. Overall, I did not find
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

 
# Main analysis
 In the below, I compare various techniques. The code for the analysis is
 mostly found in
 [readme_analysis_runs.py](so_text/readme_analysis_runs.py),
 This analysis captures the sandbox nature of data-science, where once tries
 different approaches in the effort to find evidence to pursue an
 approach further.
 
## On text processing
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

         0  1  <-  Predicted
       0 [9764 2733]
True   1 [3435 9068]
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

         0  1  <-  Predicted
       0 [9650 2847]
True   1 [3338 9165]
```

Now let's try another text processor
[`process_text`](so_text/feature_process_utils.py). The performance dips
 slightly more.

```
>>> run.run_alternate_parser_example()
roc_auc score: 0.8378418466596904
              precision    recall  f1-score   support

           0       0.74      0.77      0.76     12497
           1       0.76      0.73      0.75     12503

    accuracy                           0.75     25000
   macro avg       0.75      0.75      0.75     25000
weighted avg       0.75      0.75      0.75     25000

         0  1  <-  Predicted
       0 [9621 2876]
True   1 [3365 9138]

```
Overall, from this preliminary analysis it appears that at least for Naive
Bayes, text processing does not significantly improve results for these
metrics. Going forward, I will stick with the the preprocessed columns that
were provided in the data set.

## On feature engineering
Recall from the problem statement that standards for posts specify they
 should be: “*on topic*, *detailed*, and *not a duplicate of another question*.”
 
### "On topic":
 One way of feature engineering the concept of "on topic" is to represent word
 similarity to previously accepted documents. Given an unseen text, the model
 would give a score close to `0` if the text was similar in words
 and word frequency to a document labeled `0` (i.e., accepted).
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
 See the "Not a duplicate" discussion below for more.

### "Detailed":
 *Detailed* can refer to  the richness, sophistication, complexity of
  the text. Document length (in terms of words) is a simple proxy for detail
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
 We see that accepted texts have both longer bodies and titles. The length of
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
 
 Another possible feature is
 unique word count. The corpus of accepted posts has more than twice as many
 unique words as the corpus of closed posts:
```
>>> results_0, results_1 = set(), set()
>>> df[df['label'] == 1]['Text'].str.lower().str.split().apply(results_1.update)
>>> df[df['label'] == 0]['Text'].str.lower().str.split().apply(results_0.update)
>>> print("label 1 unique words: %d," % len(list(results_1)),
      "label 0 unique words: %d" % len(list(results_0)))

label 1 unique words: 179796, label 0 unique words: 376782
```
 However, as expected, document length and unique word count per document are
 strongly correlated, so I will leave it out of the Pipeline:
 ```
>>> df['Unique']= df.Text.apply(lambda x: len(set(x.split(" "))))
>>> print(df.corr()['Length']['Unique'])
0.8069517864895855
```
 
Ratios of parts of speech could also be an indication of how
"detailed" the text is. That is, perhaps posts that
are accepted tend to have some sort of distribution among different parts of
speech -- fewer adjectives proportionally, etc. Such a feature
could also provide signal for the post being "on topic".
(Implemented via [`pos_ratio`](so_text/feature_process_utils.py).)
However on my MacBook Air the code took too long to run, and to test
this in the future, I'd parallelize this in a cluster.


### "Not a duplicate":
 Suppose we were given the full corpus, and the model was trained on the full
 data, and retrained each time a new doc and label came in. (If there is class
 imbalance, we can attempt to remedy it with upsampling techniques, but we
 ignore that detail for this discussion.) Then, suppose that a new document
 comes in and we want to score it using the trained model. Suppose the previous
 training set included just 1 actual "duplicate" of the new document
 , and the new document is very similar to the one in the training set. Then
 if the document in the training set was accepted (i.e., `label 0`), the model is
  more
 likely to predict that the new document will be accepted as well. Thus, on the
 new datum, the model would be inaccurate.
 
 On the other hand, if the training set already contained the original
 “accepted document” and some duplicates of it that were rejected because
 they were duplicates, the model would be more likely to reject the new
 document, because it was similar to the duplicates that were rejected.

 Now, in our case, there is a limited corpus (limited number of rows) for the
 training set, because it was sampled from a larger one. So the question becomes
 even more muddied. Given a model trained on the limited set, if a new document
 comes in, it may well be a duplicate. However, because the training set
 is a sample, it may not contain the similar duplicates that were rejected on
 grounds of their duplication, and so the document may be more likely to be
 labelled `0` by the model. 

 There are at least two ways of dealing with the duplication issue:
 - Create a decision layer on top of the predictive process.
  In particular, there is a database of text documents that StackOverflow
  has, presumably from which the training was sampled in the first place. As
  a new document comes in, we can measure the cosine similarity of its
  encoded vector to the other encoded documents in the database. We can set
  a threshold, based on business logic (say, the cost of rejecting posts). If
  the new document is within the threshold to another document in the
  database, it can be marked as a duplicate and not accepted.
https://towardsdatascience.com/de-duplicate-the-duplicate-records-from-scratch-f6e5ad9e79da
 - Add a column based on similarity.  Cluster texts based on
 similarity, using, say,
 [word movers distance](http://proceedings.mlr.press/v37/kusnerb15.pdf), selecting
 the threshold by hand
 so that it appears that all the documents in each cluster are duplicates of
 each other. Now, for each cluster of texts, choose one text at random at put
 a `1`. The obvious disadvantage of this approach is that we cannot typically
 manually inspect each cluster to see that it only contains duplicates, so
 this is equally ad hoc as the decision layer before.
 
 
### Title and body

 Strictly speaking, we should encode/do tf-idf for "title" and "body" separately.
 The reason is that there is some signal to be gained from keeping subject lines
 and body separate; the vocabulary world of the subject lines and body are
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
True   1 [3338 9165]
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

             0  1  <-  Predicted
       0 [9971 2526]
True   1 [3546 8957]
```
 Overall, it appears that separating the vectorizations improves the
 performance as expected,if slightly.


## On Algorithms
 I tried two algorithms, both of which had similar performance of `roc_auc`:
 - Naive Bayes: 
     I tried the algorithm on the tf-idf alone. I chose Naive Bayes
     because it was fast to train, and the natural first choice, considering the
     analogies one can make between this problem and detecting spam.
     The result was already presented in the "Title and Body" section above
      or in the function
[`run_title_and_body_example_concat`](so_text/readme_analysis_runs.py).
 (Note that is the baseline code for naive bayes in this situation
 , incorporating no
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
- XGBoost: 
    I tried XGBoost on the tf-idf encoded features, and an additional
    numerical one of length of the document. The advantage of XGBoost is
    that it is relatively fast to train and allows for features of mixed types.

## On Evaluating performance
Overall, I generate a report that has `roc_auc`,  and sklearn's
classification report, which breaks down precision and recall by class (and
also includes accuracy). 

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
 determined without additional information; it is essentially a way of
 understanding how well a model is performing across all thresholds. 
 If the real distribution is highly skewed, and one wants to optimize on the
 capture rates of the minority class, recision-recall
 curves can be better in this situation
 (see http://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf).
 Therefore, if the artificially balanced training dataset provided was sampled
 from a highly imbalanced dataset, I may look instead to the AUC-PR.

## Questions from the problem statement
For completeness, I return to the questions.

*1. What metric did you use to judge your approach’s performance, and how
did it perform? Why did you choose that metric?*

 See “On Evaluating Performance” above, and the next question.

*2. The dataset we’ve given you is artificially balanced such that there’s an 
even split of closed posts to accepted posts. Should this influence the
metrics you measure?*
 
 Typically in practice the choice of performance metrics for the classifier
 depends on what outcome one is trying to achieve, not the data itself. That
 is, given extraneous business reasons -- such as if there is a greater cost to
 the business of mis-categorizing one category or another, or if we interpret
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
 mis-classifying more accepted posts. In such situations, of highly skewed
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

 
 See “On Evaluating Performance” above for more details.

*3. How generalizable is your method? If you were given a different (disjoint)
random sample of posts with the same labeling scheme, would you expect it to
perform well? Why or why not? Do you have evidence for your reasoning?*
 
 It would potentially suffer from the same limitations as the original
 random sample. One limitation that still stands is that with duplicates.
 In short, suppose a new post is a duplicate of an old one. If a different
 random sample misses the original post, the new post may be accepted
 mistakenly.

*4. How well would this method work on an entirely new close reason, e.g.
duplicate or spam posts?*
 For duplicates, see discussion above "feature engineering/Not a duplicate".
 Duplicates are an aspect of this problem statement and the method is already
 limited in the way I have described.
 
 Again, *"how well"* the method works depends on what we are trying to achieve
 with the classifier as it is deployed to StackOverflow. The overall
 approach of using  `tf-idf` to measure word similarity on spam posts works,
 with the “title” analogous to email subject line, and
 the “body” analogous to the body of the email. 
 
*5. Are there edge cases that your method tends to do worse with? Better? E.g.,
How well does it handle really long posts or titles?*
 
 Ive included the length of the concatenated body as a feature, so it should
 accommodate that.
 [Do ad hoc analysis on how well it did with really long posts though]

*6. If you got to work on this again, what would you do differently (if
 anything)?*
 Throughout the document I have listed what I would do with more time, but also:
- Generally speaking, the approach of focusing on the analysis first means
  that the code is too disorganized for my tastes  -- there's no overarching
  design because it's
  not software. For example, in
  [readme_analysis_runs](so_text/readme_analysis_runs.py),
  I broke out the code from the Pipeline, so there is code duplication. I
  did this to have more control over the steps, as well as to iterate faster
  for the sake of quick analysis of whether that approach works.
  However, I'm not sure there is really a better way to organize it for the
  in terms of this problem set.
- With more time, I would get my hands on the original dataset because I am
  curious to see what would be the results if the training data was not
  artificially balanced.

*7. If you found any issues with the dataset, what are they?*
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
