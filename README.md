To install all the dependencies required to execute the code, clone the repo
locally, `cd` to the root, and run 

```
pip install -e .
```

# Introduction, caveats

 In this README, I discuss my solution to the
 [problem statement](docs/problem_statement.pdf).
 The data comprises the text of questions on StackOverflow, with labels that
 indicate whether the question was accepted or closed. The set-up is that
 currently the acceptance/rejection is done
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
 

The problem statement emphasizes exploration and analysis over performance
fine-tuning and deployment. Thus I experimented
with approaches throughout the initial modeling process, employing various
preprocessing, feature engineering, and algorithm techniques, which
needed to be then represented in code in the repo. Some
approaches took too long to execute on my MacBook Air, such as the
breakdown of a document into its parts of speech to assess sentence
sophistication/complexity. Given more time to experiment, I would
launch a cloud cluster and parallelize the execution of such methods.
         
Because most of the below explores possible approaches, and the repo itself
is intended to just capture the code behind the analysis, the repo
does not follow the usual standards of a Python library. There is
necessarily some dead code; certainly, it's not software.  Nevertheless, with
more time I would still organize the code into a coherent whole that is
focused on automating some experimentation to facilitate performance 
fine-tuning. Specifically, I would have a config file that populates the hardcoded
hyperparameters with different combinations, applies different algorithms
 specified in the config, and  returns the optimal result (based on a
performance metric).

Finally, note that the choice of one modeling step over another depends on
the outcome one wants, on the context in which the model is deployed. A
general theme of the analysis is that different modeling steps lead to
different values of different performances metrics, and the choice of
performance metric around which to optimize depends on how one is intending
to take action based on the results of the model. Overall, I did not find
significant differences in performance metrics among different
approaches I tried in the initial analysis.

# Full pipeline code
 Although the problem is exploratory, the statement also indicates that I
 should choose a final method. As such, code in the form of an SkLearn Pipeline
 can be found
  [here](so_text/main.py).
 The main steps of the Pipeline are:
 - Read the data into a data frame
 - Encode stemmed word features, and perform tf-idf
 - Feed the tf-idf matrix, and also an additional feature (length of doc
 ), into XGBoost
 - Generate a performance report

 
# Main analysis

## On text processing
When we apply various text parsing and processing techniques to training text
data, we change the data. Thus the overall model estimator trained on that is
different as well. There is a trade-off between scraping the text too much
, so that important signal is lost, and scraping the text too little, so that
there is de-normalization, i.e. two words that should be treated as the same
are not. For example, in the course of text processing, one might remove a
word that would have been signal that separated the `0` and `1` class
. In particular, the provided parsed text removes some `html`, but some of
StackOverflow’s articles involve `html` code intentionally, so that means some of
that signal is lost. Conversely, if one does not pre-process the text at all
, one
would treat the words `Cluster` and `cluster` differently, even though
from the point of view of word frequency they should not be. 
  
 For illustration, we can compare the performance of the final pipeline on the
 unprocessed vs the processed code:
   
```
Performance for unprocessed text
Performance for processed text
```

We can see that the processed text performs differently in way x. 
Let us compare the top `tf-idf` features:

```
insert top tf-idf features for both 
```

Now let's try another parser. Link here.

```
Performance with processed text according to parser.
```

Example of how a particular sentence was misclassified with one or another.

## On feature engineering
Recall from the problem statement that standards for posts specify they
 should be: “*on topic*, *detailed*, and *not a duplicate of another question*.”
 
### “On topic”:
 One way of feature engineering the concept of "on topic" is to represent word
 similarity to previously accepted documents. Given an unseen text, the model
 would give a score close to `0` if the text was similar in words
 and word frequency to a document labeled `0` (i.e., accepted).
 Word similarity measure is achieved with tf-idf vectorization in the
  Pipeline:
  
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

### “Detailed”:
 *Detailed* can refer to  the richness, sophistication, complexity of
  the text. Length can be an indication of detail. For example:
```
>>> df[['Body_length', 'Title_length', 'label']].groupby('label').agg('mean')

  	  Body_length	Title_length
label		
0	  284.056282	9.722529
1	  137.723620	9.099180
```
We see that accepted texts have both longer bodies and titles. The length of
the document is not represented by the tf-idf features themselves, so I have
encoded this as a feature, which is present in the main pipeline.
Another possible feature, which would be highly correlated with length, is
unique word count. Accepted posts have more than twice as many unique words:
```
>>> results_0, results_1 = set(), set()
>>> df[df['label'] == 1]['Text'].str.lower().str.split().apply(results_1.update)
>>> df[df['label'] == 0]['Text'].str.lower().str.split().apply(results_0.update)
>>> print("label 1 unique words: %d," % len(list(results_1)),
      "label 0 unique words: %d" % len(list(results_0)))
label 1 unique words: 179796, label 0 unique words: 376782
```
Ratios of parts of speech could also be an indication of how
"detailed" the text is. That is, perhaps posts that
are accepted tend to have some sort of distribution among different parts of
speech -- fewer adjectives proportionally, etc. Such a feature
could also provide signal for the post being "on topic".
(Implemented via [`pos_ratio`](so_text/feature_process_utils.py).)
As mentioned in the introduction, the code took too long to run, and to test
this in the future I'd parallelize this in a cluster.
 
### “Not a duplicate”:
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
 similarity, using, say, k-means clustering, selecting the threshold by hand
 so that it appears that all the documents in each cluster are duplicates of
 each other. Now, for each cluster of texts, choose one text at random at put
 a `1`. The obvious disadvantage of this approach is that we cannot typically
 manually inspect each cluster to see that it only contains duplicates, so
 this is equally ad hoc as the decision layer before.
 
 
### Title and body

 Strictly speaking, we should encode/do tf-idf for "title" and "body
 " separately.
 The reason is that there is some signal to be gained from keeping subject lines
 and body separate; the vocabulary world of the subject lines and body are
 potentially different and the document might be rejected based on the failure
 to adhere to standards for the subject line specifically, say.

 In practice, I tried both:
 - vectorizing both separately,
 - vectorizing a concatenated column.
Then I ran these through Naive Bayes. 
For vectorizing both separately, I got the result:

```
Paste result
```

For vectorizing a concatenated column, I got the result:

```
Paste result
```
See the [code here](so_text/readme_analysis_runs.py).
The results were similar enough that I didn't feel compelled to pursue this
direction further. Of course, here I am trying Naive Bayes, whereas in the
"main pipeline" I stuck with XGBoost, so I could still try it with that
 algorithm.
   
### Feature selection for the model
 Since the analysis is not concerned with effectiveness, I have left the
 question aside of feature selection for now. With more time I'd follow
 [this blog post](https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/) to integrate feature selection with the sklearn
 Pipeline. 

## On Algorithms
 I tried two algorithms, both of which had similar performance of `roc_auc`:
 - Naive Bayes: 
     I tried the algorithm on the tf-idf alone. I chose Naive Bayes
     because it was fast to train, and the natural first choice, considering the
     analogies one can make between this problem and detecting spam.
     The result was already presented in the "Title and Body" section above
      or in the function [`run_title_and_body_example_concat`](so_text
/readme_analysis_runs.py). (Note that is the barest code, incorporating no
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

 Roc_auc can be a good measure of the model performance because it does not take
 into account the choice of threshold, which as explained, cannot be
 determined without additional information; it is essentially a way of
 understanding how well a model is performing across all thresholds. However
 , because the roc_auc is a curve between the true and false positive rates,
 in the case of highly imbalanced data, it does not indicate how well the model
 performs on minority classes. Precision-recall curves can be better in this
 situation
 (see http://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf).
 Therefore, if the artificially balanced training dataset provided was sampled
 from a highly imbalanced dataset, I may look instead to the AUC-PR.

## Questions from the problem statement
Wrapping up, I return to the questions to make sure they're addressed.

*What metric did you use to judge your approach’s performance, and how
did it perform? Why did you choose that metric?*

 See “On Evaluating Performance” above.

*The dataset we’ve given you is artificially balanced such that there’s an 
even split of closed posts to accepted posts. Should this influence the
metrics you measure?*

 The metrics tell us how well the model performs on the held out test set. Since
 the test set has been split out from the artificially balanced set, it is
 itself artificially balanced. If in fact, the original data from which the
 artificially balanced set was chosen is highly skewed (with many more posts
 accepted than closed, or vice versa) the model could be less ineffective
 on unseen data from the imbalanced original. See
 “On Evaluating Performance” above for more details.

*How generalizable is your method? If you were given a different (disjoint)
random sample of posts with the same labeling scheme, would you expect it to
perform well? Why or why not? Do you have evidence for your reasoning?*
 It would potentially suffer from the same limitations as the original
 random sample. One limitation that still stands is that with duplicates
 -- see the section "On feature engineering/Not a duplicate”. In short
 , suppose a new post is a duplicate of an old one. If a different random
 sample misses the original post, the new post may be accepted mistakenly.

*How well would this method work on an entirely new close reason, e.g.
duplicate or spam posts?*
 For duplicates, see discussion above ““feature engineering/Not a duplicate”.
 Duplicates are an aspect of this problem statement and the method is already
 limited in the way I have described.  I think the approach would work
 well on spam posts, with the “title” analogous to email subject line, and
 the “body”
 analogous to the body of the email. On the other hand, spam could
  potentially be more imbalanced, and the class imbalance could be remedied
   with upsampling techniques, taken with care.
 
*Are there edge cases that your method tends to do worse with? Better? E.g.,
How well does it handle really long posts or titles?*
 Ive included the length of the concatenated body as a feature, so it should
  accommodate that.
 [Do ad hoc analysis on how well it did with really long posts though]

*If you got to work on this again, what would you do differently (if anything)?*
 Throughout the document I have listed what I would do with more time. I would
 work on those things.

*If you found any issues with the dataset, what are they?*
 There were two missing values because of misapplied parser. This suggests that
 there could be other issues at the parsing stage. However, applying my own
 parser did not lead to a significant difference in performance, as discussed
 above.
