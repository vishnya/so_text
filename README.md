

To install all the dependencies, clone the repo locally, `cd` to the root,
and run `pip install -e .`.

# Link to problem statement

# Up-front caveats
I interpreted the problem statement to be emphasizing exploration and
analysis  over  performance fine-tuning and deployment. Thus I experimented
with approaches throughout the initial modeling process, employing various
preprocessing, feature engineering, and algorithm techniques, which
needed to be then partially represented in code in the repo. Some
approaches took too long to execute on my MacBook Air, such as the
breakdown of a document into its parts of speech to assess sentence
sophistication/complexity. Given more time to experiment, I would
launch a cloud cluster and parallelize the execution of such methods.
         
         
Because most of the below explores possible approaches, and the repo itself
is intended to just capture the code behind the experimentation, the repo
does not follow the usual standards of a Python library. For instance, there
is dead code and code duplication.  Nevertheless, with more time I
would still organize the code into a coherent whole that is focused on
automating some experimentation to facilitate performance fine-tuning.
Specifically, I would have a config file that populates the hardcoded
hyperparameters with different combinations, applies different
algorithms specified in the config, and  returns the optimal result (based on a
performance metric).

Finally, note that the choice of one modeling step over another depends on
the outcome one wants, on the context in which the model is deployed. A
general theme of the analysis is that different modeling steps lead to
different values of different performances metrics, and the choice of
performance metric around which to optimize depends on how one is intending
to take action based on the results of the model.

# Full pipeline code
To address the implicit expectations of the problem statement up front, a run of
a modeling pipeline can be found here.
 
# Main analysis

## On text processing
When we apply various text parsing and processing techniques to training text
data, we change the data. Thus the overall model estimator trained on that is
different as well. There is a trade-off between scraping the text too much
, so that important signal is lost, and scraping the text too little, so that
there is denormalization, i.e. two words that should be treated as the same
are not. For example, in the course of text processing, one might remove a
word that would have been signal to separate the `0` and `1` class
. In particular, the provided parsed text removes some html. But some of
StackOverflow’s articles involve html code intentionally, so that means some of
that signal is lost. Conversely, if one does not preprocess the text at all, one
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
 should be: “*on topic*, *detailed*, and *not a duplicate* of another question.”
 
### “On topic”:
 One way of feature engineering the concept of "on topic" is to represent word
 similarity to previously accepted documents. Given an unseen text, the model
 would give a score close to `0` if the text was similar in words
 and word frequency to a document labeled `0` (accepted).
 Word similarity measure is achieved with tf-idf vectorization in the pipeline.  

 However, there is some nuance here. If two documents have very close tf-idf
 encodings, they can potentially be duplicates of each other, which we would
 want to reject. Deduplication can happen outside the predictive component of
 the model on test data/new data, or can be feature engineered into the model.
 See the "Not a duplicate" discussion below.

### “Detailed”:
 This refers to potentially the richness, sophistication, complexity of the
 text. Length can be an indication of detail, and I have encoded this as a
 feature. (LINK) Ratios of parts of speech could be an indication of how
 "detailed" the text is. That is, perhaps posts that
 are accepted tend to have some sort of distribution among different parts of
 speech -- not too many adjectives, etc. Such a feature
 could also provide signal for "on topic". (Link to code)

### “Not a duplicate”:
 Suppose we were given the full corpus, and the model was trained on the full
 data, and retrained each time a new doc and label came in. (If there is class
 imbalance, we can attempt to remedy it with upsampling techniques, but we
 ignore that detail for this discussion.) Then, suppose that a new document
 comes in and we want to score it using the trained model. Suppose the previous
 training set included just 1 actual ``duplicate'' of the new document
 , and the new document is very similar to the one in the training set. Then
 if the document in the training set was accepted (label 0), the model is more
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
 accepted by the mode. 

 There are at least two ways of dealing with the duplication issue:
 - Create a decision layer on top of the predictive process.
 . In particular, there is a database of text documents that StackOverflow
  has, presumably from which the training was sampled in the first place. As
  a new document comes in, we can measure the cosine similarity of its
  encoded vector to the other encoded documents in the database. We can set
  a threshold, based on business logic (say, the cost of rejecting posts). If
  the new document is within the threshold to another document in the
  database, it can be marked as a duplicate and not accepted.
https://towardsdatascience.com/de-duplicate-the-duplicate-records-from-scratch-f6e5ad9e79da
 - Add a column based on similarity.  Cluster texts based on
 similarity, using, say, k-means clustering, selecting the threshold by hand
 so that it seems that all the documents in each cluster are duplicates of
 each other. Now, for each cluster of texts, choose one text at random at put
  a `1`.
 
 
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
 
 The roc_auc of the model I tried it with with Naive Bayes differed only
 slightly (Code). Thus going forward, I combined the two into one column. (In
 the future I could try them separately again for other algorithms)

### Feature selection for the model
 Since the analysis is not concerned with effective methods, I have left the
 question aside of feature selection for now. With more time I'd follow this
 post: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit
  -learn-pipeline/

## On Algorithms
 I tried two algorithms, with similar performance of `roc_auc`:
 - Naive Bayes
     I tried the algorithm on the tf-idf alone. I chose Naive Bayes
     because it was fast and easy, the natural first choice, considering the
     analogies one can make between this problem and detecting spam.
     . The
      downside of the approach is
     that it assumes independence across different n-grams which is an
     oversimplification. Another issue is that order is not preserved, whereas
     the order of the words could provide some signal in terms of whether the
     document is accepted or not; for example, documents that are accepted could
     have more introductory words up front. Given more time for this reason I
     would try training an LSTM, which does consider word order.
- XGBoost
    I tried XGBoost on the tf-idf encoded features, and an additional
    numerical one of length of the document. The advantage of XGBoost is
    that it is relatively fast to train and allows for features of mixed types.
    https://towardsdatascience.com/word-bags-vs-word-sequences-for-text-classification-e0222c21d2ec


## On Evaluating performance
Overall, I chose to do both a classification report and an roc_auc. (Snippet
 from final pipeline)
The advantage of the classification report is that it breaks down how well the
model is performing in different buckets for a particular threshold. However,
this is misleading, because the classification report is just for one threshold.
If we do not know what the threshold is going to be for our model, the
classification report has limited significance. The threshold should be
optimized around the outcome one wants to achieve with the model. For example,
if StackOverflow wants to use the model to automate the approval or declining of
posts, it should calculate the cost of false positives and negatives, and
determine the threshold for the classifier accordingly. (Threshold selection is
normally done in such an ad-hoc fashion.)  Then the classification report will
have more meaning.

 Roc_auc can be a good measure of the model performance because it does not take
 into account the choice of threshold, which as explained is scientifically
 arbitrary. However, because the roc_auc is a curve between the TPR and the FPR,
 in the case of highly imbalanced data, it does not indicate how well the model
 performs on minority classes. Precision-recall curves can be better in this
 situation
 (see http://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf).
 Therefore, if the training dataset provided to me was sampled from a highly
 imbalanced dataset, I would perhaps look to the AUCPR.

# Questions from the problem statement:

## What metric did you use to judge your approach’s performance, and how

## did it perform? Why did you choose that metric?
 See “On Evaluating Performance” above.

## The dataset we’ve given you is artificially balanced such that there’s an 
## even split of closed posts to accepted posts. Should this influence the
## metrics you measure?

 The metrics tell us how well the model performs on the held out test set. Since
 the test set has been split out from the artificially balanced set, it is
 itself artificially balanced. If in fact, the original data from which the
 artificially balanced set was chosen is highly skewed (with many more posts
 accepted than closed, or vice versa) the model could be less ineffective
 on unseen data from the imbalanced original. See
 “On Evaluating Performance” above for more details.

## How generalizable is your method? If you were given a different (disjoint)
## random sample of posts with the same labeling scheme, would you expect it to
## perform well? Why or why not? Do you have evidence for your reasoning?
 Since the data seems to be “not too small” for the approach chosen, it would
 potentially suffer from the same limitations previously, but if it was
 truly randomly chosen, no new limitations come to mind.
 One limitation that still stands is that with duplicates -- see the section 
 "On feature engineering/Not a duplicate”. In short, for duplicates, a
 different random sample will potentially be missing an original post
 (the original non-duplicated post which future posts duplicated), so that
 given a new post, depending on how one handles duplicates,
 it can be treated as the original, whereas it is a duplicate.

## How well would this method work on an entirely new close reason, e.g.
## duplicate or spam posts?
 For duplicates, see discussion above ““feature engineering/Not a duplicate”.
 Duplicates are an aspect of this problem statement and the method is already
 limited in the way I describe.  I think this approach would work well on spam
 posts, with the “title” analogous to email subject line, and the “body”
 analogous to the body of the email. Spam could potentially be imbalanced, and
 the class imbalance could be remedied with upsampling techniques, with care.
 
## Are there edge cases that your method tends to do worse with? Better? E.g.,
## How well does it handle really long posts or titles?
 Ive included the length of the concatenated body as a feature, so it should
  accommodate that.
 [Do ad hoc analysis on how well it did with really long posts though]

## If you got to work on this again, what would you do differently (if anything)?
 Throughout the document I have listed what I would do with more time. I would
 work on those things.

## If you found any issues with the dataset, what are they?
 There were two missing values because of misapplied parser. This suggests that
 there could be other issues at the parsing stage. However, applying my own
 parser did not lead to a significant difference in performance, as discussed
 above.
