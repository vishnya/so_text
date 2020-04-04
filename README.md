

To install all the dependencies, clone the repo locally, `cd` to the root, and run `pip install -e .`.

# Link to problem statement

# Up-front caveats
I interpreted the problem statement to be emphasizing exploration and
analysis  over  performance fine-tuning and deployment. Thus I experimented
with approaches throughout the initial modeling process, employing various
preprocessing, feature engineering, and algorithm techniques, which
needed to be then partially represented in code in the repo. Some
approaches took too long to execute on my Macbook Air, such as the
breakdown of a document into its parts of speech to assess sentence
sophistication/complexity. Given more time to experiment, I would
launch a cloud cluster and parallelize the execution of such methods.
         
         
Because most of the below explores possible approaches, and the repo itself
is intended to just capture the code behind the experimentation, the repo
does not follow the usual standards of a python library. For instance, there
is dead code and code duplication.  Nevertheless, with more time I
would still organize the code into a coherent whole that is focused on
automating some experimentation to facilitate performance fine-tuning.
Specifically, I would have a config file that populates the hardcoded
hyperparameters with different combinations, applies different
algorithms specified in the config, and  returns the optimal result (based on a
performance metric).

# Full pipeline code
To address the implicit expectations of the problem statement up front, a run of
a modeling pipeline can be found here.
 
# Main analysis

## On text processing
When we apply various text parsing and processing techniques to training text
data, we change the data. Thus the overall model estimator trained on that is
different as well, and the model performance can change. There is a trade-off
 between scraping the text too much, so that important signal is lost, and
  scraping the text too little, so that there is denormalization, i.e. two
   words that should be treated as the same are not.
   
For example, the provided parsed data clearly removes some html. But some of
 StackOverflow’s articles involve html code, so that means some of that
  signal is lost. 
  
  We can compare the performance of the final pipeline on the unprocessed vs
   processed code:
   
```
Performance for unprocessed text
Performance for processed text
```
We can see that the processed text performs differently in way x. 
Let us compare the top `tf-idf` features:

```
insert that 
```

Next, 
 To negotiate that
tradeoff, we can play around with including different steps and seeing how
that affects the outcome that we are optimizing around. For example, can include
or not include html tags, or certain ones, and compare roc_auc.
Tried my own text parser but that lead to very similar results in the end in
terms of perf metrics so didn’t use it (See code) The performance of the
model across different classes depends on the text processing technique that is
used. For example, recall could go up for one class (=1) and down for another 
(=0) depending on the processing technique. To understand why, need to gain 
intuition by seeing examples of documents that were classified correctly then 
misclassified (and vice versa) under the different text processing techniques

## On feature engineering
Recall from the problem statement that standards for posts specify they be: “on topic, detailed, and not a duplicate of another question.”
### “On topic”:
Word similarity to accepted documents.  (Done with tf-idf vectorization in the pipeline. That represents a normalized frequency of words as a feature to the model. Given a new document, if the normalized frequency of the words is sufficiently similar to some existing data, it will be labelled similarly. That is, documents with relatively similar words and frequencies will be scored similarly.)
However, a too on topic document, as it has a too similar tf-idf encoding, can mean that it represents a duplicate. Deduplication can happen outside the predictive component of the model on test data/new data, or can be feature engineered. See below.
### “Detailed”:
Really what is meant here is complexity and variability of the text.
Length can be an indication -- this can go in as a feature. (Done -- )
Ratios of parts of speech could be an indication. This could also be related to “on topic”. (Link to code) Was too slow, am working on a macbook air, could parallelize this on a cluster.
### “Not a duplicate”:
Suppose we were given the full corpus and the model was trained on the full data, and retrained each time a new doc and label came in. (If there is class imbalance, we can attempt to remedy it with upsampling techniques, but ignore that detail for this discussion.) Then, suppose that a new document comes in and we want to score it using the model. If the previous training set included just 1 actual duplicate that is similar to the new document that was accepted (label 0), the model is more likely to predict that the new document would also be accepted, so on the new datum the model would be inaccurate.
On the other hand, if the training set already contained the original “accepted document” and the duplicates of it that were rejected, the model would be more likely to reject the new document, because it was similar to the duplicates that were rejected.
Now, in our case, there is a limited corpus (limited rows) for the training set. So the question becomes more muddied. Given this limited model, if a new document comes in, it may well be a duplicate, but because the training set hadn’t seen enough examples of the duplicate document and so had many of those labeled 0, the limited model is more likely to accept the new document.
Solution for duplications: create an initial layer of clustering in the new document coming in step, using cosine similarity of encoded vectors.  We can set a threshold, depending on business logic (say, cost of rejecting the post), that will auto-reject the document if it is too similar to existing ones:
https://towardsdatascience.com/de-duplicate-the-duplicate-records-from-scratch-f6e5ad9e79da
In terms of feature engineering for this model: Cluster texts based on similarity. Probably the threshold should be low, leaning towards more documents not being duplicates. If we did have the time at which the question was posted, the earliest document in the cluster, we could mark as 0, and the others we could mark as 1. Even in the absence of the time, what’s really important to the predictive model is just 1 of them is the OG, not which one, so we could just mark one as 1, and others as 0.
### Title and body
Strictly speaking, should encode these separately (and do tfidf) into sparse matrices, and then feed both sparse matrices separately into a model.
Why: There is signal from keeping subject lines and body separate
The vocabulary world of the subject lines and body are different and might have different standards
In practice, I tried vectorizing both separately , or just vectorizing the concatenated thing, and the roc_auc of the model i tried it with differed only slightly (Code)
Feature selection: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/

## On Algorithms
Naive Bayes: Assumption of independence across different words.
Including context/ordering (ngrams vs not) improves the performance -- but this still loses context (Naive Bayes pipeline code)
XGBoost
If include other kinds of features, especially numerical ones, it makes more sense to use an algorithm that allows for this. (xgboost pipeline code)
LSTM
Given more time, I’d try an LSTM on the encoded word features. The advantage of this would be to preserve the order. https://towardsdatascience.com/word-bags-vs-word-sequences-for-text-classification-e0222c21d2ec
Concatenating the subject/body vs. leaving them separate didn’t produce a significant difference in the roc_auc. (Link to code)

## On Evaluating performance
Overall, I chose to do both a classification report and an roc_auc. (Snippet from final pipeline)
The advantage of the classification report is that it breaks down how well the model is performing in different buckets for a particular threshold. However, this is misleading, because the classification report is just for one threshold. If we do not know what the threshold is going to be for our model, the classification report has limited significance. The threshold should be  optimized around the outcome one wants to achieve with the model. For example, if StackOverflow wants to use the model to automate the approval or declining of posts, it should calculate the cost of false positives and negatives, and determine the threshold for the classifier accordingly. (Threshold selection is normally done in such an ad-hoc fashion.)  Then the classification report will have more meaning.
 	Roc_auc can be a good measure of the model performance because it does not take into account the choice of threshold, which as explained is scientifically arbitrary. However, because the roc_auc is a curve between the TPR and the FPR, in the case of highly imbalanced data, it does not indicate how well the model performs on minority classes. Precision-recall curves can be better in this situation (see http://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf). Therefore, if the training dataset provided to me was sampled from a highly imbalanced dataset, I would perhaps look to the AUCPR.

# Questions from the problem statement:

## What metric did you use to judge your approach’s performance, and how did it perform? Why did you choose that metric?
See “On Evaluating Performance” above.
											 		## The dataset we’ve given you is artificially balanced such that there’s an even split of closed posts to accepted posts. Should this influence the metrics you measure?

The metrics tell us how well the model performs on the held out test set, which, since it has been split from the artificially balanced original set, is itself artificially balanced. If in fact, the original data is highly skewed (more accepted than closed, or vice versa) this model could actually be ineffective on unseen data that is not in this artificially balanced distribution. See “On Evaluating Performance” above for more details.

## How generalizable is your method? If you were given a different (disjoint) random sample of posts with the same labeling scheme, would you expect it to perform well? Why or why not? Do you have evidence for your reasoning?

Since the data seems to be “not too small” for the approach chosen, it would potentially suffer from the same limitations previously, but if it was randomly chosen, I think it’s unlikely it would introduce new limitations. One limitation that stands is that with duplicates-- see ““feature engineering/Not a duplicate”. In short, for duplicates, a different random sample will potentially be missing an original post (the original non-duplicated post which others can duplicate), so that given a new post, depending on how one handles duplicates, it can be treated as the original, whereas it is a duplicate.

## How well would this method work on an entirely new close reason, e.g. duplicate or spam posts?
 For duplicates, see discussion above ““feature engineering/Not a duplicate”. Duplicates are an aspect of this problem statement and the method is already limited in the way I describe.  I think this approach would work well on spam posts, with the “title” analogous to email subject line, and the “body” analogous to the body of the email. Spam could potentially be imbalanced, and the class imbalance could be remedied with upsampling techniques, with care.
## Are there edge cases that your method tends to do worse with? Better? E.g., How well does it handle really long posts or titles?
Ive included the length of the concatenated body as a feature, so it should accommodate that.
[Do ad hoc analysis on how well it did with really long posts though]
## If you got to work on this again, what would you do differently (if anything)? 			Throughout this analysis I listed what I would try with more time.

## If you found any issues with the dataset, what are they?
 There were two missing values because of misapplied parser. This suggests that there could be other issues at the parsing stage. However, applying my own parser did not lead to a significant difference in performance.
