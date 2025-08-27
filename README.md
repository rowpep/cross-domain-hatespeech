# Hate-speech cross-domain detection 

The goal of this code is to evaluate various hate speech classifiers in both zero shot and few shot conditions and how they generalise to different social media platforms. 

The pipeline of the study:

Pre-processing the data, standardising formats, and tokenising data with spaCy - data_preprocessing.ipynb

Model training, testing and evaluation for baseline models(kNN, SVM, RF), cnn, rnn, logistic regression, and BERT- baseline models knn rf svm.py, logistic regression.py, cnn.py, rnn.py, BERT.py

The statistical testing was performed in two sets. 
One testing the first research question: how models perform from in-domain to cross domain settings - per_model_domain_stats.py
The second research question: how does few-shot learning affect the generalisability of hate speech classifiers - per_model_fewshot_stats.py

The datasets used are:
Davidson et al (2017)'s Twitter hate speech dataset
HateXplain
Wikipedia Detox
Reddit Slur Corpus
Gab Hate Corpus
All contain offesnive language.

Labels are mapped 1 = hate speech and 0 = non-hate speech

The random seed used is 42 in scripts. 

