# Machine Learning and Statistical Data Analysis - Final Project
## Goal
Our goal was to create a machine learning model that, give a set of reviews for an Amazon product, could predict whether or not the product was "Awesome". "Awesome" was defined by the product having an average rating of greater than 4.4 stars.
## Models
We used three models, Multinomial Naive Bayes, Gradient Boosting Classifier, and AdaBoost. We then fed all of these models into a Support Vector Machine, which took all of the models predictions and generated a final outcome.
## Data Pre-Processing
We elected to feed certain aspects of the review data to certain models, depending on what the model was best suited for. For the models that used text data, we turned the review text into a bag of words to help the learning process. We also included sentiment scores in the final SVM, which were generated using the VaderSentiment library. We used the 25th, 50th, and 75th percentile of the sentiment scores.
## Outcome
Our final SVM had an average f1 score of 0.84, which was more than satisfactory for the assignment.
## Authors 
Scott Crawshaw, Alex Bakos, Sam Wang, Daniel Akili, and Jeff Cho