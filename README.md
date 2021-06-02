# Classification-1
1. The Data:
Read in the bank telemarketing data located here: CSV download link
Randomly split 80% of the data into the training set and the remainder 20% into the test set. Use set.seed(1) so that I can replicate your results.
Check the distribution of the outcome variable (“y”). What proportion are “yes” and “no”? Given this info, what would be a good baseline performance to compare our models?

2. Decision Tree:
Build a decision tree on the training set using all the predictors with “y” as the class variable. Plot the tree. What do we notice about the terminal nodes? Why might this be the case?
Use 10-fold cross-validation to find the optimal tree size. What tree size did you pick and why? Prune the tree and plot it again. Provide your interpretation of the new tree.
Using your pruned tree, make predictions on the test set. Show the confusion matrix. Calculate accuracy, sensitivity, and precision. Discuss why the model is useful or not.


3. KNN:

3A. Data preparation
Check the min and max values for the continuous variables to see if they are on even scale. Normalize them to 0-1 scale if needed.
Convert the categorical variables into dummy variables.

3B. Model building & comparison
Use 5-fold cross validation on the training set to compare performance between KNN where K=3 and KNN where K=5 (make sure you run set.seed(1) so I can replicate your results). Compare the accuracy, sensitivity of “yes”, and precision of “yes”.
Let’s assume the cost of a marketing call is high and promoting to uninterested clients will result in annoyed customers. In this sense, false positives would be costly. Which would be the more optimal model? Explain in terms of sensitivity/precision and/or FP/FN. 

3C. Predictions
Select either the K=3 or K=5 KNN model based on the previous question and make predictions on the test set. Evaluate its performance on unseen data.
