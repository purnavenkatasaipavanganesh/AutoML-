# AutoML-
Identifies which model is suitable for the given supervised dataset

Supervised datasets refers to the data which is labeled.

**#Steps** 
1. Upload the data set
2. Select the type of data whether the data is classification or regression {(Classification deals with CLASSIFICATION values),(Regression deals with NUMERICAL values)}
3. Then check the data preview
4. Select the target column  
5. Press the train model button

**#Result** 
Gives the table of algorithms checked and their accuracies in descending order.
Suggests the best algorithm to use and its accuracy. 

**Flow of the model**
1.Loads the dataset
2.Preprocess the data set
3.Train on every model available
4.Calculates the evaluation metrics
5.Suggests the best model according to the metric values
