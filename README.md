# Neural_Network_Charity_Analysis
## Purpose
 Create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.
 
## Deliverables:
* Deliverable 1: Preprocessing Data for a Neural Network Model
* Deliverable 2: Compile, Train, and Evaluate the Model
* Deliverable 3: Optimize the Model
* Deliverable 4: A Written Report on the Neural Network Model (README.md)

 ## Data
Customer provided a CSV file of past funding efforts(charity_data.csv)

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

 ### Deliverable 1: Preprocessing Data for a Neural Network Model
1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
    * What variable(s) are considered the target(s) for your model?
        - APPLICATION_TYPE        34299 non-null  object
        - AFFILIATION             34299 non-null  object
        - CLASSIFICATION          34299 non-null  object
        - USE_CASE                34299 non-null  object
        - ORGANIZATION            34299 non-null  object
        - STATUS                  34299 non-null  int64 
        - INCOME_AMT              34299 non-null  object
        - SPECIAL_CONSIDERATIONS  34299 non-null  object
        - ASK_AMT                 34299 non-null  int64 

    * What variable(s) are considered the feature(s) for your model?
        - IS_SUCCESSFUL—Was the money used effectively
2. Drop the EIN and NAME columns.
![pic](./Images/drop_EIN_NAME.png)
3. Determine the number of unique values for each column.
![pic](./Images/Unq_col_values.png)
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
![pic](./Images/Data_Point_Per_Unq_Val.png)
5. Create a density plot to determine the distribution of the column values.
![pic](./Images/Density_1.png)
6. Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.
![pic](./Images/otherLT100.png)
7. Generate a list of categorical variables.
![pic](./Images/encodedVar.png)
8. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
![pic](./Images/encodedVar.png)
9. Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
![pic](./Images/mergeenc.png)
10. Split the preprocessed data into features and target arrays.
![pic](./Images/splittest.png)
11. Split the preprocessed data into training and testing datasets.
![pic](./Images/splittest.png)
12. Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.
![pic](./Images/scaler.png)


### Deliverable 2: Compile, Train, and Evaluate the Model
1. Continue using the AlphabetSoupCharity.ipynb file where you’ve already performed the preprocessing steps from Deliverable 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
![pic](./Images/nnmodel.png.png)
3. Create the first hidden layer and choose an appropriate activation function.
![pic](./Images/1stlayer.png)
If necessary, add a second hidden layer with an appropriate activation function.
![pic](./Images/2ndlayer.png)
4. Create an output layer with an appropriate activation function.
![pic](./Images/outlayer.png)
5. Check the structure of the model.
![pic](./Images/checkstructureofmodel.png)
6. Compile and train the model.
![pic](./Images/compliletrain.png)
![pic](./Images/train.png)
7. Create a callback that saves the model's weights every 5 epochs.
![pic](./Images/callback.png)
8. Evaluate the model using the test data to determine the loss and accuracy.
![pic](./Images/evalthemodel.png
9. Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.
![pic](./Images/HDF5.png)
10. Save your AlphabetSoupCharity.ipynb file and AlphabetSoupCharity.h5 file to your Neural_Network_Charity_Analysis folder and Trained_Models respectively.
![pic](./Images/HDF5.png)



### Deliverable 3: Optimize the Model