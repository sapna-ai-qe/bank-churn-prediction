# <center><font size=6> Bank Churn Prediction </font></center>

# Problem Statement

# Context

# Businesses like banks which provide service have to worry about problem of 'Customer Churn' i.e. customers leaving and j

# Objective

# You as a Data scientist with the  bank need to  build a neural network based classifier that can determine whether a cus

# Data Dictionary

# * CustomerId: Unique ID which is assigned to each customer
# * Surname: Last name of the customer
# * CreditScore: It defines the credit history of the customer.
# * Geography: A customer’s location
# * Gender: It defines the Gender of the customer
# * Age: Age of the customer
# * Tenure: Number of years for which the customer has been with the bank
# * NumOfProducts: refers to the number of products that a customer has purchased through the bank.
# * Balance: Account balance
# * HasCrCard: It is a categorical variable which decides whether the customer has credit card or not.
# * EstimatedSalary: Estimated salary
# * isActiveMember: Is is a categorical variable which decides whether the customer is active member of the bank or not ( 
# * Exited : whether or not the customer left the bank within six month. It can take two values
# ** 0=No ( Customer did not leave the bank )
# ** 1=Yes ( Customer left the bank )

# Installing the libraries with the specified version.
!pip install numpy==1.25.2 pandas==1.5.3 scikit-learn==1.5.2 matplotlib==3.7.1 seaborn==0.13.1 xgboost==2.0.3 -q --user

# Importing necessary libraries

# Library for data manipulation and analysis.
import pandas as pd
# Fundamental package for scientific computing.
import numpy as np
#splitting datasets into training and testing sets.
from sklearn.model_selection import train_test_split
#Imports tools for data preprocessing including label encoding, one-hot encoding, and standard scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
#Imports a class for imputing missing values in datasets.
from sklearn.impute import SimpleImputer
#Imports the Matplotlib library for creating visualizations.
import matplotlib.pyplot as plt
# Imports the Seaborn library for statistical data visualization.
import seaborn as sns
# Time related functions.
import time
#Imports functions for evaluating the performance of machine learning models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report

# To tune model, get different metric scores, and split data
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


#Imports the tensorflow,keras and layers.
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout,BatchNormalization
from tensorflow.keras import backend

# to suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# **Observation:**
# All the libraries are imported successfully

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
tf.keras.utils.set_random_seed(812)

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
tf.config.experimental.enable_op_determinism()

# Loading the dataset

# from google.colab import drive  # Remove if not using Colab
# drive.mount('/content/drive') #Mount the drive  # Remove if not using Colab

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/bank-1.csv')  #Read a csv file located at the path
bankdata = df.copy()  #Create a copy of the Dataframe df and assign it to a variable
bankdata #Displays the dataframe

# **Observation:**
# Dataframe is displayed

# Data Overview

bankdata.head(10) #Displays first ten rows of the dataframe

# **Observation:**
# Display the first 10 rows of the dataframe

bankdata.tail(10) #Displays last ten rows of the dataframe

# **Observation:**
# Display the last 10 rows of the dataframe

bankdata.shape #shape will return rows & column of data frame

# **Observation:**
# Total number of rows & column present in dataframe are :
# 1. 10000 rows
# 2. 14 column

bankdata.info() #Info method, provide quick overview of the dataframe

# **Observation:**
# Following observation are made using data.info
# 1. There are total of 10000 rows and 14 column in the dataframe
# 2. There are no missing value present
# 3. Only three columns (Surname, Geography, Gender) have datatype object
# 4. Only two column (Balance, EstimatedSalary) have datatype float64
# 5. Rest all columns (RowNumber, CustomerId, CreditScore, Age, Tenure, NumOfProducts, HasCrCard, IsActiveMember, Exited) 
# 6. Memory usage is : 1.1+ MB

bankdata.isnull().sum() #sum of null values per column

# **Observations:**
# Based on the output of the method, the data frame does not contain any missing values. The dataset is complete, with no 

bankdata.duplicated().sum() #Used to find and count the number of duplicate rows in the Dataframe

# **Observation:**
# Based on the output of the method, the data frame does not contain any duplicate values.

bankdata.describe(include= 'all').T # used to generate descriptive statistics of the dataframe (.T will transpose the dataframe, swapping rows and columns)

# **Observation:**
# By using describe function we can conlcude that:
# 1. Rownumber, CustomerId, are having unique value. There are 2932 unique surname.
# 2. Mean Creditscore is 650, median is around 652. which seems that data is not heavily skewed. Max Creditscore is 850.
# 3. There are 3 unique Geography out of which France is on the Top.
# 4. There are only two Gender i.e Male and Female out of which Male is on the Top.
# 5. Mean Age is 38, median age is 37, it also seems that data is not heavely skewed. As max age is aorund 92 which implie
# 6. Mean Tenure is 5, median is aso 5. and max tenure is 10 it also implies that data is not heavily skewed.
# 7. Mean balance is 76485, median is 97198, it seems like data is left skewed. Max balance is 250898 which implies that t
# 8. Mean Numberofproducts is 1.5 , median is 1 which means data is right skewed but not heavely skewed.
# 9. Mean Hascrcard is 0.7, median is 1. which implies that data is left skewed but not heavely skewed.
# 10. Mean IsActivemember is 0.5, median is 1. which means data is left skewed but not heavely skewed.
# 11. Mean Estimatedsalary is 100090, median is 100193. which implies that data is left skewed and also max values is 1999
# 12. Exited mean value is 0.2, median is 0.0 and max value is 1. which shows that data is right skewed but not heavely sk

# **Checking the count of each unique value in each column**

bankdata.nunique() #Used to find the number of unique values in each column

# **Observation:**
# 1. Rownumber has a total of 10000 unique value.
# 2. CustomerId  has a total of 10000 unique value.
# 3. Surname  has a total of 2932 unique value.
# 4. Creditscore  has a total of 460 unique value.
# 5. Geography  has a total of 3 unique value.
# 6. Gender has a total of 2 unique value.
# 7. Age has a total of 70 unique value.
# 8. Tenure has a total of 11 unique value.
# 9. Balance has a total of 6382 unique value.
# 10. Numberofproducts has a total of 4 unique value.
# 11. Hascrcard has a total of 2 unique value.
# 12. IsActiveMember has a total of 2 unique value.
# 13. Estimatedsalary has a total of 9999 unique value.
# 14. Exited has a total of 2 unique value.

data = bankdata.drop(labels = ['RowNumber', 'CustomerId', 'Surname'], axis=1)  #Drop the column

# **Observation:**
# Here we are dropping "RowNumber" , "CustomerId", "Surname" column as it doesnot give any valuable information.

data

# **Observation:**
# Displaying the data after removing the column.

data['Exited'].value_counts(1)

# **Observation:**
# 1. Customer who will not churn is around ~79%
# 2. Customer who is going to churn is around ~20%

# Exploratory Data Analysis

# Univariate Analysis

def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

### function to plot distributions wrt target


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()

# **Observation on CreditScore**

histogram_boxplot(data, "CreditScore") #create and display a histogram and boxplot

# Observation:
# 1. From the above boxplot, it seems like data is not heavily skewed, seems like data is identical. But there are few out
# 2. From the above histplot, it seems like mostly the creditscore lies between 400 to 800. There are few creditscore lowe

# **Observation on Age**

histogram_boxplot(data, "Age") #create and display a histogram and boxplot

# **Observation:**
# 1. From the above Boxplot, it can be concluded that data is slightly right skewed. There are a large number of outliers 
# 2. From the Histplot, we can conclude that, maximun individual falls in the age range of 25 to 50. Few are below 25 and 

# **Observation on Tenure:**

histogram_boxplot(data, "Tenure") #create and display a histogram and boxplot

# **Observation:**
# 1. From the above graph, it seems like data is symmertical . There is no skewness. And there is not any presence of Outl

# **Observation on Balance:**

histogram_boxplot(data, "Balance") #create and display a histogram and boxplot

# **Observation:**
# 1. From the above Boxplot, it seems like data is heavely left skewed.
# 2. From the above Histplot, it seems like most of the people has 0 balance. Few individual has balance in the range of 0

# **Observation on NumOfProducts:**

histogram_boxplot(data, "NumOfProducts") #create and display a histogram and boxplot

# **Observation:**
# 1. From the above boxplot, it seems like data is rightly skewed.
# 2. From the Histplot, it seems like most of the people has 1 or 2 products, only a few has 3 or 4 more products.

# **Observation on HasCrCard:**

histogram_boxplot(data, "HasCrCard") #create and display a histogram and boxplot

# **Observation:**
# 1. From the Boxplot, it seems like data is heavily left skewed.
# 2. From the Histplot, it seems like, few like doesnot have creditcard while majority has creditcard.

# **Observation on IsActiveMember:**

histogram_boxplot(data, "IsActiveMember") #create and display a histogram and boxplot

# **Observation:**
# 1. From the graph above, it seems like data is not heavily skewed. It seems balanced. There are only minor difference be

# **Observation on EstimatedSalary:**

histogram_boxplot(data, "EstimatedSalary") #create and display a histogram and boxplot

# **Observation:**
# 1. From the above graph, nothing much can be predicted. Graph is uniformly distributed. There seems no outliers.

# **Observation on Exited:**

histogram_boxplot(data, "Exited") #create and display a histogram and boxplot

# **Observation:**
# 1. From the above graph, it seems like majority of the customer did not leave the bank but there are few who will leave.

# **Observation on Geography:**

labeled_barplot(data, "Geography", perc=True) #Display labeled barplot

# **Observation:**
# From the above graph we can conclude that
# 1. Majority of the individual is from France with 50.1%
# 2. Second is from Germay with 25.1%
# 3. Third comes Sapin with 24.8%

# **Observation on Gender:**

labeled_barplot(data, "Gender", perc=True) #Display labeled barplot

# **Observation:**
# From the above graph we can conclude that:
# 1. 54.6% of the individual is Male.
# 2. 45.4% is Female.

# Bivariate Analysis

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

# **Heat-Map**

# seperate the numerical values
cols_list = data.select_dtypes(include=np.number).columns.tolist()

# create the correlation matrix
plt.figure(figsize=(12, 7))
sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()

# **Observation:**
# From the above heat map we can conclude that:
# 1. Age and Exited have a moderate positive correlation (0.29). This suggests that older customers may be more likely to 
# 2. Balance and Exited have a weak positive correlation (0.12). This indicates that customers with higher balances might 
# 3. IsActiveMember and Exited have a moderate negative correlation (-0.16). This implies that active members are less lik
# 4. Balance and NumberofProducts have a moderate negative correlation(-0.30).

# **Observation on Exited vs Geography:**

stacked_barplot(data, "Geography", "Exited") #create and display a stacked bar plot

# **Observation:**
# From the above graph we can conclude that:
# 1. More people exited from Germay.
# 2. Spain and France has almost similar exited pattern.

# **Observation on Exited vs Gender:**

stacked_barplot(data, "Gender", "Exited") #create and display a stacked bar plot

# **Observation:**
# From the above graph we can conclude that:
# 1. Female population will churn more than the male.

# **Observation on Exited vs NumOfProducts:**

stacked_barplot(data, "NumOfProducts", "Exited") #create and display a stacked bar plot

# **Observation:**
# From the above graph we can conclude that:
# 1. Customers with 4 products are most likely to churn
# 2. Customers with 3 products are also at risk of churn: While not as high as those with 4 products
# 3. Customers with 2 products are less likely to churn than those with 1 product

# **Observation on Exited vs HasCrCard:**

stacked_barplot(data, "HasCrCard", "Exited") #create and display a stacked bar plot

# **Observation:**
# From the above graph it implies that the customer who has credit card and who doesnot have credit card has similar churn

# **Observation on Exited vs IsActiveMember:**

stacked_barplot(data, "IsActiveMember", "Exited") #create and display a stacked bar plot

# **Observation:**
# From the above graph we can conclude that:
# 1. Customer who is not an Active member is more likely to churn than the one who is an Active member.

# **Observation on Exited vs CreditScore:**

distribution_plot_wrt_target(data, "CreditScore", "Exited") #Create and visualize plot like histogram , boxplot

# **Observation:**
# From the above graph we can concldue that creditscore is not the potenetial reason for customer churning.

# **Observation on Exited vs Age:**

distribution_plot_wrt_target(data, "Age", "Exited") #Create and visualize plot like histogram , boxplot

# **Observation:**
# From the above graph we can conclude that customer who are older is age is more likely to churn .

# **Observation on Exited vs Tenure:**

distribution_plot_wrt_target(data, "Tenure", "Exited") #Create and visualize plot like histogram , boxplot

# **Observation:**
# From the above histplot & Boxplot, the distribution of Tenure is relatively similar for customer who churned and who did

# **Observation on Exited vs Balance:**

distribution_plot_wrt_target(data, "Balance", "Exited") #Create and visualize plot like histogram , boxplot

# **Observation:**
# From the above graph we can conclude that customer having higher balance is more likely to churn

# **Observation on Exited vs EstimatedSalary:**

distribution_plot_wrt_target(data, "EstimatedSalary", "Exited") #Create and visualize plot like histogram , boxplot

# **Observation:**
# From the above graph it seems like Estimated Salary does not seem to affect the customer churn much

# Data Preprocessing

# Dummy Variable Creation

#Converting Categorical feature into Numerical using one hot encoding
datad = pd.get_dummies(data,columns=data.select_dtypes(include=["object"]).columns.tolist(),drop_first=True)
datad = datad.astype(float)
datad.head()

# Train-validation-test Split

X = datad.drop(['Exited'],axis=1)
y = datad['Exited']

# Splitting the dataset into the Training and Test set.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42,stratify = y)

# Splitting the Train dataset into the Training and Validation set.
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42,stratify = y_train)

#Printing the shapes.
print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(X_test.shape,y_test.shape)

# **Observation:**
# 1. Shape of Training set is 6400.
# 2. Shape of validation set is 1600.
# 3. Shape of Testing set is 2000.

# Data Normalization

# creating an instance of the standard scaler
sc = StandardScaler()

cols_list.remove('Exited')

X_train[cols_list] = sc.fit_transform(X_train[cols_list])
X_val[cols_list] = sc.transform(X_val[cols_list])
X_test[cols_list] = sc.transform(X_test[cols_list])

# **Utility function**

def plot(history, name):
    """
    Function to plot loss/accuracy

    history: an object which stores the metrics and losses.
    name: can be one of Loss or Accuracy
    """
    fig, ax = plt.subplots() #Creating a subplot with figure and axes.
    plt.plot(history.history[name]) #Plotting the train accuracy or train loss
    plt.plot(history.history['val_'+name]) #Plotting the validation accuracy or validation loss

    plt.title('Model ' + name.capitalize()) #Defining the title of the plot.
    plt.ylabel(name.capitalize()) #Capitalizing the first letter.
    plt.xlabel('Epoch') #Defining the label for the x-axis.
    fig.legend(['Train', 'Validation'], loc="outside right upper") #Defining the legend, loc controls the position of the legend.

# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred = model.predict(predictors) > threshold
    # pred_temp = model.predict(predictors) > threshold
    # # rounding off the above values to get classes
    # pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred, average='weighted')  # to compute Recall
    precision = precision_score(target, pred, average='weighted')  # to compute Precision
    f1 = f1_score(target, pred, average='weighted')  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1 Score": f1,},
        index=[0],
    )

    return df_perf

# Model Building

# Model Evaluation Criterion

# Write down the logic for choosing the metric that would be the best metric for this business scenario.
# -The primary goal for bank is the "customer churn"  customers leaving and joining another service provider.Customer chur
# **As per the objectives, below are the metrices which are relevant.**
# 1. **Accuracy** (This measures the overall correctness of the predictions.)
# 2. **Precision** (This focuses on how many of the predicted positive cases are actually correct).
# 3. **Recall** (This focuses on how many of the actual positive cases are correctly identified).
# 4. **f1-score** (This is the harmonic mean of precision and recall, balancing the trade-off between the two.)
# **Model can make wrong predictions as:**
# 1. **False Positive**: The model predicts that a customer will churn (leave), but the customer doesn't actually churn.
# 2. **False Negative**: The model predicts that a customer will not churn (stay), but the customer actually churns..
# **Which case is more important?**
# **False Negatives**: Missing customers who are at risk of churning means you lose valuable clients, which can directly i
# **How to reduce this loss i.e need to reduce False Negatives?**
# For this problem, where the emphasis is likely on customer churning, recall is the more critical metric. We need to Prio
# Hence **Recall score** is the best choice for us in case of customer churn problem.

# As we are dealing with class imbalance problem , Hence we need to defne class weight

# Calculate class weights for imbalanced dataset
cw = (y_train.shape[0]) / np.bincount(y_train)

# Create a dictionary mapping class indices to their respective class weights
cw_dict = {}
for i in range(cw.shape[0]):
    cw_dict[i] = cw[i]

cw_dict

# Neural Network with SGD Optimizer

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential() #initialize a Sequentail neural network model
model.add(Dense(64,activation="relu",input_dim=X_train.shape[1])) #add Input layer to the model with 64 neurons and relu as activation function
model.add(Dense(32,activation="relu")) #add hidden layer to the model with 32 neurons and relu as activation function
model.add(Dense(1,activation="sigmoid"))  #add output layer to the model with sigmoid as activation function

model.summary() #Print concise summary of the defined neural network architecture

# **Observation:**
# 1. Layer display each layer in the model
# 2. Output shape, shows the dimension of output data after passing through that layer
# 3. Number of parameter, calculates and present total trainable and no-trainable parameter in each layer and entire model
# 4. Total parameter is 2881, Trainable parameter is 2881 and Non-trainable parameter is 0.

optimizer = tf.keras.optimizers.SGD(0.001)    # defining SGD  optimizer and a learning rate of 0.001
model.compile(loss='binary_crossentropy', optimizer=optimizer) #function is used to configure the learning process of the model before training begins.

# **Observation:**
# 1. Here we are using SGD as the optimizer

epochs = 25 #model will see and learn from the complete training data 25 times
batch_size = 64 #Model will take 64 training example, calculate error and update is weight based on error.

start = time.time() #record current time and store in a variable
history = model.fit(X_train, y_train, validation_data=(X_val,y_val) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict) #train model with specified parameter.
end=time.time() #record the time after the training is finished in a variable.

print("Time taken in seconds ",end-start) #prinitng the value

plot(history,'loss') #Plot the training and validation loss over the training epochs

model_0_train_perf = model_performance_classification(model, X_train, y_train) #calculate the performce of trained neural network model on training data
model_0_train_perf

model_0_val_perf = model_performance_classification(model, X_val, y_val) #calculate the performce of trained neural network model on validation data
model_0_val_perf

# **Observation:**
# With SGD Optimizer, we are getting a training Recall score of ~0.69 and validation Recall score of ~0.66.

# Model Performance Improvement

# Neural Network with Adam Optimizer

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential() #initialize a Sequentail neural network model
model.add(Dense(64,activation="relu",input_dim=X_train.shape[1])) #add input layer to the model with 64 neurons and relu as activation function
model.add(Dense(32,activation="relu")) #add  hidden layer to the model with 32 neurons and relu as activation function
model.add(Dense(1,activation="sigmoid")) #add output layer to the model with sigmiod as activation function

model.summary() #Print concise summary of the defined neural network architecture

# **Observation:**
# 1. Layer display each layer in the model
# 2. Output shape, shows the dimension of output data after passing through that layer
# 3. Number of parameter, calculates and present total trainable and no-trainable parameter in each layer and entire model
# 4. Total parameter is 2881, Trainable parameter is 2881 and Non-trainable parameter is 0.

optimizer = tf.keras.optimizers.Adam(0.001)    # defining Adam as the optimizer and learning rate of 0.001
model.compile(loss='binary_crossentropy', optimizer=optimizer) #function is used to configure the learning process of the model before training begins.

# **Observation:**
# 1. Here we are using Adam as the optimizer

epochs = 25 #model will see and learn from the complete training data 25 times
batch_size = 64 #Model will take 64 training example, calculate error and update is weight based on error.

start = time.time() #record current time and store in a variable
history = model.fit(X_train, y_train, validation_data=(X_val,y_val) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict) #train model with specified parameter.
end=time.time() #record end time and store in a variable

print("Time taken in seconds ",end-start) #printing the value

plot(history,'loss') #Plot the training and validation loss over the training epochs

model_1_train_perf = model_performance_classification(model, X_train, y_train) #calculate the performce of trained neural network model on training data
model_1_train_perf

model_1_val_perf = model_performance_classification(model, X_val, y_val) #calculate the performce of trained neural network model on validation data
model_1_val_perf

# **Observation:**
# With Adam Optimizer, Recall score is little bit better, we are getting a training Recall score of ~0.81 and validation R

# Neural Network with Adam Optimizer and Dropout

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
#input layer with 32 neurons and relu as activation function
model.add(Dense(32,activation='relu',input_dim = X_train.shape[1]))
# adding dropout of 0.2
model.add(Dropout(0.2))
# Hidden layer with 64 neurons and relu as activation function
model.add(Dense(64,activation='relu'))
# Hidden layer with 32 neurons and relu as activation function
model.add(Dense(32,activation='relu'))
# adding dropout of 0.1
model.add(Dropout(0.1))
# Hidden layer with 16 neurons and relu as activation function
model.add(Dense(16,activation='relu'))
# Output layer with sigmoid as activation function
model.add(Dense(1, activation = 'sigmoid'))

model.summary() #Print concise summary of the defined neural network architecture

# **Observation:**
# 1. Layer display each layer in the model
# 2. Output shape, shows the dimension of output data after passing through that layer
# 3. Number of parameter, calculates and present total trainable and no-trainable parameter in each layer and entire model
# 4. Total parameter is 5121, Trainable parameter is 5121 and Non-trainable parameter is 0.

optimizer = tf.keras.optimizers.Adam(0.001)    # defining Adam as the optimizer and learning rate of 0.001
model.compile(loss='binary_crossentropy', optimizer=optimizer) #function is used to configure the learning process of the model before training begins.

epochs = 25 #model will see and learn from the complete training data 25 times
batch_size = 64 #Model will take 64 training example, calculate error and update is weight based on error.

start = time.time() #record current time and store in a variable
history = model.fit(X_train, y_train, validation_data=(X_val,y_val) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict) #train model with specified parameter.
end=time.time() #record end time and store in a variable

print("Time taken in seconds ",end-start) #printing the value

plot(history,'loss') #Plot the training and validation loss over the training epochs

model_2_train_perf = model_performance_classification(model, X_train, y_train) #calculate the performce of trained neural network model on training data
model_2_train_perf

model_2_val_perf = model_performance_classification(model, X_val, y_val)  #calculate the performce of trained neural network model on validation data
model_2_val_perf

# **Observation:**
# With Adam Optimizer & Drop-out, Recall score improves bit by ~0.81, and a validation Recall score is ~0.80

# Neural Network with Balanced Data (by applying SMOTE) and SGD Optimizer

#Import the SMOTE class
from imblearn.over_sampling import SMOTE

# Create an instance of the SMOTE class
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Apply SMOTE to the validation data
X_val_smote, y_val_smote = smote.fit_resample(X_val, y_val)


print('After Applying SMOTE, the shape of X_train: {}'.format(X_train_smote.shape))
print('After Applying SMOTE, the shape of y_train: {}'.format(y_train_smote.shape))




# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()



# Initializing the neural network
model = Sequential() #initialize a Sequentail neural network model
model.add(Dense(64, activation="relu", input_dim=X_train_smote.shape[1])) #add input layer to the model with 64 neurons and relu as activation function
model.add(Dense(32, activation="relu")) #add hidden layer to the model with 32 neurons and relu as activation function
model.add(Dense(16, activation="relu")) #add hidden layer to the model with 16 neurons and relu as activation function
model.add(Dense(1, activation="sigmoid")) #add output layer with sigmoid as activation function

model.summary() #Print concise summary of the defined neural network architecture

# **Observation:**
# 1. Layer display each layer in the model
# 2. Output shape, shows the dimension of output data after passing through that layer
# 3. Number of parameter, calculates and present total trainable and no-trainable parameter in each layer and entire model
# 4. Total parameter is 3393, Trainable parameter is 3393 and Non-trainable parameter is 0.

# Defining SGD as the optimizer
optimizer = tf.keras.optimizers.SGD(0.001) #adding SGD as optimizer and learning rate as 0.001
model.compile(loss='binary_crossentropy', optimizer=optimizer) #function is used to configure the learning process of the model before training begins.

epochs = 25 #model will see and learn from the complete training data 25 times
batch_size = 64 #Model will take 64 training example, calculate error and update is weight based on error.

start = time.time() #record current time and store in a variable
history = model.fit(X_train_smote, y_train_smote, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)  #train model with specified parameter.
end = time.time() #record end time and store in a variable

print("Time taken in seconds ",end-start)

plot(history,'loss') #Plot the training and validation loss over the training epochs

model_3_train_perf = model_performance_classification(model, X_train_smote, y_train_smote) #calculate the performce of trained neural network model on training data
model_3_train_perf

model_3_val_perf = model_performance_classification(model, X_val_smote, y_val_smote) #calculate the performce of trained neural network model on validation data
model_3_val_perf

# **Observation:**
# By applying SMOTE & SGD optimizer, our Recall score again decreases. Training Recall score is ~0.67 and validation Recal

# Neural Network with Balanced Data (by applying SMOTE) and Adam Optimizer

#Import the SMOTE class
from imblearn.over_sampling import SMOTE

# Create an instance of the SMOTE class
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Apply SMOTE to the validation data
X_val_smote, y_val_smote = smote.fit_resample(X_val, y_val)

print('After Applying SMOTE, the shape of X_train: {}'.format(X_train_smote.shape))
print('After Applying SMOTE, the shape of y_train: {}'.format(y_train_smote.shape))


# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

model = Sequential() #initialize a Sequentail neural network model
model.add(Dense(64, activation="relu", input_dim=X_train_smote.shape[1])) #add input layer to the model with 64 neurons and relu as activatin function
model.add(Dense(32, activation="relu")) #add hidden layer to the model with 32 neurons and relu as activation function
model.add(Dense(16, activation="relu")) #add hidden layer to the model with 16 neurons and relu as activation function
model.add(Dense(1, activation="sigmoid")) #add output layer with sigmoid as activation function

model.summary() #Print concise summary of the defined neural network architecture

# **Observation:**
# 1. Layer display each layer in the model
# 2. Output shape, shows the dimension of output data after passing through that layer
# 3. Number of parameter, calculates and present total trainable and no-trainable parameter in each layer and entire model
# 4. Total parameter is 3393, Trainable parameter is 3393 and Non-trainable parameter is 0.

optimizer = tf.keras.optimizers.Adam(0.001) # defining Adam optimizer and learning rate as 0.001
model.compile(loss='binary_crossentropy', optimizer=optimizer) #function is used to configure the learning process of the model before training begins.

epochs = 25 #model will see and learn from the complete training data 25 times
batch_size = 64 #Model will take 64 training example, calculate error and update is weight based on error.

start = time.time() #record current time and store in a variable
history = model.fit(X_train_smote, y_train_smote, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)  #train model with specified parameter.
end = time.time() #record end time and store in a variable

print("Time taken in seconds ",end-start)

plot(history,'loss') #Plot the training and validation loss over the training epochs

model_4_train_perf = model_performance_classification(model, X_train_smote, y_train_smote) #calculate the performce of trained neural network model on training data
model_4_train_perf

model_4_val_perf = model_performance_classification(model, X_val_smote, y_val_smote) #calculate the performce of trained neural network model on validation data
model_4_val_perf

# **Observation:**
# By Applying SMOTE and Adam optimizer, Recall score improves little bit. Training Recall score is ~0.88 and validation Re

# Neural Network with Balanced Data (by applying SMOTE), Adam Optimizer, and Dropout

#Import the SMOTE class
from imblearn.over_sampling import SMOTE

# Create an instance of the SMOTE class
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Apply SMOTE to the validation data
X_val_smote, y_val_smote = smote.fit_resample(X_val, y_val)

print('After Applying SMOTE, the shape of X_train: {}'.format(X_train_smote.shape))
print('After Applying SMOTE, the shape of y_train: {}'.format(y_train_smote.shape))


# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the model
model = Sequential()
# Add input layer with relu as activation function and 32 neurons
model.add(Dense(32,activation='relu',input_dim = X_train_smote.shape[1]))
#Add dropout rate
model.add(Dropout(0.2))
# Add hidden layer with relu as activation function and 16 neurons
model.add(Dense(16,activation='relu'))
# add dropout rate.
model.add(Dropout(0.1))
# Adding hidden layer with relu as activation function with 8 neurons
model.add(Dense(8,activation='relu'))
# Add output layer with sigmoid as activation function
model.add(Dense(1, activation = 'sigmoid'))

model.summary() #Print concise summary of the defined neural network architecture

# **Observation:**
# 1. Layer display each layer in the model
# 2. Output shape, shows the dimension of output data after passing through that layer
# 3. Number of parameter, calculates and present total trainable and no-trainable parameter in each layer and entire model
# 4. Total parameter is 1057, Trainable parameter is 1057 and Non-trainable parameter is 0.

optimizer = tf.keras.optimizers.Adam() # defining Adam as the optimizer
model.compile(loss='binary_crossentropy', optimizer=optimizer) #function is used to configure the learning process of the model before training begins.

epochs = 25 #model will see and learn from the complete training data 25 times
batch_size = 64 #Model will take 64 training example, calculate error and update is weight based on error.

start = time.time() #record current time and store in a variable
history = model.fit(X_train_smote, y_train_smote, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs) #train model with specified parameter.
end = time.time() #record end time and store in a variable

print("Time taken in seconds ",end-start) #printing the value

plot(history,'loss') #Plot the training and validation loss over the training epochs

model_5_train_perf = model_performance_classification(model, X_train_smote, y_train_smote) #calculate the performce of trained neural network model on training data
model_5_train_perf

model_5_val_perf = model_performance_classification(model, X_val_smote, y_val_smote) #calculate the performce of trained neural network model on validation data
model_5_val_perf

# **Observation:**
# By Applying SMOTE, Adam optimizer and Drop-out, Recall score improves little bit only. Training Recall score is ~0.81 an

# Model Performance Comparison and Final Model Selection

# training performance comparison

models_train_comp_df = pd.concat(
    [
        model_0_train_perf.T,
        model_1_train_perf.T,
        model_2_train_perf.T,
        model_3_train_perf.T,
        model_4_train_perf.T,
        model_5_train_perf.T
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Neural Network (SGD Optimizer)",
    "Neural Network (Adam Optimizer)",
    "Neural Network (Adam Optimizer, Drop-out)",
    "Neural Network (Smote, SGD Optimizer)",
    "Neural Network (Smote, Adam Optimizer)",
    "Neural Network (Smote, Adam Optimizer, Drop-out)",
]


#Validation performance comparison

models_val_comp_df = pd.concat(
    [
        model_0_val_perf.T,
        model_1_val_perf.T,
        model_2_val_perf.T,
        model_3_val_perf.T,
        model_4_val_perf.T,
        model_5_val_perf.T
    ],
    axis=1,
)
models_val_comp_df.columns = [
    "Neural Network (SGD Optimizer)",
    "Neural Network (Adam Optimizer)",
    "Neural Network (Adam Optimizer, Drop-out)",
    "Neural Network (Smote, SGD Optimizer)",
    "Neural Network (Smote, Adam Optimizer)",
    "Neural Network (Smote, Adam Optimizer, Drop-out)",
]

models_train_comp_df

models_val_comp_df

# **Observation:**
# Among all other models, Model 3 (Neural Network model with Adam Optimizer an Drop-out)achieved the highest training and 
# This model has a training Recall score of ~81% and a validation Recall score of 80%
# This indicates that the model is likely not overfitting and is generalizing well on unseen data.
# We will choose this model as our final model.

# **Final Model**

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
#input layer with 32 neurons and relu as activation function
model.add(Dense(32,activation='relu',input_dim = X_train.shape[1]))
# adding dropout of 0.2
model.add(Dropout(0.2))
# Hidden layer with 64 neurons and relu as activation function
model.add(Dense(64,activation='relu'))
# Hidden layer with 32 neurons and relu as activation function
model.add(Dense(32,activation='relu'))
# adding dropout of 0.1
model.add(Dropout(0.1))
# Hidden layer with 16 neurons and relu as activation function
model.add(Dense(16,activation='relu'))
# Output layer with sigmoid as activation function
model.add(Dense(1, activation = 'sigmoid'))

model.summary() #Print concise summary of the defined neural network architecture

# **Observation:**
# 1. Layer display each layer in the model
# 2. Output shape, shows the dimension of output data after passing through that layer
# 3. Number of parameter, calculates and present total trainable and no-trainable parameter in each layer and entire model
# 4. Total parameter is 5121, Trainable parameter is 5121 and Non-trainable parameter is 0.

optimizer = tf.keras.optimizers.Adam(0.001)    # defining Adam as the optimizer and learning rate of 0.001
model.compile(loss='binary_crossentropy', optimizer=optimizer) #function is used to configure the learning process of the model before training begins.

epochs = 25 #model will see and learn from the complete training data 25 times
batch_size = 64 #Model will take 64 training example, calculate error and update is weight based on error.

start = time.time() #record current time and store in a variable
history = model.fit(X_train, y_train, validation_data=(X_val,y_val) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict) #train model with specified parameter.
end=time.time() #record end time and store in a variable

print("Time taken in seconds ",end-start) #printing the time

plot(history,'loss') #Plot the training and validation loss over the training epochs

model_train_perf = model_performance_classification(model, X_train, y_train) #calculate the performce of trained neural network model on training data
print("Train performance")
model_train_perf

model_val_perf = model_performance_classification(model, X_val, y_val) #calculate the performce of trained neural network model on validation data
print("Validation performance")
model_val_perf

model_test_perf = model_performance_classification(model, X_test, y_test) #calculate the performce of trained neural network model on test data
print("Test performance")
model_test_perf

# **Observation:**
# The Recall score on the test data is ~0.79
# A recall score of 0.79 indicates a moderately good performance in identifying churners.

y_train_pred = model.predict(X_train) #Generate predictions using trained neural network model on train data
y_val_pred = model.predict(X_val) #Generate predictions using trained neural network model on validation data
y_test_pred = model.predict(X_test) #Generate predictions using trained neural network model on test data

print("Classification Report - Train data",end="\n\n") #printing classification report on Training data
cr = classification_report(y_train,y_train_pred>0.5)
print(cr)

print("Classification Report - Validation data",end="\n\n") #printing classiication report on validation data
cr = classification_report(y_val,y_val_pred>0.5)
print(cr)

print("Classification Report - Test data",end="\n\n") #printing classification report on test data
cr = classification_report(y_test,y_test_pred>0.5)
print(cr)

# **Observation:**
# The Recall score on the test data is ~0.75 for class 1.0, which means that the model correctly identified 75% of the cus

# Actionable Insights and Business Recommendations

# **Key Insights**
# 1. Customers in Germany are more prone to churn compared to those in France or Spain. This could be driven by competitiv
# 2. Female customers exhibit a higher churn rate than male customers, potentially indicating unmet needs or expectations 
# 3. Customers with 4 products have the highest likelihood of churning, suggesting potential risks tied to over-diversific
# 4. Customers with higher balances but reduced activity show significant churn risk, emphasizing the importance of mainta
# 5. Older customers are more likely to churn than younger ones, indicating the need for age-specific retention strategies
# **Business Recommendation**
# 1. Focus retention efforts on German customers through:
# - Personalized offers tailored to their preferences.
# - Loyalty programs that reward long-term commitment.
# - Enhanced customer support to address specific concerns and foster trust.
# 2. Address Gender-Specific Needs:
# - Develop initiatives to understand and cater to the preferences of female customers.
# 3. Reduce Product Diversification Risks:
# - Identify potential challenges faced by customers with multiple products and simplify offerings where necessary.
# - Offer specialized support or benefits for customers with diversified product portfolios.
# 4. Engagement for High-Balance Inactive Customers:
# - Launch premium loyalty programs or exclusive services targeting customers with high balances.
# - Use personalized outreach campaigns to re-engage inactive customers effectively.
# 5. Build Long-Term Relationships with Younger Customers:
# - Design engagement strategies aimed at younger demographics, such as loyalty programs or mobile-friendly platforms.
# - Develop campaigns to attract younger customers into staying connected over the years.

# <font size=6 color='blue'>Power Ahead</font>
# ___
