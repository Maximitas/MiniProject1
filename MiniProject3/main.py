import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm

import pickle

from data_loader import load_csv

#   1. Data wrangling and exploration

#I use a csv loader from last project and then just paste in the path to the data
data = load_csv("~/Desktop/MiniProject3/data/Employee-Attrition.csv")

#I use some commands from pandas to get a feel of the data.
#data.head lets me look at the first 5 rows on the dataframe
#data.shape tells me how many rows and columns there are
#data.nunique tells me how many unique values there are in each column
#Then i can delete the rows with only one unique value as they are not useful
def printData(data):
    print(data.head())
    print(data.shape)
    print(data.nunique())
    print(data.isnull().sum())

printData(data)

#I drop the columns with only one unique value
data = data.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])
#I can also see that some of the columns are binary, meaning they only have two unique values
unique_counts = data.nunique()
binary_columns = unique_counts[unique_counts == 2].index

print(f"These cols are binary: {list(binary_columns)}")

#The values are tekst and mashine learning works better with numbers, so I convert the binary columns to 0 and 1
binary_cols = ['Attrition', 'Gender', 'OverTime', 'PerformanceRating']
for col in binary_cols:

    data[col] = data[col].astype('category').cat.codes

cols_to_display = ['Attrition', 'Gender', 'OverTime', 'PerformanceRating']

print(data[cols_to_display].head())

#I also convert the other tekst columns to numbers using one-hot encoding

datatypes = data.dtypes
object_columns = datatypes[datatypes == 'object'].index

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(data[object_columns])

encoded_data = encoder.transform(data[object_columns])

#I make a new dataframe with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(object_columns))

#Then i clean out the original columns with tekst data
data_cleaned = data.drop(columns=object_columns)

#And then combine the two dataframes
data_final = pd.concat([data_cleaned, encoded_df], axis=1)

print(data_final.head())
print(data_final.shape)

#Now i have data that is ready for machine learning


#   2. Supervised machine learning: linear regression

#I calulate the corralation matrix to monthly income and sort it to see which features are most correlated
correlation_matrix = data_final.corr()
income_corr = correlation_matrix['MonthlyIncome'].sort_values(ascending=False)
print(income_corr)

top_corr_features = income_corr.index[1:13]
cols_to_plot = list(top_corr_features) + ['MonthlyIncome']

selected_features = list(top_corr_features)

#I make a new correlation matrix with only the top 12 features
correlation_subset = data_final[cols_to_plot].corr()

#then here i plot the heatmap
plot.figure(figsize=(10, 8))
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plot.title('Correlation matrix for MonthlyIncome')
plot.show()


#Then i make box plots for the top 12 features to look for outliers that could skew the data
plot.figure(figsize=(12, 12))

for i, col in enumerate(top_corr_features):
    plot.subplot(3, 4, i + 1) 
    sns.boxplot(y=data_final[col])
    plot.title(f'Box Plot for {col}')
plot.tight_layout()
plot.show()

#I make a function to remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

#I can see that there are some outliers in the data
outlier_columns = [
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'NumCompaniesWorked'
]
for col in outlier_columns:
    data_final = remove_outliers(data_final, col)
print(f"Data shape after outlier removal: {data_final.shape}")

plot.figure(figsize=(12, 12))

for i, col in enumerate(top_corr_features):
    plot.subplot(3, 4, i + 1)
    sns.boxplot(y=data_final[col])
    plot.title(f'Box Plot uden outliers for {col}')

plot.tight_layout()
plot.show()

#Now that i have loaded and cleaned the data, and made it viable for machine learning
#I can choose the dependent and independent variables
#I want to make linear regression to predict monthly income so i use the 12 most correlated features
X = data_final[top_corr_features]
y = data_final['MonthlyIncome']

#Here i split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#Here i train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Then i use the model to predict the test data that it has never seen before
y_pred = model.predict(X_test)

#Then i evaluate how good the model is with mean absolute error(mae), mean squared error(mse), root mean squared error(rmse)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")


plot.figure(figsize=(8, 6))
plot.scatter(y_test, y_pred, alpha=0.6)
plot.xlabel("Faktisk MonthlyIncome")
plot.ylabel("Forudsagt MonthlyIncome")
plot.title("Faktisk vs. Forudsagt MonthlyIncome")
plot.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal linje
plot.show()


#    3. Supervised machine learning: classification


#So i take all the data and drop Attrition to make X and then use Attrition as y
#So X is the data that tries to predict y
X_class = data_final.drop(columns=['Attrition'])
y_class = data_final['Attrition']

#Here i split the data into 80% training and 20% testing just like in the linear regression
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, random_state=42, test_size=0.2,)

#I train the logistic regression model, i had to use a pipeline with standard scaler to get it to converge
log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
log_reg.fit(X_train, y_train)

#Here i use the model to predict the test data
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1] # Probability estimates for the positive class

acc_stay = recall_score(y_test, y_pred, pos_label=0)
acc_leave = recall_score(y_test, y_pred, pos_label=1)

#Then i print some metrics to evaluate how good the model is
#And a 87% accuracy is pretty good, but the data is quite skewed towards the employees that do not leave
#When it tries to predict employees that stay it is right 95% of the time
#but when it tries to predict employees that leave it is only right 56% of the time
print("Accuracy:", accuracy_score(y_test, y_pred))
print(f"Accuracy for staying (0): {acc_stay:.2f}")
print(f"Accuracy for leaving (1): {acc_leave:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#And then a ROC curve plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plot.figure(figsize=(8,6))
plot.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})")
plot.plot([0,1], [0,1], 'k--')  # baseline
plot.xlabel("False Positive Rate")
plot.ylabel("True Positive Rate")
plot.title("ROC Curve for Logistic Regression (Attrition)")
plot.legend(loc="lower right")
plot.show()


#    4. Unsupervised machine learning: clustering

#So for this i will use KMeans clustering to try to find patterns in the data
#I dont have a y variable to predict as i want the data to find the patterns itself
X_cluster = data_final.drop(columns=['Attrition'])  

#This right here scales the data so some data doesnt dominate the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

#Here i will try different number of clusters  to find what makes the most sense.
#+1 is perfect, 0 is means the groups are overlapping, and -1 is an error and the data is placed wrongly.
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)
    print(f"Number of clusters: {k}, Silhouette Score: {score:.3f}")

#Here i plot the silhouette scores to see the optimal number of clusters
plot.figure(figsize=(8,5))
plot.plot(K_range, silhouette_scores, 'bo-')
plot.xlabel('Number of Clusters')
plot.ylabel('Silhouette Score')
plot.title('Silhouette Scores for Different K')
plot.show()

#Here i fit in the best number of clusters
best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)
data_final['Cluster'] = cluster_labels

#Here i can look at the cluster sizes
print(data_final['Cluster'].value_counts())

#All in all my cluster scores are quite low, meaning that the data is quite overlapping and not very well suited for clustering

print(data_final[['Education','JobSatisfaction']].corr())
print(data_final[['DistanceFromHome','WorkLifeBalance']].corr())
print(data_final[['MaritalStatus_Single','Attrition']].corr())
