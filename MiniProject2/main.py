import os
from data_loader import load_wine_data
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

folder = os.path.expanduser("~/Desktop/MiniProject2/winedata")
wine_df = load_wine_data(folder)

#Task 5: Explore the dataset
#So here i look look at the data
print("\n Basic Info")
print(wine_df.info())

print("\n First Rows")
print(wine_df.head())

#Im Identifying Dependent and Independent Variables
dependent_var = "quality"
independent_vars = [col for col in wine_df.columns if col not in [dependent_var, "type"]]
print("\n Dependent variable:", dependent_var)
print("\n Independent variables:", independent_vars)

#Here im plotting some basic stuff to visiualize the data
#Like the distribution of alcohol content and average quality by wine type
plot.hist(wine_df["alcohol"], bins=20, edgecolor='black')
plot.title("Distribution of Alcohol Content")
plot.xlabel("Alcohol Content (%)")
plot.ylabel("Count")
plot.show()

avg_quality = wine_df.groupby('type')['quality'].mean().reset_index()
plot.bar(avg_quality['type'], avg_quality['quality'], edgecolor='black')
plot.title("Average Quality by Wine Type")
plot.xlabel("Wine Type")
plot.ylabel("Average Quality")
plot.ylim(5.0, 6.0)
plot.show()

#Task 6: Transform the dataset
#Here i make a new dataframe for the transformed data. dp.get_dummies makes a hot-one encoding of the type column. So White wine and Wine fra public sources become 0s and 1s instead of strings. drop_first=True drops the first column to avoid multicollinearity, so i drops Red wine.
#This is what i read online one should do. It is quite intresting. Apparently it can give trouble if you take all the categories as columns in mashinelearning and statistical modules. 
#So if it dosent hit either of the two categories it must be the dropped one. So if they both say flase it must be red wine.
wine_encoded = pd.get_dummies(wine_df, columns=['type'], drop_first=True)

print("\nOriginal columns:", wine_df.columns.tolist())
print("\nEncoded DataFrame columns:", wine_encoded.columns.tolist())
print("\nFirst 5 rows of the encoded data:")
print(wine_encoded.head())

#Task 7
#Here im printing some basic statistics and plotting some plots to see the distribution of some of the key variables
print("\n Descriptive Statistics")
print(wine_df.describe().T)

for col in ["alcohol", "residual_sugar", "quality"]:
    plot.hist(wine_df[col], bins=30, edgecolor="black")
    plot.title(f"Distribution of {col}")
    plot.xlabel(col)
    plot.ylabel("Count")
    plot.show()

#Task 8:
#B: I have a diagram from earlier that shows the average quality by wine type, and it was whitewine that had the highest average quality.

#C: This one shows the average alcohol content by wine type and again white wine wins. IT has the highest Average Alcohol (%). 
avg_alcohol = wine_df.groupby("type")["alcohol"].mean().reset_index()
plot.bar(avg_alcohol["type"], avg_alcohol["alcohol"], edgecolor="black")
plot.title("Average Alcohol Content by Wine Type")
plot.ylabel("Average Alcohol (%)")
plot.ylim(10.0, 11.0)
plot.show()

# D: And once again white wine comes out on top with the highest average residual sugar on average.
avg_sugar = wine_df.groupby("type")["residual_sugar"].mean().reset_index()
plot.bar(avg_sugar["type"], avg_sugar["residual_sugar"], edgecolor="black")
plot.title("Average Residual Sugar by Wine Type")
plot.ylabel("Average Residual Sugar (g/L)")
plot.show()

#E: Here im plotting some scatter plots to see if there is any correlation between alcohol content, residual sugar and quality.
#It can be seen on the scatterplot that wines with higher alcohol content acually tent to have a higher quality rating.
plot.scatter(wine_df["alcohol"], wine_df["quality"], alpha=0.5)
plot.title("Alcohol vs Quality")
plot.xlabel("Alcohol (%)")
plot.ylabel("Quality")
plot.show()

#With residual sugar it is the opposite. Wines with higher residual sugar tent to lean towards a quality of 6.
#A lower amount of residual sugar tent to have a higher or lover rating than 6.
plot.scatter(wine_df["residual_sugar"], wine_df["quality"], alpha=0.5)
plot.title("Residual Sugar vs Quality")
plot.xlabel("Residual Sugar (g/L)")
plot.ylabel("Quality")
plot.show()

#Task 9: I would think that consumers would primarely be intressed in taste, price and quality.
#And distributers would be intressed in quality to know what to promote and sell.
#They would like to know how much of certain chemicals coralates with the best quality.


#Task 10: So im splitting it into 5 bins to to see the density of the ph values.
wine_df["ph_bin_5"] = pd.cut(wine_df["ph"], bins=5)
ph_counts_5 = wine_df["ph_bin_5"].value_counts().sort_index()

#Im plotting a histogram with 5 bins
#And it can be seen that the highest density of phvalue is at 3.0 to 3.45
plot.hist(wine_df["ph"], bins=5, edgecolor="black")
plot.title("Distribution of ph (5 bins)")
plot.xlabel("ph")
plot.ylabel("Count")
plot.show()

#Im doing the same just with 10 bins
wine_df["ph_bin_10"] = pd.cut(wine_df["ph"], bins=10)
ph_counts_10 = wine_df["ph_bin_10"].value_counts().sort_index()

#And here it gets plotted
#And if we make it 10 bins we can see that that it is actually at 3.1 to 3.4 that is the highest density of ph values.
plot.hist(wine_df["ph"], bins=10, edgecolor="black")
plot.title("Distribution of pH (10 bins)")
plot.xlabel("pH")
plot.ylabel("Count")
plot.show()

#Task 11: Here i calculate the correlation betwteen the numeric variables in the dataset.
#Correlation is a number between 1 and -1 that shows how much two variables are related.
#1 mean they are perfectly correlated, so if one goes up the other does too. -1 means the opposite, if one goes up the other goes down.
#And 0 means there is no correlation.
corr_matrix = wine_df.corr(numeric_only=True)

plot.figure(figsize=(10,8))
plot.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plot.colorbar(label="Correlation")
plot.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=30, ha='right')
plot.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plot.title("Correlation Matrix Heatmap")
plot.show()

#Here i can see in a print the correlation with quality
#It shows that alchol has the highest correlation with quality at 0.47 and density has the lowest at -0.32
quality_corr = corr_matrix["quality"].sort_values(ascending=False)
print("\n Correlation with Quality:")
print(quality_corr)


#Task 12: Here i am box plotting to see if there are any outliers in the data.
#It can be seen there there are indeed some. Expecially in residual sugar.
wine_df.boxplot(figsize=(12,6))
plot.title("Boxplots of Wine Features (for Outlier Detection)")
plot.xticks(rotation=30, ha='right')
plot.show()

df = wine_df.copy()

#Here since i could see that residual sugar and free_sulfur_diaxide had outliers on my boxplot i will take that col and find the outliers using the IQR method.
#I found a method online, but since i only really wanted it on residual sugar i had help from chatgpt to make it take a column as input.
#q1=df.quantile(0.25)
#q3=df.quantile(0.75)
#IQR=q3-q1
#outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
def find_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower) | (data[col] > upper)]
    return outliers, lower, upper

#Residual_sugar
outliers_sugar, low_sugar, high_sugar = find_outliers_iqr(df, "residual_sugar")
print("\nOutliers for residual_sugar")
print(outliers_sugar[["residual_sugar", "type", "quality"]])
print(f"Lower bound: {low_sugar:.2f}, Upper bound: {high_sugar:.2f}")
print(f"Number of outliers: {len(outliers_sugar)}")

#Free_sulfur_dioxide
outliers_sulfur, low_sulfur, high_sulfur = find_outliers_iqr(df, "free_sulfur_dioxide")
print("\nOutliers for free_sulfur_dioxide")
print(outliers_sulfur[["free_sulfur_dioxide", "type", "quality"]])
print(f"Lower bound: {low_sulfur:.2f}, Upper bound: {high_sulfur:.2f}")
print(f"Number of outliers: {len(outliers_sulfur)}")

#Removing outliers
df_cleaned = df[
    (df["residual_sugar"].between(low_sugar, high_sugar)) &
    (df["free_sulfur_dioxide"].between(low_sulfur, high_sulfur))
]

print("\n Shape before removing outliers:", df.shape)
print("\n Shape after removing outliers:", df_cleaned.shape)

#Task 13. If i make use of a prior print that showed the correlation with quality i can see what virables arent really correlated with quality.
#Such as ph, total_sulfur_dioxide, residual_sugar and fixed_acidity.
#So i will drop those and make a new dataframe.
df_final = df_cleaned.drop(columns=["ph", "total_sulfur_dioxide", "residual_sugar", "fixed_acidity"])

#Task 14
#This line finds all the columsn that are numeric in the dataframe and makes a list of them.
#This way i dont get any errors when i try to scale the data.
numeric_cols = df_final.select_dtypes(include="number").columns.tolist()

#Here i do the min max scaling. I make a copy of my dataframe and then i scale the columns.
#fit_transform is where all the magic happens.
#It calculates the min and max for each column and then scales the values to be between 0 and 1.
#I then print them
scaler = MinMaxScaler()
df_scaled = df_final.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])

print("\n First 5 rows after Min-Max scaling:")
print(df_scaled.head())

#Here i do the standardization. I make a copy of my dataframe and then i scale the columns.
#fit_transform is where all the magic happens again, but this time it uses the standard form.
scaler_std = StandardScaler()
df_standardized = df_final.copy()
df_standardized[numeric_cols] = scaler_std.fit_transform(df_final[numeric_cols])

print("\n First 5 rows after Standardization:")
print(df_standardized.head())