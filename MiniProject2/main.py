import os
from data_loader import load_wine_data
import matplotlib.pyplot as plot

#https://www.kaggle.com/datasets/yasserh/wine-quality-dataset?resource=download
#Where the public data is from

folder = os.path.expanduser("~/Desktop/MiniProject2/winedata")

wine_df = load_wine_data(folder)

print("Dataset shape:", wine_df.shape)
print(wine_df.head())

print("\nMissing values per column:\n", wine_df.isnull().sum())

avg_quality = wine_df.groupby('type')['quality'].mean().reset_index()
plot.bar(avg_quality['type'], avg_quality['quality'])
plot.title("Average Quality by Wine Type")
plot.xlabel("Wine Type")
plot.ylabel("Average Quality")
plot.show()

plot.hist(wine_df["alcohol"], bins=20, edgecolor='black')
plot.title("Distribution of Alcohol Content")
plot.xlabel("Alcohol Content")
plot.ylabel("Count")
plot.show()

plot.scatter(wine_df["alcohol"], wine_df["quality"], alpha=0.5)
plot.title("Alcohol vs. Quality")
plot.xlabel("Alcohol Content")
plot.ylabel("Quality Score")
plot.show()
