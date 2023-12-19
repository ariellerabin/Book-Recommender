from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from recommend import BookRecs

file_path = 'Books.csv'

# Load CSV into a DataFrame
df = pd.read_csv(file_path)

columns_to_drop = ['Author', 'Date Published', 'Decade']
df.drop(columns=columns_to_drop, inplace=True)

rec = BookRecs()
dict_ASL = (rec.avg_sentence_length(df=df, excerpt_column='Excerpt', title_column='Title'))
dict_AWL = (rec.avg_word_length(df=df, excerpt_column='Excerpt', title_column='Title'))
dict_AWF = (rec.avg_word_frequency(df=df, excerpt_column='Excerpt', title_column='Title'))
dict_ASC = (rec.avg_syllable_count(df=df, excerpt_column='Excerpt', title_column='Title'))
dict_FKS = (rec.flesch_kincaid_score(df=df, excerpt_column='Excerpt', title_column='Title'))
dict_ARI = (rec.ARI_score(df=df, excerpt_column='Excerpt', title_column='Title'))

dictionaries = [dict_ASL, dict_AWL, dict_AWF, dict_ASC, dict_FKS, dict_ARI]
# Your combined dictionary
comb = {key: [d[key] for d in dictionaries] for key in dict_ASL}

X = [stats for stats in comb.values()]

# Convert the list of lists to a NumPy array
X_array = np.array(X)

# Apply K-means clustering
wcss = []  # List to store the within-cluster sum of squares for each k

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_array)
    wcss.append(kmeans.inertia_)  # Inertia is the within-cluster sum of squares

# Plot the elbow curve
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()


