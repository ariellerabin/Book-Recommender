
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
# combined dictionary
comb = {key: [d[key] for d in dictionaries] for key in dict_ASL}

X = [stats for stats in comb.values()]

# Convert the list of lists to a NumPy array
X_array = np.array(X)
# Choose the number of clusters
num_clusters = 5 # taken from elbowing

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_array)

# Add a new key to dictionary indicating the cluster for each text
text_clusters = {text: cluster for text, cluster in zip(comb.keys(), kmeans.labels_)}

# Use PCA to reduce the feature dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_array)

# Plot the clusters
fig_cluster = plt.figure(figsize=(8, 6))
scatter_traces = []
for i in range(num_clusters):
    cluster_points = X_pca[kmeans.labels_ == i]
    cluster_titles = [title for title, cluster_label in text_clusters.items() if cluster_label == i]
    scatter_trace = go.Scatter(
        x=cluster_points[:, 0],
        y=cluster_points[:, 1],
        mode='markers',
        name=f'Cluster {i + 1}',
        text=cluster_titles,
        hoverinfo='text+x+y',  # Display text, x, and y in the hover tooltip
    )
    scatter_traces.append(scatter_trace)

layout = go.Layout(
    title='Clustering Books Based on Various Statistics',
    xaxis=dict(title='Principal Component 1'),
    yaxis=dict(title='Principal Component 2'),
    legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color to fully transparent
    paper_bgcolor='rgba(0,0,0,0)'
)

fig_cluster = go.Figure(data=scatter_traces, layout=layout)

app = Dash(__name__)

# Define the layout with tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Get Your Book Recs!', children=[
            html.H1('Book Recommendation Engine', style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
            html.H3('Arielle Rabinovich and Andy Sun', style={'font-family': 'Didot, Georgia', 'color': '#565264'}),
            html.Div([
                html.H3('Figure of Clustered Books', style={'font-family': 'Didot, Georgia', 'color': '#565264'}),
                dcc.Graph(figure=fig_cluster)
            ], style={'background-color': 'rgba(0,0,0,0.0)'}),  # Set background color to fully transparent
            html.Div([
                html.H3('Book Recommendation', style={'font-family': 'Didot, Georgia', 'color': '#565264'}),
                dcc.Dropdown(
                    id='book-dropdown',
                    options=[{'label': book, 'value': book} for book in text_clusters.keys()],
                    value=list(text_clusters.keys())[0],
                    style = {'background-color': 'transparent'}# Set default value
                ),
                dcc.Markdown(id='recommendation-output', dangerously_allow_html=True, style={'background-color': 'transparent'}),
            ], style = {'background-color':'transparent'})  # Set background color to fully transparent
        ]),
        dcc.Tab(label='The Boring Stuff', children=[
        html.H1('The Inspiration', style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
         html.P("Every passionate reader understands the bittersweet moment of finishing a great book and yearning to relive the thrill of the first read. Short of using a memory eraser, the next best solution is to discover a book that captures a similar essence. That's the mission of this project. By quantifying writing styles, we've grouped together excerpts, allowing us to pinpoint the most kindred literary companions for each user. Welcome to a world where we unravel the magic of your favorite reads and lead you to new adventures that resonate with your unique taste", style={'font-family': 'Exo, Georgia', 'color': '#565264'}),

        html.H1('How we Quantified Writing Style', style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
            html.P(f"Capturing the essence of writing style is a nuanced challenge. It embodies the unique voice of an author, defying any attempts to break it down to numbers and a formula. So, how can one truly encapsulate it?", style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
                    html.P(f" Our journey commenced by examining term frequencies using the TFIDF library. However, our initial attempt resulted in scattered clusters, lacking coherence. Recognizing the need for refinement, we delved deeper. Exploring the intricacies of writing style, we considered factors such as the average syllable count, word frequency, sentence length, word length, Flesch-Kincaid score, Automated Readability Index (ARI). While coding emotions into a machine remains an insurmountable task, we wanted to get close. We used the aforementioned statistics to try and put a value on writing style, and vectorized them, allowing us to plot clearer clusters. The result was a more refined representation, far surpassing the clarity of our initial attempt.", style={'font-family': 'Exo, Georgia', 'color': '#565264'}),

            html.H1('What we Want to Acheive in the Future', style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
            html.P(
                " One of the most time-consuming aspects of our project was building our database. We gathered 100 excerpts from different online sources, including well-known ones like Penguin Publishers, Goodreads, and other free platforms. We're excited about the potential to expand our database since we can easily process and analyze book lists of any size in CSV format. Adding new books will enhance the quality of our recommendations.",
                style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
            html.P(
                "Furthermore, we're eager to include user data in our recommendations. Unlike popular platforms like Goodreads, our goal is to move away from conventional methods that often left us wanting more. Relying solely on finding similar users didn't align with what we wanted to achieve. However, people have a unique understanding of emotions and writing style, which can't be replicated by a program. We see value in receiving recommendations from individuals who have enjoyed books similar to yours. Combining our approach with user-based methods is something we believe will elevate the overall quality of our recommendations.", style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
            html.P(
                "Additionally, we acknowledge that some aspects need fine-tuning. Due to the intricacies of the English language, accurately determining the number of syllables in a word can be challenging for a computer. This may result in some metrics being slightly off, although they are mostly accurate. These small variations can impact our calculations.", style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
            html.P(
                "Overall our goal was to recommend books in a distinctive and non-traditional manner, and we believe we've achieved that. Looking ahead, our next step is to expand and enhance our project, and we're excited about the possibilities.", style={'font-family': 'Exo, Georgia', 'color': '#565264'}),
            html.P(
                "Happy Reading!", style={'font-family': 'Exo, Georgia', 'color': '#565264'}),

        ]),
    ]),
], style={'background-color': "#D6CFCB"})  # making it pretty

def get_user_input_recommendations(user_input):
    user_input_cluster = text_clusters.get(user_input, -1)

    if user_input_cluster != -1:
        # Filter books in the same cluster as the user-input book
        cluster_books = [book for book, cluster in text_clusters.items() if cluster == user_input_cluster]

        # Remove the user-input book from the list
        cluster_books.remove(user_input)

        # Format the recommendations as a bulleted list with a specific font
        recommendations = [f"<li style='font-family: Exo; color: #565264'>{book}</li>" for book in cluster_books[:5]]

        # Use a different font for the title
        title = f"<h3 style='font-family: Exo; color: #565264'>Top 5 books most similar to '{user_input}'</h3>"

        # Combine the title and recommendations
        return f"{title}<ul>{''.join(recommendations)}</ul>"


@app.callback(
    Output('recommendation-output', 'children'),
    [Input('book-dropdown', 'value')]
)
def update_recommendation_output(selected_book):
    return get_user_input_recommendations(selected_book)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)