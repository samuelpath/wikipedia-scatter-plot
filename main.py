import numpy as np
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import cohere
import plotly.io as pio
import plotly.graph_objects as go
import textwrap
import random

# Constants
COHERE_API_KEY = "COHERE_API_KEY"
SUBSET = "simple"
SAMPLE_SIZE = 30000
N_COMPONENTS = 2
N_CLUSTERS = 20
RANDOM_STATE = 0

def load_data():
    """
    Load the dataset and extract the documents and embeddings.
    """
    docs_stream = load_dataset(f"Cohere/wikipedia-2023-11-embed-multilingual-v3", SUBSET, split="train", streaming=True)
    docs = []
    embeddings = []
    for doc in docs_stream:
        docs.append(doc)
        embeddings.append(doc['emb'])
    return np.asarray(embeddings), docs

def sample_embeddings(embeddings, sample_size):
    """
    Sample a subset of embeddings and save the indices of the sampled embeddings.
    """
    total_embeddings = embeddings.shape[0]
    indices = np.random.choice(total_embeddings, size=sample_size, replace=False)
    np.save('indices.npy', indices)
    return embeddings[indices], indices

def reduce_dimensions(embeddings):
    """
    Reduce the dimensionality of the embeddings using t-SNE.
    """
    tsne = TSNE(n_components=N_COMPONENTS, random_state=RANDOM_STATE, verbose=1, n_jobs=-1)
    return tsne.fit_transform(embeddings)

def create_dataframe(embeddings_2d, docs, indices):
    """
    Create a DataFrame from the 2D embeddings and the corresponding document texts.
    """
    titles = [docs[i]['title'] for i in indices]
    hover_texts = [docs[i]['text'] for i in indices]
    hover_texts = ['<br>'.join(text[i:i+50] for i in range(0, len(text), 50)) for text in hover_texts]
    return pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'title': titles,
        'text': hover_texts,
    })

def perform_clustering(embeddings_2d):
    """
    Perform K-Means clustering on the 2D embeddings.
    """
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit(embeddings_2d)
    return kmeans.labels_

def get_cluster_texts(df):
    """
    Group the DataFrame by the 'cluster' column and get the 'text' column for each group.
    """
    return df.groupby('cluster')['text'].apply(list)

def generate_prompt(cluster_texts):
    """
    Generate a prompt for the language model by selecting random texts from each cluster.
    """
    all_prompts = []
    for cluster_number, texts in cluster_texts.items():
        random_texts = random.sample(texts, min(10, len(texts)))
        numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(random_texts))
        all_prompts.append(f"Cluster {cluster_number}:\n{numbered_texts}")
    return "\n\n".join(all_prompts) + "\n\nWhat would be suitable labels for these clusters of texts?"

def get_labels(prompt):
    """
    Feed the prompt to the language model and extract the labels from the response.
    """
    co = cohere.Client(COHERE_API_KEY)
    response = co.chat(message=prompt, model="command-r")
    return response.text.split("\n")

def assign_labels(df, cluster_texts, labels):
    """
    Assign the labels to the DataFrame.
    """
    for cluster_number, label in zip(cluster_texts.keys(), labels):
        df.loc[df['cluster'] == cluster_number, 'label'] = label
    return df

def create_plot(df):
    """
    Create a scatter plot of the clusters and save it as an HTML file.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly[:20]
    for cluster_number in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_number]
        fig.add_trace(go.Scatter(
            x=cluster_df['x'],
            y=cluster_df['y'],
            mode='markers',
            name=cluster_df['label'].iloc[0],
            marker=dict(color=colors[(cluster_number - 1) % len(colors)]),
            hovertemplate='Title: %{text}<br>Cluster: ' + cluster_df['label'].iloc[0] + '<br>Text: %{customdata}<extra></extra>',
            text=cluster_df['title'],
            customdata=[textwrap.fill(text, 50) for text in cluster_df['text']],  # Wrap the text every 50 characters
        ))

    fig.show()
    pio.write_html(fig, 'index.html')

def main():
    """
    Main function to orchestrate the execution of the script.
    """
    embeddings, docs = load_data()
    sampled_embeddings, indices = sample_embeddings(embeddings, SAMPLE_SIZE)
    embeddings_2d = reduce_dimensions(sampled_embeddings)
    df = create_dataframe(embeddings_2d, docs, indices)
    df['cluster'] = perform_clustering(embeddings_2d)
    cluster_texts = get_cluster_texts(df)
    df['label'] = ''
    prompt = generate_prompt(cluster_texts)
    labels = get_labels(prompt)
    df = assign_labels(df, cluster_texts, labels)
    create_plot(df)

if __name__ == "__main__":
    main()
