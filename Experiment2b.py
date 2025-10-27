# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import re
import os.path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def get_ada_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    api_key = "your open-ai api key here"            
    
    if api_key == "your open-ai api key here" or not api_key:
        print("Error: Missing OpenAI API Key...")
        print("Please replace 'your open-ai api key here' in the script with your actual key.")
        return None

    if texts is None or not texts:
        print("No texts provided to embed.")
        return None

    texts_to_embed = []
    for text in texts:
        if text is None:
            texts_to_embed.append("neutral")
        else:
            str_text = str(text).strip()
            if not str_text:
                texts_to_embed.append("neutral")
            else:
                texts_to_embed.append(str_text)
                
    if not texts_to_embed:
        print("All texts were empty, nothing to embed.")
        return None


    BATCH_SIZE = 2048
    all_embeddings = []

    try:
        client = OpenAI(api_key=api_key)
        
        for i in range(0, len(texts_to_embed), BATCH_SIZE):
            batch = texts_to_embed[i:i + BATCH_SIZE]
            
            # Print progress
            print(f"Embedding batch {i // BATCH_SIZE + 1} / {len(texts_to_embed) // BATCH_SIZE + 1}...")

            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch,
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return np.array(all_embeddings)

    except Exception as e:
        print(f"Error generating ADA embeddings during batching: {e}")
        return None


def load_vad_lexicon(file_path: str = 'vad_lexicon.csv') -> Optional[Dict[str, float]]:
    try:
        df = pd.read_csv(file_path)
        lexicon = dict(zip(df['Word'], df['A.Mean.Sum']))
        return lexicon
    except Exception as e:
        print(f"Error loading VAD lexicon '{file_path}': {e}")
        return None


def calculate_vad_scores(texts: List[str], vad_lexicon: Dict[str, float]) -> np.ndarray:
    arousal_scores = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        word_scores = [vad_lexicon[word] for word in words if word in vad_lexicon]
        
        if not word_scores:
            text_arousal = 5.0
        else:
            text_arousal = np.mean(word_scores)
        arousal_scores.append(text_arousal)
        
    return np.array(arousal_scores)


def estimate_emotion_intensity(texts: List[str]) -> np.ndarray:
    word_scores = {
        # Extreme Positive
        'absolutely': 1.0, 'incredibly': 1.0, 'magnificently': 1.0, 'phenomenally': 1.0,
        'ecstatic': 1.0, 'euphoric': 1.0, 'elated': 1.0, 'jubilant': 1.0, 'divine': 1.0,
        'wonderful': 1.0, 'happiness': 1.0, 'joy': 1.0, 'active': 1.0, 'sports': 1.0, 
        'optimistic': 1.0, 'successful': 1.0, 'outgoing': 1.0,
        # Extreme Negative
        'devastated': 1.0, 'destroyed': 1.0, 'shattered': 1.0, 'furious': 1.0,
        'enraged': 1.0, 'livid': 1.0, 'terrified': 1.0, 'horrifyingly': 1.0,
        'depressed': 1.0, 'psychiatrist': 1.0, 'exhausted': 1.0, 'crying': 1.0, 
        'extremely': 1.0, 'losing': 1.0, 'uncomfortable': 1.0, 'mental': 1.0,
        # Very Positive
        'amazing': 0.8, 'awesome': 0.8, 'fantastic': 0.8, 'wonderful': 0.8,
        'superb': 0.8, 'excellent': 0.8, 'brilliantly': 0.8,
        # Very Negative
        'hate': 0.8, 'despise': 0.8, 'loathe': 0.8, 'terrible': 0.8,
        'horrible': 0.8, 'awful': 0.8, 'dreadful': 0.8,
        # Positive
        'love': 0.6, 'adore': 0.6, 'happy': 0.6, 'glad': 0.6, 'pleased': 0.6, 'joyfully': 0.6,
        'great': 0.6, 'encouragingly': 0.6, 'fortunately': 0.6,
        # Negative
        'sad': 0.6, 'unhappy': 0.6, 'disappointed': 0.6, 'frustrated': 0.6,
        'angry': 0.6, 'miserably': 0.6, 'sadly': 0.6,
        # Mild Positive
        'okay': 0.3, 'fine': 0.3, 'decent': 0.3, 'nice': 0.3, 'good': 0.3, 'cool': 0.3,
        # Mild Negative
        'bad': 0.3, 'poor': 0.3, 'weak': 0.3, 'boring': 0.3
    }
    
    intensities = []
    for text in texts:
        intensity = 0.1
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        for word in words:
            intensity += word_scores.get(word, 0)
        if any(mod in words for mod in ['so', 'very', 'really', 'truly']):
            intensity += 0.3
        if any(mod in words for mod in ['never', 'always']):
            intensity += 0.2

        intensity += 0.25 * min(text.count('!'), 4)
        if text.isupper() and len(text) > 3:
            intensity += 0.5
        intensities.append(min(intensity, 2.0))
    return np.array(intensities)


if __name__ == "__main__":
    vad_lexicon = load_vad_lexicon()
    if not vad_lexicon:
        exit()

    emotions = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
        "remorse", "sadness", "surprise", "neutral"
    ]

    try:
        df_source = pd.read_csv('emotions27.tsv', sep="\t")
        df_source.columns = ["text", "label_id", "id"]
    except FileNotFoundError:
        print("Error: 'emotions27.tsv' not found.")
        exit()

    all_texts = []
    all_labels = []
    
    for emotion_name in emotions:
        emotion_index = str(emotions.index(emotion_name))
        emotion_texts = df_source[df_source["label_id"] == emotion_index]["text"].astype(str).tolist()
        
        if not emotion_texts:
            print(f"Warning: No texts found for emotion '{emotion_name}'. Skipping.")
            continue
            
        all_texts.extend(emotion_texts)
        all_labels.extend([emotion_name] * len(emotion_texts))

    df_analysis = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })

    embeddings = get_ada_embeddings(df_analysis['text'].tolist())
    if embeddings is None:
        print("Failed to get embeddings. Exiting.")
        exit()
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    N_CLUSTERS = 12
    print(f"Running K-Means to find {N_CLUSTERS} clusters (on 1536-dim data)...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df_analysis['cluster'] = kmeans.fit_predict(embeddings_scaled)

    for i in range(N_CLUSTERS):
        cluster_data = df_analysis[df_analysis['cluster'] == i]
        print(f"\n--- Cluster {i} (Size: {len(cluster_data)} texts) ---")
        
        emotion_dist = cluster_data['label'].value_counts(normalize=True).head(5)
        print("Top 5 Emotions:")
        for emotion, percent in emotion_dist.items():
            print(f"  - {emotion}: {percent:.1%}")
            
        vad_scores = calculate_vad_scores(cluster_data['text'].tolist(), vad_lexicon)
        print(f"Average VAD Arousal (Ground Truth): {np.mean(vad_scores):.3f}")
        
        custom_scores = estimate_emotion_intensity(cluster_data['text'].tolist())
        print(f"Average Custom Intensity (Heuristic): {np.mean(custom_scores):.3f}")



    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    
    df_analysis['tsne_x'] = embeddings_2d[:, 0]
    df_analysis['tsne_y'] = embeddings_2d[:, 1]
    print("t-SNE complete. Generating plots.")

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df_analysis,
        x='tsne_x',
        y='tsne_y',
        hue='label',
        palette=sns.color_palette("gist_ncar", n_colors=len(emotions)),
        s=10,
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Emotion Embeddings (Colored by True Label)', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df_analysis,
        x='tsne_x',
        y='tsne_y',
        hue='cluster',
        palette=sns.color_palette("tab20", n_colors=N_CLUSTERS),
        s=10,
        alpha=0.7
    )
    plt.title(f't-SNE Visualization of Emotion Embeddings (Colored by {N_CLUSTERS} K-Means Clusters)', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=2)
    plt.tight_layout()
    plt.show()

