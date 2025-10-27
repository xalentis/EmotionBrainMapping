# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def plot_comparison_charts(pos_intensities: np.ndarray, pos_assignments: np.ndarray,
                           neg_intensities: np.ndarray, neg_assignments: np.ndarray,
                           region_names: list):

    sns.set_theme(style="whitegrid")
    df_pos = pd.DataFrame({'intensity': pos_intensities, 'region_idx': pos_assignments, 'group': 'Healthy'})
    df_neg = pd.DataFrame({'intensity': neg_intensities, 'region_idx': neg_assignments, 'group': 'Depressed'})
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    df_combined['region_name'] = df_combined['region_idx'].apply(lambda x: region_names[x].replace('_', ' ').title())
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 1, 1)
    sns.countplot(data=df_combined, y='region_name', hue='group', ax=ax1, palette={'Healthy': 'skyblue', 'Depressed': 'salmon'})
    ax1.set_title('Comparison of Activation Counts per Brain Region', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Number of Associated Texts', fontsize=12)
    ax1.set_ylabel('Brain Region', fontsize=12)
    ax1.legend(title='Group')
    plt.tight_layout()
    plt.show(block=False)


def get_ada_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    api_key = "your open-ai api key here"
    if not api_key:
        print("Missing OpenAI API Key...") 
        return None
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return np.array([item.embedding for item in response.data])
    except Exception as e:
        print(f"Error generating ADA embeddings: {e}")
        return None


class EmotionBrainMapper:
    def __init__(self):
        self.emotion_regions = self.define_emotion_regions()
        self.n_emotion_regions = len(self.emotion_regions)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=self.n_emotion_regions, random_state=42, n_init=10)
        self.brain = None

    def define_emotion_regions(self) -> Dict[str, List[float]]:
        return {
            'amygdala_left': [-20, -5, -18],
            'amygdala_right': [20, -5, -18],
            'anterior_cingulate_left': [-5, 25, 25],
            'anterior_cingulate_right': [5, 25, 25],
            'insula_left': [-40, 8, 0],
            'insula_right': [40, 8, 0],
            'orbitofrontal_left': [-25, 40, -15],
            'orbitofrontal_right': [25, 40, -15],
            'hippocampus_left': [-25, -15, -20],
            'hippocampus_right': [25, -15, -20],
            'prefrontal_cortex_left': [-45, 30, 30],
            'prefrontal_cortex_right': [45, 30, 30],
            'temporal_pole_left': [-40, 20, -25],
            'temporal_pole_right': [40, 20, -25],
            'superior_temporal_left': [-55, -25, 10],
            'superior_temporal_right': [55, -25, 10],
            'caudate_left': [-12, 12, 8],
            'caudate_right': [12, 12, 8],
            'putamen_left': [-25, 5, 0],
            'putamen_right': [25, 5, 0],
            'nucleus_accumbens_left': [-8, 8, -8],
            'nucleus_accumbens_right': [8, 8, -8],
            'hypothalamus': [0, -2, -15],
            'periaqueductal_gray': [0, -28, -10],
            'ventral_tegmental_area': [0, -15, -20],
            'raphe_nuclei': [0, -25, -30],
            'locus_coeruleus': [0, -37, -28],
            'posterior_cingulate': [0, -50, 25],
            'medial_prefrontal_cortex': [0, 45, 20]
        }

    def fit_transform_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided.")
        
        n_samples, n_features = embeddings.shape
        n_components = min(3, n_samples, n_features)
        if n_components < 3:
            self.pca = PCA(n_components=n_components)
            
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        embeddings_3d = self.pca.fit_transform(embeddings_scaled)
        
        if embeddings_3d.shape[1] < 3:
            padding = np.zeros((embeddings_3d.shape[0], 3 - embeddings_3d.shape[1]))
            embeddings_3d = np.hstack([embeddings_3d, padding])

        n_clusters = min(self.n_emotion_regions, embeddings.shape[0])
        if n_clusters != self.n_emotion_regions:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            kmeans = self.kmeans
            
        kmeans.fit(embeddings_3d)
        cluster_centers_3d = kmeans.cluster_centers_
        region_assignments = np.argmin(cdist(embeddings_3d, cluster_centers_3d), axis=1)
        region_coords_map = np.array(list(self.emotion_regions.values()), dtype=np.float64)

        assigned_regions = []
        used_indices = set()
        for center in cluster_centers_3d:
            dists = cdist([center], region_coords_map)[0]
            for idx in np.argsort(dists):
                if idx not in used_indices:
                    assigned_regions.append(idx)
                    used_indices.add(idx)
                    break

        cluster_to_region = dict(zip(range(len(assigned_regions)), assigned_regions))
        region_assignments_mapped = np.array([cluster_to_region[c] for c in region_assignments])
        brain_coordinates = region_coords_map[region_assignments_mapped].copy()
        return brain_coordinates, region_assignments

    def estimate_emotion_intensity(self, texts: List[str]) -> np.ndarray:
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

            if any(mod in words for mod in ['so', 'very', 'really', 'truly', 'completely', 'totally']):
                intensity += 0.3
            if any(mod in words for mod in ['never', 'always', 'everything', 'nothing']):
                intensity += 0.2

            intensity += 0.25 * min(text.count('!'), 4)
            intensity += 0.15 * min(text.count('?'), 3)
            if text.isupper() and len(text) > 3:
                intensity += 0.5
            intensities.append(min(intensity, 2.0))
        return np.array(intensities)


def run_emotion_mapping_analysis(title: str, conversation: List[str]):
    np.random.seed(42)
    embeddings = get_ada_embeddings(conversation)
    if embeddings is None:
        return None, None, None, None

    mapper = EmotionBrainMapper()
    brain_coords, region_assignments = mapper.fit_transform_embeddings(embeddings)
    emotion_intensities = mapper.estimate_emotion_intensity(conversation)

    # the results below are used in the 3D visualization, copy and paste:
    print(f"{title} Emotion Intensities: {np.round(emotion_intensities, 2)}")
    return mapper, embeddings, emotion_intensities, region_assignments, brain_coords


if __name__ == "__main__":
    df = pd.read_csv('diacwoz_data.csv', sep="\t")

    healthy_conversation = [text.replace('.', ' ') for text in df[df['label'] == 0]['text'].tolist()]
    depressed_conversation = [text.replace('.', ' ') for text in df[df['label'] == 1]['text'].tolist()]
    pos_results = run_emotion_mapping_analysis("Healthy", healthy_conversation)
    neg_results = run_emotion_mapping_analysis("Depressed", depressed_conversation)
    pos_mapper, pos_embeddings, pos_intensities, pos_assignments, pos_coords = pos_results if pos_results else (None,)*5
    neg_mapper, neg_embeddings, neg_intensities, neg_assignments, neg_coords = neg_results if neg_results else (None,)*5

    if pos_results and neg_results:
        all_region_names = list(pos_mapper.emotion_regions.keys())
        plot_comparison_charts(
            pos_intensities, 
            pos_assignments, 
            neg_intensities, 
            neg_assignments, 
            all_region_names
        )

    input("\nPress Enter to exit...")