# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.utils import resample


def get_ada_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    api_key = "your open-ai key here"

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
            # Strong Positive (Healthy)
            'ecstatic': 1.0, 'euphoric': 1.0, 'elated': 1.0, 'jubilant': 1.0, 'thrilled': 1.0,
            'radiant': 1.0, 'joy': 1.0, 'passionate': 1.0, 'laughing': 1.0, 'motivated': 1.0,
            'grateful': 0.9, 'proud': 0.9, 'accomplished': 0.9,

            # Strong Negative (Depressed)
            'depressed': 1.2, 'suicidal': 1.2, 'worthless': 1.1, 'hopeless': 1.1, 'exhausted': 1.0,
            'isolated': 1.0, 'crying': 1.0, 'numb': 1.0, 'anxious': 0.9, 'fatigue': 0.9,
            'useless': 1.1, 'empty': 1.0, 'alone': 1.0, 'miserable': 1.0,

            # Moderate Positive
            'happy': 0.8, 'love': 0.8, 'excited': 0.8, 'smiling': 0.8, 'hopeful': 0.7, 'peaceful': 0.7,

            # Moderate Negative
            'sad': 0.8, 'angry': 0.8, 'upset': 0.7, 'irritated': 0.7, 'stressed': 0.7, 
            'worried': 0.7, 'nervous': 0.7, 'lonely': 0.8,

            # Mild Positive
            'okay': 0.3, 'fine': 0.3, 'good': 0.4, 'cool': 0.3, 'calm': 0.3,

            # Mild Negative
            'bad': 0.4, 'tired': 0.4, 'meh': 0.3, 'bored': 0.3, 'blah': 0.3,

            # Intensifiers
            'absolutely': 0.3, 'incredibly': 0.3, 'completely': 0.3, 'extremely': 0.3,
            'totally': 0.3, 'really': 0.2, 'very': 0.2
        }

        polarity_neg = {'depressed', 'hopeless', 'suicidal', 'crying', 'fatigue', 'worthless', 'alone', 'empty'}
        polarity_pos = {'ecstatic', 'joy', 'grateful', 'proud', 'hopeful', 'happy', 'love'}

        intensities = []
        for text in texts:
            intensity = 0.1  # Base value
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)

            word_intensity = 0.0
            polarity_score = 0.0

            for word in words:
                word_score = word_scores.get(word, 0.0)
                word_intensity += word_score

                if word in polarity_pos:
                    polarity_score += 0.5
                elif word in polarity_neg:
                    polarity_score -= 0.5

            if -0.5 < polarity_score < 0.5 and word_intensity > 0.5:
                intensity *= 0.75  # dampen ambiguous emotion
            
            intensities.append(min(intensity, 2.0))
        return np.array(intensities)


def run_emotion_mapping_analysis(title: str, conversation: List[str]):
    np.random.seed(42)
    embeddings = get_ada_embeddings(conversation)
    if embeddings is None:
        return None
    
    mapper = EmotionBrainMapper()
    brain_coords, region_assignments = mapper.fit_transform_embeddings(embeddings)
    emotion_intensities = mapper.estimate_emotion_intensity(conversation)
    print(f"\n{title} Emotion Intensities: {np.round(emotion_intensities, 2)}")
    return mapper, embeddings, emotion_intensities, region_assignments, brain_coords


def perform_statistical_tests(pos_intensities, pos_assignments, neg_intensities, neg_assignments, region_names):
    # Oversample the minority class
    if len(pos_intensities) < len(neg_intensities):
        pos_intensities_resampled = resample(pos_intensities, 
                                           replace=True, 
                                           n_samples=len(neg_intensities), 
                                           random_state=42)
        pos_assignments_resampled = resample(pos_assignments, 
                                           replace=True, 
                                           n_samples=len(neg_assignments), 
                                           random_state=42)
        neg_intensities_resampled = neg_intensities
        neg_assignments_resampled = neg_assignments
    else:
        neg_intensities_resampled = resample(neg_intensities, 
                                           replace=True, 
                                           n_samples=len(pos_intensities), 
                                           random_state=42)
        neg_assignments_resampled = resample(neg_assignments, 
                                           replace=True, 
                                           n_samples=len(pos_assignments), 
                                           random_state=42)
        pos_intensities_resampled = pos_intensities
        pos_assignments_resampled = pos_assignments
    
    # Calculate overall statistics
    u_stat, p_value = stats.mannwhitneyu(pos_intensities_resampled, neg_intensities_resampled, alternative='two-sided')
    print(f"Healthy mean: {np.mean(pos_intensities_resampled):.4f} ± {np.std(pos_intensities_resampled):.4f}")
    print(f"Depressed mean: {np.mean(neg_intensities_resampled):.4f} ± {np.std(neg_intensities_resampled):.4f}")
    print(f"Mann-Whitney U statistic: {u_stat:.4f}, p-value: {p_value:.6f}")
    print(f"Significant: {'YES' if p_value < 0.05 else 'NO'} (alpha = 0.05)")
    
    # Calculate regional means - this is the key output for visualization
    healthy_regional_means = []
    depressed_regional_means = []
    significant_regions = []
    
    for region_idx in range(len(region_names)):
        pos_region_mask = pos_assignments_resampled == region_idx
        neg_region_mask = neg_assignments_resampled == region_idx
        pos_region_intensities = pos_intensities_resampled[pos_region_mask]
        neg_region_intensities = neg_intensities_resampled[neg_region_mask]
        
        if len(pos_region_intensities) > 0 and len(neg_region_intensities) > 0:
            healthy_mean = np.mean(pos_region_intensities)
            depressed_mean = np.mean(neg_region_intensities)
            healthy_regional_means.append(healthy_mean)
            depressed_regional_means.append(depressed_mean)
            region_name = region_names[region_idx].replace('_', ' ').title()
            print(f"{region_name}:")
            print(f"  Healthy: {len(pos_region_intensities)} texts, mean: {healthy_mean:.4f}")
            print(f"  Depressed: {len(neg_region_intensities)} texts, mean: {depressed_mean:.4f}")
            
            if len(pos_region_intensities) > 1 and len(neg_region_intensities) > 1:
                try:
                    u_stat_region, p_value_region = stats.mannwhitneyu(pos_region_intensities, neg_region_intensities, alternative='two-sided')
                    print(f"  Mann-Whitney U: {u_stat_region:.4f}, p-value: {p_value_region:.6f}")
                    if p_value_region < 0.05:
                        significant_regions.append(region_name)
                        print(f"  *** SIGNIFICANT ***")
                except ValueError:
                    continue
        else:
            # Use baseline values for regions with no data
            healthy_regional_means.append(0.1)
            depressed_regional_means.append(0.1)
    
    print(f"Regions with significant differences ({len(significant_regions)}):")
    for region in significant_regions:
        print(f"  - {region}")
    
    return np.array(healthy_regional_means), np.array(depressed_regional_means), significant_regions


def create_differences_bar_plot(healthy_values: np.ndarray, depressed_values: np.ndarray, 
                               region_names: List[str], significant_regions: List[str]):
    differences = healthy_values - depressed_values
    SCALE_FACTOR = 50.0  # Amplify differences for visibility
    scaled_differences = differences * SCALE_FACTOR
    non_zero_mask = np.abs(scaled_differences) > 1e-6
    if not np.any(non_zero_mask):
        print("No non-zero differences found!")
        return None, None
    
    filtered_differences = scaled_differences[non_zero_mask]
    filtered_region_names = [region_names[i] for i in range(len(region_names)) if non_zero_mask[i]]
    print(f"Showing {len(filtered_differences)} regions with non-zero differences (out of {len(region_names)} total)")

    x_pos = np.arange(len(filtered_differences))
    fig, ax = plt.subplots(figsize=(18, 10))
    colors = []
    for i, region_name in enumerate(filtered_region_names):
        region_title = region_name.replace('_', ' ').title()
        if region_title in significant_regions:
            colors.append('darkgreen' if filtered_differences[i] > 0 else 'darkred')
        else:
            colors.append('lightgreen' if filtered_differences[i] > 0 else 'lightcoral')
    
    bars = ax.bar(x_pos, filtered_differences, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xlabel('Brain Region', fontsize=16, color='black', fontweight='bold')
    ax.set_ylabel(f'Scaled Differences (×{SCALE_FACTOR})', fontsize=16, color='black', fontweight='bold')
    ax.set_title('Statistical Analysis Results: Healthy vs Depressed Brain Region Differences\n' +
                f'{len(filtered_differences)} Regions with Non-Zero Differences (Dark colors = statistically significant, p<0.05)', 
                fontsize=18, color='black', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.replace('_', ' ').title() for name in filtered_region_names], 
                       rotation=45, ha='right', fontsize=12, color='black', fontweight='bold')
    ax.tick_params(axis='y', labelsize=12, colors='black')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(False)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + (0.002 if height > 0 else -0.004),
                f'{filtered_differences[i]:.3f}', ha='center', 
                va='bottom' if height > 0 else 'top', 
                fontsize=10, color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    return fig, ax


def main():
    df = pd.read_csv('BrainEmbeddings/diacwoz_data.csv', sep="\t")
    healthy_conversation = [text.replace('.', ' ') for text in df[df['label'] == 0]['text'].tolist()]
    depressed_conversation = [text.replace('.', ' ') for text in df[df['label'] == 1]['text'].tolist()]

    pos_results = run_emotion_mapping_analysis("Healthy", healthy_conversation)
    neg_results = run_emotion_mapping_analysis("Depressed", depressed_conversation)
    pos_mapper, _, pos_intensities, pos_assignments, _ = pos_results
    _, _, neg_intensities, neg_assignments, _ = neg_results
    all_region_names = list(pos_mapper.emotion_regions.keys())
    healthy_regional_means, depressed_regional_means, significant_regions = perform_statistical_tests(
        pos_intensities, 
        pos_assignments, 
        neg_intensities, 
        neg_assignments, 
        all_region_names
    )

    print(f"Generated regional means for {len(healthy_regional_means)} regions")
    print(f"Found {len(significant_regions)} statistically significant regions")
    
    # These are the values that should replace the hardcoded arrays in the visualization
    print("healthy_values = np.array([")
    print("    " + ", ".join([f"{x:.6f}" for x in healthy_regional_means]))
    print("])")
    print("depressed_values = np.array([")
    print("    " + ", ".join([f"{x:.6f}" for x in depressed_regional_means]))
    print("])")
    
    fig, ax = create_differences_bar_plot(healthy_regional_means, depressed_regional_means, 
                                        all_region_names, significant_regions)
    if fig is not None:
        print(f"Statistically significant regions ({len(significant_regions)}):")
        for region in significant_regions:
            print(f"  - {region}")
    else:
        print("No significant differences found to visualize.")

if __name__ == "__main__":
    main()