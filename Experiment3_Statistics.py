# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import os
import sys
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

import numba
numba.config.DISABLE_JIT = True
numba.config.THREADING_LAYER = 'safe'
os.environ.update({
    'NUMBA_DISABLE_JIT': '1',
    'MNE_DISABLE_NUMBA': '1',
    'NUMBA_CACHE_DIR': '/tmp/numba_cache',
    'NUMBA_ENABLE_CUDASIM': '0'
})
# reload mne if numba was imported first
modules_to_remove = [k for k in sys.modules.keys() if 'numba' in k.lower()]
for mod in modules_to_remove:
    if mod in sys.modules:
        del sys.modules[mod]

from openai import OpenAI
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.utils import resample


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
    print(f"Human mean: {np.mean(pos_intensities_resampled):.4f} ± {np.std(pos_intensities_resampled):.4f}")
    print(f"LLM mean: {np.mean(neg_intensities_resampled):.4f} ± {np.std(neg_intensities_resampled):.4f}")
    print(f"Mann-Whitney U statistic: {u_stat:.4f}, p-value: {p_value:.6f}")
    print(f"Significant: {'YES' if p_value < 0.05 else 'NO'} (α = 0.05)")
    
    # Calculate regional means - this is the key output for visualization
    human_regional_means = []
    llm_regional_means = []
    significant_regions = []
    
    for region_idx in range(len(region_names)):
        pos_region_mask = pos_assignments_resampled == region_idx
        neg_region_mask = neg_assignments_resampled == region_idx
        pos_region_intensities = pos_intensities_resampled[pos_region_mask]
        neg_region_intensities = neg_intensities_resampled[neg_region_mask]
        
        if len(pos_region_intensities) > 0 and len(neg_region_intensities) > 0:
            human_mean = np.mean(pos_region_intensities)
            llm_mean = np.mean(neg_region_intensities)
            
            human_regional_means.append(human_mean)
            llm_regional_means.append(llm_mean)
            
            region_name = region_names[region_idx].replace('_', ' ').title()
            
            print(f"  {region_name}:")
            print(f"    Human: {len(pos_region_intensities)} texts, mean: {human_mean:.4f}")
            print(f"    LLM: {len(neg_region_intensities)} texts, mean: {llm_mean:.4f}")
            
            if len(pos_region_intensities) > 1 and len(neg_region_intensities) > 1:
                try:
                    u_stat_region, p_value_region = stats.mannwhitneyu(pos_region_intensities, neg_region_intensities, alternative='two-sided')
                    print(f"    Mann-Whitney U: {u_stat_region:.4f}, p-value: {p_value_region:.6f}")
                    
                    if p_value_region < 0.05:
                        significant_regions.append(region_name)
                        print(f"    *** SIGNIFICANT ***")
                except ValueError:
                    continue
        else:
            # Use baseline values for regions with no data
            human_regional_means.append(0.1)
            llm_regional_means.append(0.1)
    
    print(f"\nRegions with significant differences ({len(significant_regions)}):")
    for region in significant_regions:
        print(f"  - {region}")
    
    return np.array(human_regional_means), np.array(llm_regional_means), significant_regions


def get_ada_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    api_key = "your open-ai api key here"
    
    all_embeddings = []
    batch_size = 2000
    
    try:
        client = OpenAI(api_key=api_key)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        if not all_embeddings:
            return None
        return np.array(all_embeddings)
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
            'absolutely': 1.0, 'incredibly': 1.0, 'magnificently': 1.0, 'phenomenally': 1.0, 'ecstatic': 1.0, 'euphoric': 1.0,
            'elated': 1.0, 'jubilant': 1.0, 'divine': 1.0, 'wonderful': 1.0, 'happiness': 1.0, 'joy': 1.0, 'active': 1.0,
            'sports': 1.0, 'optimistic': 1.0, 'successful': 1.0, 'outgoing': 1.0, 'devastated': 1.0, 'destroyed': 1.0,
            'shattered': 1.0, 'furious': 1.0, 'enraged': 1.0, 'livid': 1.0, 'terrified': 1.0, 'horrifyingly': 1.0, 'depressed': 1.0,
            'psychiatrist': 1.0, 'exhausted': 1.0, 'crying': 1.0, 'extremely': 1.0, 'losing': 1.0, 'uncomfortable': 1.0, 'mental': 1.0,
            'amazing': 0.8, 'awesome': 0.8, 'fantastic': 0.8, 'superb': 0.8, 'excellent': 0.8, 'brilliantly': 0.8, 'hate': 0.8,
            'despise': 0.8, 'loathe': 0.8, 'terrible': 0.8, 'horrible': 0.8, 'awful': 0.8, 'dreadful': 0.8, 'love': 0.6, 'adore': 0.6,
            'happy': 0.6, 'glad': 0.6, 'pleased': 0.6, 'joyfully': 0.6, 'great': 0.6, 'encouragingly': 0.6, 'fortunately': 0.6,
            'sad': 0.6, 'unhappy': 0.6, 'disappointed': 0.6, 'frustrated': 0.6, 'angry': 0.6, 'miserably': 0.6, 'sadly': 0.6,
            'okay': 0.3, 'fine': 0.3, 'decent': 0.3, 'nice': 0.3, 'good': 0.3, 'cool': 0.3, 'bad': 0.3, 'poor': 0.3, 'weak': 0.3, 'boring': 0.3
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
    if not conversation:
        return None
    
    np.random.seed(42)
    embeddings = get_ada_embeddings(conversation)
    if embeddings is None:
        return None

    mapper = EmotionBrainMapper()
    brain_coords, region_assignments = mapper.fit_transform_embeddings(embeddings)
    emotion_intensities = mapper.estimate_emotion_intensity(conversation)
    
    print(f"\n{title} Emotion Intensities: {np.round(emotion_intensities, 2)}")
    return mapper, emotion_intensities, region_assignments


def create_differences_bar_plot(human_values: np.ndarray, llm_values: np.ndarray, 
                               region_names: List[str], significant_regions: List[str]):

    differences = human_values - llm_values
    SCALE_FACTOR = 50.0
    scaled_differences = differences * SCALE_FACTOR
    non_zero_mask = np.abs(scaled_differences) > 1e-6
    if not np.any(non_zero_mask):
        print("No non-zero differences found!")
        return None, None
    
    filtered_differences = scaled_differences[non_zero_mask]
    filtered_region_names = [region_names[i] for i in range(len(region_names)) if non_zero_mask[i]]
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
    ax.set_title('Statistical Analysis Results: Human vs LLM Brain Region Differences\n' +
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
    df = pd.read_csv('BrainEmbeddings/dstc8.csv')
    df['Text'] = df['Text'].str.replace('"', '', regex=False).str.strip()
    df.dropna(subset=['Text'], inplace=True)
    user_texts = df[df['Input'] == 'USER']['Text'].tolist()[1:10000]
    system_texts = df[df['Input'] == 'SYSTEM']['Text'].tolist()[1:10000]
    
    user_results = run_emotion_mapping_analysis("Human", user_texts)
    system_results = run_emotion_mapping_analysis("LLM", system_texts)
    user_mapper, user_intensities, user_assignments = user_results
    _, system_intensities, system_assignments = system_results
    
    all_region_names = list(user_mapper.emotion_regions.keys())
    human_regional_means, llm_regional_means, significant_regions = perform_statistical_tests(
        user_intensities, 
        user_assignments, 
        system_intensities, 
        system_assignments, 
        all_region_names
    )
    
    print(f"Generated regional means for {len(human_regional_means)} regions")
    print(f"Found {len(significant_regions)} statistically significant regions")
    print(f"\n=== VALUES FOR VISUALIZATION CODE ===")
    print("human_values = np.array([")
    print("    " + ", ".join([f"{x:.6f}" for x in human_regional_means]))
    print("])")
    print("llm_values = np.array([")
    print("    " + ", ".join([f"{x:.6f}" for x in llm_regional_means]))
    print("])")
    
    user_emotion_range = np.sum(human_regional_means > 0.1)
    system_emotion_range = np.sum(llm_regional_means > 0.1)
    user_avg_intensity = np.mean(user_intensities)
    system_avg_intensity = np.mean(system_intensities)
    
    print(f"\n=== EMOTIONALITY ANALYSIS: HUMAN vs. LLM ===")
    print(f"Emotion Range (Active Regions Above Baseline):")
    print(f"  - Human: {user_emotion_range} regions")
    print(f"  - LLM:   {system_emotion_range} regions")
    print("Average Emotion Intensity:")
    print(f"  - Human: {user_avg_intensity:.3f}")
    print(f"  - LLM:   {system_avg_intensity:.3f}")

    fig, ax = create_differences_bar_plot(human_regional_means, llm_regional_means, 
                                        all_region_names, significant_regions)

    if fig is not None:
        print(f"\nStatistically significant regions ({len(significant_regions)}):")
        for region in significant_regions:
            print(f"  - {region}")
    else:
        print("No significant differences found to visualize.")


if __name__ == "__main__":
    main()