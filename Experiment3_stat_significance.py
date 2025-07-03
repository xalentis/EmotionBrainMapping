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

# This block attempts to mitigate potential conflicts between numba and other libraries
# on certain systems. It's a workaround and might not be necessary in all environments.
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
    """
    Performs statistical significance tests between healthy and depressed groups.
    Uses oversampling to handle class imbalance.
    """
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    
    # Oversample the minority class
    if len(pos_intensities) < len(neg_intensities):
        # Oversample healthy (positive) group
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
        # Oversample depressed (negative) group
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
    
    print(f"Original sizes - Healthy: {len(pos_intensities)}, Depressed: {len(neg_intensities)}")
    print(f"Resampled sizes - Healthy: {len(pos_intensities_resampled)}, Depressed: {len(neg_intensities_resampled)}")
    
    # Overall intensity comparison - use Mann-Whitney U test for robustness
    u_stat, p_value = stats.mannwhitneyu(pos_intensities_resampled, neg_intensities_resampled, alternative='two-sided')
    print(f"\nOVERALL INTENSITY COMPARISON:")
    print(f"Healthy mean: {np.mean(pos_intensities_resampled):.4f} ± {np.std(pos_intensities_resampled):.4f}")
    print(f"Depressed mean: {np.mean(neg_intensities_resampled):.4f} ± {np.std(neg_intensities_resampled):.4f}")
    print(f"Mann-Whitney U statistic: {u_stat:.4f}, p-value: {p_value:.6f}")
    print(f"Significant: {'YES' if p_value < 0.05 else 'NO'} (α = 0.05)")
    
    # Per-region analysis
    print(f"\nPER-REGION ANALYSIS:")
    significant_regions = []
    
    for region_idx in range(len(region_names)):
        # Get intensities for this region
        pos_region_mask = pos_assignments_resampled == region_idx
        neg_region_mask = neg_assignments_resampled == region_idx
        
        pos_region_intensities = pos_intensities_resampled[pos_region_mask]
        neg_region_intensities = neg_intensities_resampled[neg_region_mask]
        
        # Only test if both groups have data for this region
        if len(pos_region_intensities) > 1 and len(neg_region_intensities) > 1:
            # Use Mann-Whitney U test for robustness against precision issues
            try:
                u_stat_region, p_value_region = stats.mannwhitneyu(pos_region_intensities, neg_region_intensities, alternative='two-sided')
                region_name = region_names[region_idx].replace('_', ' ').title()
                
                print(f"  {region_name}:")
                print(f"    Healthy: {len(pos_region_intensities)} texts, mean: {np.mean(pos_region_intensities):.4f}")
                print(f"    Depressed: {len(neg_region_intensities)} texts, mean: {np.mean(neg_region_intensities):.4f}")
                print(f"    Mann-Whitney U: {u_stat_region:.4f}, p-value: {p_value_region:.6f}")
                
                if p_value_region < 0.05:
                    significant_regions.append(region_name)
                    print(f"    *** SIGNIFICANT ***")
            except ValueError:
                # Skip regions with identical values
                continue
    
    print(f"\nSUMMARY:")
    print(f"Regions with significant differences ({len(significant_regions)}):")
    for region in significant_regions:
        print(f"  - {region}")
    
    return pos_intensities_resampled, neg_intensities_resampled


def get_ada_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    api_key = "your api key here"
    if not api_key:
        print("Error: OpenAI API key not found.")
        print("Please set the OPENAI_API_KEY environment variable.")
        return None
    
    all_embeddings = []
    batch_size = 2000
    
    try:
        client = OpenAI(api_key=api_key)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}...")
            
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
        if 'max_tokens_per_request' in str(e):
             print("Adjust or chunk the text to fix this error.")
        return None


class EmotionBrainMapper:
    def __init__(self, n_emotion_regions=25):
        self.n_emotion_regions = n_emotion_regions
        self.emotion_regions = self.define_emotion_regions()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=n_emotion_regions, random_state=42, n_init=10)

    def define_emotion_regions(self) -> Dict[str, List[float]]:
        """
        Defines key emotional brain regions with their CORRECTED MNI coordinates.
        """
        return {
            # Amygdala - corrected coordinates in mm
            'amygdala_left': [-20, -5, -18], 'amygdala_right': [20, -5, -18],
            # Anterior Cingulate Cortex
            'anterior_cingulate_left': [-5, 25, 25], 'anterior_cingulate_right': [5, 25, 25],
            # Insula
            'insula_left': [-40, 8, 0], 'insula_right': [40, 8, 0],
            # Orbitofrontal Cortex
            'orbitofrontal_left': [-25, 40, -15], 'orbitofrontal_right': [25, 40, -15],
            # Hippocampus
            'hippocampus_left': [-25, -15, -20], 'hippocampus_right': [25, -15, -20],
            # Dorsolateral Prefrontal Cortex
            'prefrontal_cortex_left': [-45, 30, 30], 'prefrontal_cortex_right': [45, 30, 30],
            # Temporal Pole
            'temporal_pole_left': [-40, 20, -25], 'temporal_pole_right': [40, 20, -25],
            # Superior Temporal Gyrus
            'superior_temporal_left': [-55, -25, 10], 'superior_temporal_right': [55, -25, 10],
            # Caudate
            'caudate_left': [-12, 12, 8], 'caudate_right': [12, 12, 8],
            # Putamen
            'putamen_left': [-25, 5, 0], 'putamen_right': [25, 5, 0],
            # Nucleus Accumbens
            'nucleus_accumbens_left': [-8, 8, -8], 'nucleus_accumbens_right': [8, 8, -8],
            # Midline structures
            'hypothalamus': [0, -2, -15], 'periaqueductal_gray': [0, -28, -10],
            'ventral_tegmental_area': [0, -15, -20], 'raphe_nuclei': [0, -25, -30],
            'locus_coeruleus': [0, -37, -28], 'posterior_cingulate': [0, -50, 25],
            'medial_prefrontal_cortex': [0, 45, 20]
        }

    def fit_transform_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms embeddings and maps them to the nearest emotion region.
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided.")
        
        n_samples, n_features = embeddings.shape
        n_components = min(3, n_samples, n_features)
        
        if n_components < 3:
            print(f"Warning: Only {n_components} PCA components available (samples: {n_samples}, features: {n_features})")
            self.pca = PCA(n_components=n_components)
            
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        embeddings_3d = self.pca.fit_transform(embeddings_scaled)
        
        if embeddings_3d.shape[1] < 3:
            padding = np.zeros((embeddings_3d.shape[0], 3 - embeddings_3d.shape[1]))
            embeddings_3d = np.hstack([embeddings_3d, padding])
        
        n_clusters = min(self.n_emotion_regions, embeddings.shape[0])
        if n_clusters != self.n_emotion_regions:
            print(f"Warning: Reducing clusters from {self.n_emotion_regions} to {n_clusters} due to insufficient samples")
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
        """Estimates emotional intensity of texts based on keywords and syntax."""
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
    print(f"--- {title.upper()} EMOTION MAPPING ANALYSIS ---")
    if not conversation:
        print(f"No text found for '{title}'. Skipping analysis.\n")
        return None, None, None, None, None

    np.random.seed(42)
    embeddings = get_ada_embeddings(conversation)
    if embeddings is None:
        print(f"Analysis for '{title}' skipped due to embedding generation failure.\n")
        return None, None, None, None, None

    mapper = EmotionBrainMapper(n_emotion_regions=25)
    
    brain_coords, region_assignments = mapper.fit_transform_embeddings(embeddings)
    emotion_intensities = mapper.estimate_emotion_intensity(conversation)

    print(f"\n{title} Emotion Intensities (Sample): {np.round(emotion_intensities[:10], 2)}")
    print(f"Range: {np.min(emotion_intensities):.3f} - {np.max(emotion_intensities):.3f}")

    region_names = list(mapper.emotion_regions.keys())
    print(f"\n--- {title.upper()} MAPPING SUMMARY ---")
    
    region_summary = {name: {'count': 0, 'intensities': []} for name in region_names}
    for i, region_idx in enumerate(region_assignments):
        region_name = region_names[region_idx]
        region_summary[region_name]['count'] += 1
        region_summary[region_name]['intensities'].append(emotion_intensities[i])

    for region_name, data in sorted(region_summary.items()):
        count = data['count']
        if count > 0:
            avg_intensity = np.mean(data['intensities'])
            print(f"  > {region_name}: {count} instance(s) (avg intensity: {avg_intensity:.3f})")

    print("-" * 50 + "\n")
    return mapper, embeddings, emotion_intensities, region_assignments, region_summary


def plot_brain_region_activations(user_summary, system_summary):
    user_data = {region: data['count'] for region, data in user_summary.items()}
    system_data = {region: data['count'] for region, data in system_summary.items()}
    df_user = pd.DataFrame(list(user_data.items()), columns=['Region', 'USER'])
    df_system = pd.DataFrame(list(system_data.items()), columns=['Region', 'SYSTEM'])
    df = pd.merge(df_user, df_system, on='Region', how='outer').fillna(0)
    df['Region'] = df['Region'].str.replace('_', ' ').str.title()
    df = df[(df['USER'] > 0) | (df['SYSTEM'] > 0)]
    df = df.sort_values(by='USER', ascending=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 14))
    y_pos = np.arange(len(df['Region']))
    height = 0.35
    bar1 = ax.barh(y_pos - height / 2, df['USER'], height, label='USER', color="#1492E6")
    bar2 = ax.barh(y_pos + height / 2, df['SYSTEM'], height, label='SYSTEM', color="#33DA57")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Region'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Number of Activations (Instances)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brain Region', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Brain Region Activations: USER vs. SYSTEM', fontsize=16, fontweight='bold')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.grid(axis='y', linestyle='', alpha=0.7)
    plt.tight_layout()
    plt.savefig('region_activations.png')
    plt.show()


if __name__ == "__main__":
    try:
        df = pd.read_csv('BrainEmbeddings/dstc8.csv')
        df['Text'] = df['Text'].str.replace('"', '', regex=False).str.strip()
        df.dropna(subset=['Text'], inplace=True)
        user_texts = df[df['Input'] == 'USER']['Text'].tolist()[1:10000]
        system_texts = df[df['Input'] == 'SYSTEM']['Text'].tolist()[1:10000]
        print(f"Loaded {len(user_texts)} USER texts and {len(system_texts)} SYSTEM texts.")
    except FileNotFoundError:
        print("Error: DSTC8.csv not found. Please ensure the file is in the correct directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        sys.exit(1)

    user_results = run_emotion_mapping_analysis("USER", user_texts)
    _, _, user_intensities, _, user_summary = user_results if user_results else (None,) * 5
    system_results = run_emotion_mapping_analysis("SYSTEM", system_texts)
    _, _, system_intensities, _, system_summary = system_results if system_results else (None,) * 5

    pos_mapper, pos_embeddings, pos_intensities, pos_assignments, pos_coords = user_results if user_results else (None,)*5
    neg_mapper, neg_embeddings, neg_intensities, neg_assignments, neg_coords = system_results if system_results else (None,)*5

    if user_summary and system_summary and user_intensities is not None and system_intensities is not None:

        all_region_names = list(pos_mapper.emotion_regions.keys())

        # Perform statistical tests
        pos_intensities_resampled, neg_intensities_resampled = perform_statistical_tests(
            pos_intensities, 
            pos_assignments, 
            neg_intensities, 
            neg_assignments, 
            all_region_names
        )

        user_emotion_range = sum(1 for data in user_summary.values() if data['count'] > 0)
        system_emotion_range = sum(1 for data in system_summary.values() if data['count'] > 0)
        user_avg_intensity = np.mean(user_intensities)
        system_avg_intensity = np.mean(system_intensities)
        print("\n" + "="*60)
        print("EMOTIONALITY ANALYSIS: USER vs. SYSTEM")
        print("="*60)
        print(f"Emotion Range (Unique Regions Activated):")
        print(f"  - USER:   {user_emotion_range} regions")
        print(f"  - SYSTEM: {system_emotion_range} regions")
        print("\nAverage Emotion Intensity:")
        print(f"  - USER:   {user_avg_intensity:.3f}")
        print(f"  - SYSTEM: {system_avg_intensity:.3f}")
        print("="*60)

        user_metrics = {'range': user_emotion_range, 'intensity': user_avg_intensity}
        system_metrics = {'range': system_emotion_range, 'intensity': system_avg_intensity}
        plot_brain_region_activations(user_summary, system_summary)

