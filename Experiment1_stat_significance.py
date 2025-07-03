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
    """Generates text embeddings using OpenAI's text-embedding-ada-002 model."""
    api_key = "your api key here"
    if not api_key:
        print("Error: OpenAI API key not found.")
        print("Please set the OPENAI_API_KEY environment variable.")
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
    """
    Handles the logic for mapping text embeddings to brain regions and visualizing them.
    """
    def __init__(self):
        self.emotion_regions = self.define_emotion_regions()
        self.n_emotion_regions = len(self.emotion_regions)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=self.n_emotion_regions, random_state=42, n_init=10)
        self.brain = None

    def define_emotion_regions(self) -> Dict[str, List[float]]:
        """
        Defines key emotional brain regions with their MNI coordinates.
        Coordinates are based on neuroimaging atlases and converted to MNE coordinate system.
        MNE uses RAS+ coordinate system: Right(+X), Anterior(+Y), Superior(+Z)
        """
        return {
            # Amygdala - corrected coordinates in mm
            'amygdala_left': [-20, -5, -18],
            'amygdala_right': [20, -5, -18],
            
            # Anterior Cingulate Cortex - more anterior and superior
            'anterior_cingulate_left': [-5, 25, 25],
            'anterior_cingulate_right': [5, 25, 25],
            
            # Insula - more lateral and in correct position
            'insula_left': [-40, 8, 0],
            'insula_right': [40, 8, 0],
            
            # Orbitofrontal Cortex - more anterior and inferior
            'orbitofrontal_left': [-25, 40, -15],
            'orbitofrontal_right': [25, 40, -15],
            
            # Hippocampus - corrected position
            'hippocampus_left': [-25, -15, -20],
            'hippocampus_right': [25, -15, -20],
            
            # Dorsolateral Prefrontal Cortex - more lateral and anterior
            'prefrontal_cortex_left': [-45, 30, 30],
            'prefrontal_cortex_right': [45, 30, 30],
            
            # Temporal Pole - CORRECTED: more anterior and lateral
            'temporal_pole_left': [-40, 20, -25],
            'temporal_pole_right': [40, 20, -25],
            
            # Superior Temporal Gyrus - for comparison
            'superior_temporal_left': [-55, -25, 10],
            'superior_temporal_right': [55, -25, 10],
            
            # Caudate - corrected position
            'caudate_left': [-12, 12, 8],
            'caudate_right': [12, 12, 8],
            
            # Putamen - more lateral
            'putamen_left': [-25, 5, 0],
            'putamen_right': [25, 5, 0],
            
            # Nucleus Accumbens - ventral striatum
            'nucleus_accumbens_left': [-8, 8, -8],
            'nucleus_accumbens_right': [8, 8, -8],
            
            # Midline structures
            'hypothalamus': [0, -2, -15],
            'periaqueductal_gray': [0, -28, -10],
            'ventral_tegmental_area': [0, -15, -20],
            'raphe_nuclei': [0, -25, -30],
            'locus_coeruleus': [0, -37, -28],
            'posterior_cingulate': [0, -50, 25],
            'medial_prefrontal_cortex': [0, 45, 20]
        }

    def fit_transform_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms embeddings and maps them to the nearest emotion region.
        
        This logic first runs K-Means on the embeddings to find emotional
        cluster centers in 3D space. It then assigns each embedding to its
        closest cluster. Finally, it maps these cluster assignments to the predefined
        anatomical region coordinates for visualization.
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided.")
        
        # Determine appropriate number of PCA components
        n_samples, n_features = embeddings.shape
        n_components = min(3, n_samples, n_features)
        
        if n_components < 3:
            print(f"Warning: Only {n_components} PCA components available (samples: {n_samples}, features: {n_features})")
            self.pca = PCA(n_components=n_components)
            
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        embeddings_3d = self.pca.fit_transform(embeddings_scaled)
        
        # If we have fewer than 3 dimensions, pad with zeros
        if embeddings_3d.shape[1] < 3:
            padding = np.zeros((embeddings_3d.shape[0], 3 - embeddings_3d.shape[1]))
            embeddings_3d = np.hstack([embeddings_3d, padding])
        
        # Adjust k-means clusters if we have fewer samples than regions
        n_clusters = min(self.n_emotion_regions, embeddings.shape[0])
        if n_clusters != self.n_emotion_regions:
            print(f"Warning: Reducing clusters from {self.n_emotion_regions} to {n_clusters} due to insufficient samples")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            kmeans = self.kmeans
            
        # Fit KMeans to the 3D embeddings to find cluster centers
        kmeans.fit(embeddings_3d)
        cluster_centers_3d = kmeans.cluster_centers_

        # For each embedding, find the index of the closest cluster center
        # This determines which emotional "category" the text belongs to.
        region_assignments = np.argmin(cdist(embeddings_3d, cluster_centers_3d), axis=1)

        # Map the assigned cluster to the predefined anatomical coordinates
        region_coords_map = np.array(list(self.emotion_regions.values()), dtype=np.float64)

        # Map each cluster center to the closest brain region
        assigned_regions = []
        used_indices = set()
        for center in cluster_centers_3d:
            dists = cdist([center], region_coords_map)[0]
            for idx in np.argsort(dists):
                if idx not in used_indices:
                    assigned_regions.append(idx)
                    used_indices.add(idx)
                    break

        # Now map each embedding's assigned cluster to a brain region
        cluster_to_region = dict(zip(range(len(assigned_regions)), assigned_regions))
        region_assignments_mapped = np.array([cluster_to_region[c] for c in region_assignments])
        brain_coordinates = region_coords_map[region_assignments_mapped].copy()

        return brain_coordinates, region_assignments

    def estimate_emotion_intensity(self, texts: List[str]) -> np.ndarray:
        """Estimates emotional intensity of texts based on emotional vocabulary and syntax."""

        # Refined and expanded emotional vocabulary
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

            # Penalize mixed polarity (neutral emotion is less intense)
            if -0.5 < polarity_score < 0.5 and word_intensity > 0.5:
                intensity *= 0.75  # dampen ambiguous emotion
            
            intensities.append(min(intensity, 2.0))

        return np.array(intensities)


def run_emotion_mapping_analysis(title: str, conversation: List[str]):

    np.random.seed(42)
    embeddings = get_ada_embeddings(conversation)
    if embeddings is None:
        print(f"Analysis for '{title}' skipped due to embedding generation failure.\n")
        return None, None, None, None

    mapper = EmotionBrainMapper()
    
    # --- Perform mapping and intensity calculations ONCE ---
    brain_coords, region_assignments = mapper.fit_transform_embeddings(embeddings)
    emotion_intensities = mapper.estimate_emotion_intensity(conversation)

    print(f"\n{title} Emotion Intensities: {np.round(emotion_intensities, 2)}")
    print(f"Range: {np.min(emotion_intensities):.3f} - {np.max(emotion_intensities):.3f}")

    # --- Print Summary Report ---
    region_names = list(mapper.emotion_regions.keys())
    print(f"\n--- {title.upper()} MAPPING SUMMARY ---")
    for i in range(mapper.n_emotion_regions):
        # Find which texts were assigned to this cluster
        assigned_indices = np.where(region_assignments == i)[0]
        count = len(assigned_indices)
        region_name = region_names[i]
        
        if count > 0:
            region_intensities = emotion_intensities[assigned_indices]
            avg_intensity = np.mean(region_intensities)
            print(f"  > {region_name}: {count} emotions (avg intensity: {avg_intensity:.3f})")

    print(f"\n--- {title.upper()} DETAILED ANALYSIS ---")
    for i, (text, intensity, region_idx) in enumerate(zip(conversation, emotion_intensities, region_assignments)):
        region_name = region_names[region_idx]
        emotion_type = "High" if intensity > 1.0 else "Medium" if intensity > 0.5 else "Low"
        print(f"{i+1:2d}. {emotion_type} ({intensity:.3f}) -> {region_name}")

    print("-" * 50 + "\n")
    return mapper, embeddings, emotion_intensities, region_assignments, brain_coords


if __name__ == "__main__":
    df = pd.read_csv('BrainEmbeddings/diacwoz_data.csv', sep="\t")

    # Extract healthy subject conversations (label = 0)
    healthy_conversation = [text.replace('.', ' ') for text in df[df['label'] == 0]['text'].tolist()]
    # Extract depressed subject conversations (label = 1)
    depressed_conversation = [text.replace('.', ' ') for text in df[df['label'] == 1]['text'].tolist()]

    pos_results = run_emotion_mapping_analysis("Healthy", healthy_conversation)
    neg_results = run_emotion_mapping_analysis("Depressed", depressed_conversation)
    pos_mapper, pos_embeddings, pos_intensities, pos_assignments, pos_coords = pos_results if pos_results else (None,)*5
    neg_mapper, neg_embeddings, neg_intensities, neg_assignments, neg_coords = neg_results if neg_results else (None,)*5

    print("-" * 60)
    if pos_intensities is not None:
        print(f"HEALTHY emotions - Average intensity: {np.mean(pos_intensities):.3f}")
    if neg_intensities is not None:
        print(f"DEPRESSED emotions - Average intensity: {np.mean(neg_intensities):.3f}")

if pos_results and neg_results:
    all_region_names = list(pos_mapper.emotion_regions.keys())

    # Perform statistical tests
    pos_intensities_resampled, neg_intensities_resampled = perform_statistical_tests(
        pos_intensities, 
        pos_assignments, 
        neg_intensities, 
        neg_assignments, 
        all_region_names
    )
