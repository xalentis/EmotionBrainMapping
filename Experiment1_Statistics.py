
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
from statsmodels.stats.multitest import multipletests


def get_ada_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    api_key = "your open-ai api key here"
    if not api_key or api_key == "your open-ai api key here":
        print("Error: Missing OpenAI API Key...")
        return None

    BATCH_SIZE = 2048
    all_embeddings = []

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

    try:
        client = OpenAI(api_key=api_key)
        
        for i in range(0, len(texts_to_embed), BATCH_SIZE):
            batch = texts_to_embed[i:i + BATCH_SIZE]
            print(f"Embedding batch {i // BATCH_SIZE + 1} / {len(texts_to_embed) // BATCH_SIZE + 1}...")
            
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    except Exception as e:
        print(f"Error generating ADA embeddings: {e}")
        return None


def chunk_conversations(transcripts: List[str], chunk_size: int = 300) -> List[str]:
    """
    Segments a list of long transcripts into a list of smaller text chunks,
    as per the methodology in manuscript Section 2.2.
    This is critical for the intensity function to work as intended.
    """
    all_chunks = []
    for transcript in transcripts:
        cleaned_text = transcript.replace('.', ' ')
        words = cleaned_text.split()
        
        if not words:
            continue

        current_chunk = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                all_chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            all_chunks.append(" ".join(current_chunk))
            
    return all_chunks


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
        print(f"Failed to get embeddings for {title}")
        return None
    
    mapper = EmotionBrainMapper()
    brain_coords, region_assignments = mapper.fit_transform_embeddings(embeddings)
    emotion_intensities = mapper.estimate_emotion_intensity(conversation)
    return mapper, embeddings, emotion_intensities, region_assignments, brain_coords


def perform_statistical_tests(pos_intensities, pos_assignments, neg_intensities, neg_assignments, region_names):
    
    raw_p_values = []
    region_names_tested = []
    
    for region_idx in range(len(region_names)):
        pos_region_mask = pos_assignments == region_idx
        neg_region_mask = neg_assignments == region_idx
        
        pos_region_intensities = pos_intensities[pos_region_mask]
        neg_region_intensities = neg_intensities[neg_region_mask]
        
        region_name = region_names[region_idx]
        region_names_tested.append(region_name)
        
        if len(pos_region_intensities) > 1 and len(neg_region_intensities) > 1:
            try:
                u_stat_region, p_value_region = stats.mannwhitneyu(pos_region_intensities, 
                                                                 neg_region_intensities, 
                                                                 alternative='two-sided')
                raw_p_values.append(p_value_region)
            except ValueError as e:
                # This can happen if all values in one group are identical
                print(f"Skipping {region_name}: {e}")
                raw_p_values.append(1.0)
        else:
            # If not enough data in one of the groups, it's not significant
            raw_p_values.append(1.0)

    reject_bonf, p_corrected_bonf, _, _ = multipletests(raw_p_values, alpha=0.05, method='bonferroni')
    
    # FDR (Benjamini/Hochberg): Less strict, better for exploratory work
    reject_fdr, p_corrected_fdr, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')

    print("\nStatistical Analysis of Imbalanced Data")
    print(f"{'Brain Region':<28} | {'Raw p-value':<12} | {'Bonferroni p':<14} | {'FDR (B-H) p':<12} | {'Sig (FDR < 0.05)':<10}")
    
    significant_count_fdr = 0
    for i in range(len(region_names_tested)):
        region_title = region_names_tested[i].replace('_', ' ').title()
        is_sig_fdr = "YES" if reject_fdr[i] else "no"
        if reject_fdr[i]:
            significant_count_fdr += 1
            
        print(f"{region_title:<28} | {raw_p_values[i]:<12.6f} | {p_corrected_bonf[i]:<14.6f} | {p_corrected_fdr[i]:<12.6f} | {is_sig_fdr:<10}")

    print(f"Total regions found significant after FDR correction: {significant_count_fdr}")

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
    
    u_stat, p_value = stats.mannwhitneyu(pos_intensities_resampled, neg_intensities_resampled, alternative='two-sided')
    print(f"Healthy mean (resampled): {np.mean(pos_intensities_resampled):.4f} ± {np.std(pos_intensities_resampled):.4f}")
    print(f"Depressed mean (resampled): {np.mean(neg_intensities_resampled):.4f} ± {np.std(neg_intensities_resampled):.4f}")
    print(f"Mann-Whitney U statistic: {u_stat:.4f}, p-value: {p_value:.6f}")
    
    healthy_regional_means = []
    depressed_regional_means = []
    significant_regions = []
    
    for region_idx in range(len(region_names)):
        pos_region_mask = pos_assignments_resampled == region_idx
        neg_region_mask = neg_assignments_resampled == region_idx
        pos_region_intensities = pos_intensities_resampled[pos_region_mask]
        neg_region_intensities = neg_intensities_resampled[neg_region_mask]
        region_name = region_names[region_idx].replace('_', ' ').title()

        if len(pos_region_intensities) > 0 and len(neg_region_intensities) > 0:
            healthy_mean = np.mean(pos_region_intensities)
            depressed_mean = np.mean(neg_region_intensities)
            healthy_regional_means.append(healthy_mean)
            depressed_regional_means.append(depressed_mean)
            
            if len(pos_region_intensities) > 1 and len(neg_region_intensities) > 1:
                try:
                    u_stat_region, p_value_region = stats.mannwhitneyu(pos_region_intensities, neg_region_intensities, alternative='two-sided')
                    if p_value_region < 0.05:
                        significant_regions.append(region_name)
                except ValueError:
                    continue
        else:
            healthy_regional_means.append(0.1)
            depressed_regional_means.append(0.1)
    
    print(f"Regions with p < 0.05 in this *single resampled run* ({len(significant_regions)}):")
    for region in significant_regions:
        print(f"  - {region}")
    
    return np.array(healthy_regional_means), np.array(depressed_regional_means), significant_regions

def main():
    df = pd.read_csv('diacwoz_data.csv', sep="\t")
    healthy_transcripts = df[df['label'] == 0]['text'].astype(str).tolist()
    depressed_transcripts = df[df['label'] == 1]['text'].astype(str).tolist()
    healthy_chunks = chunk_conversations(healthy_transcripts)
    depressed_chunks = chunk_conversations(depressed_transcripts)

    pos_results = run_emotion_mapping_analysis("Healthy", healthy_chunks)
    neg_results = run_emotion_mapping_analysis("Depressed", depressed_chunks)
    
    if pos_results is None or neg_results is None:
        print("Analysis failed, exiting. Check API key and data.")
        return

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

    print(f"Found {len(significant_regions)} statistically significant regions in the resampled run")
    

if __name__ == "__main__":
    main()