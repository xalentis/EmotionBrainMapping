# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency, ttest_ind
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon')
UMAP = None

def balance_classes(healthy_texts: List[str], depressed_texts: List[str], 
                   strategy: str = 'undersample', random_state: int = None) -> Tuple[List[str], List[str]]:
    if random_state is not None:
        np.random.seed(random_state)
    
    n_healthy = len(healthy_texts)
    n_depressed = len(depressed_texts)
    
    if strategy == 'undersample':
        target_size = min(n_healthy, n_depressed)
        if n_healthy > target_size:
            healthy_balanced = list(np.random.choice(healthy_texts, target_size, replace=False))
        else:
            healthy_balanced = healthy_texts.copy()
        if n_depressed > target_size:
            depressed_balanced = list(np.random.choice(depressed_texts, target_size, replace=False))
        else:
            depressed_balanced = depressed_texts.copy()
            
    elif strategy == 'oversample':
        target_size = max(n_healthy, n_depressed)
        if n_healthy < target_size:
            healthy_balanced = list(np.random.choice(healthy_texts, target_size, replace=True))
        else:
            healthy_balanced = healthy_texts.copy()
        if n_depressed < target_size:
            depressed_balanced = list(np.random.choice(depressed_texts, target_size, replace=True))
        else:
            depressed_balanced = depressed_texts.copy()
            
    elif strategy == 'hybrid':
        target_size = int((n_healthy + n_depressed) / 2)
        healthy_balanced = list(np.random.choice(healthy_texts, target_size, replace=(n_healthy < target_size)))
        depressed_balanced = list(np.random.choice(depressed_texts, target_size, replace=(n_depressed < target_size)))
    
    return healthy_balanced, depressed_balanced


def get_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    api_key = "your open-ai api key here"
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return np.array([item.embedding for item in response.data])
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


class EmotionBrainMapper:
    def __init__(self, dim_reduction_method: str = 'pca', n_components: int = 3):
        self.dim_reduction_method = dim_reduction_method.lower()
        self.n_components = n_components
        self.emotion_regions = self.define_emotion_regions()
        self.n_emotion_regions = len(self.emotion_regions)
        self.scaler = StandardScaler()
        self.reducer = None
        self.kmeans = KMeans(n_clusters=self.n_emotion_regions, random_state=42, n_init=10)
        self.sia = SentimentIntensityAnalyzer()
        self.embeddings_reduced = None
        self.original_cluster_labels = None
        self.region_assignments = None

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
        n_components = min(self.n_components, n_samples, n_features)
        
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        if self.dim_reduction_method == 'pca':
            self.reducer = PCA(n_components=n_components)
        elif self.dim_reduction_method == 'tsne':
            self.reducer = TSNE(n_components=n_components, random_state=42)
        elif self.dim_reduction_method == 'umap' and UMAP is not None:
            self.reducer = UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Invalid reduction method: {self.dim_reduction_method}")
        
        self.embeddings_reduced = self.reducer.fit_transform(embeddings_scaled)
        
        if self.embeddings_reduced.shape[1] < 3:
            padding = np.zeros((self.embeddings_reduced.shape[0], 3 - self.embeddings_reduced.shape[1]))
            self.embeddings_reduced = np.hstack([self.embeddings_reduced, padding])

        n_clusters = min(self.n_emotion_regions, embeddings.shape[0])
        if n_clusters != self.n_emotion_regions:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        self.kmeans.fit(self.embeddings_reduced)
        cluster_centers = self.kmeans.cluster_centers_
        self.original_cluster_labels = np.argmin(cdist(self.embeddings_reduced, cluster_centers), axis=1)
        region_coords_map = np.array(list(self.emotion_regions.values()), dtype=np.float64)

        assigned_regions = []
        used_indices = set()
        for center in cluster_centers:
            dists = cdist([center], region_coords_map)[0]
            for idx in np.argsort(dists):
                if idx not in used_indices:
                    assigned_regions.append(idx)
                    used_indices.add(idx)
                    break

        cluster_to_region = dict(zip(range(len(assigned_regions)), assigned_regions))
        self.region_assignments = np.array([cluster_to_region[c] for c in self.original_cluster_labels])
        brain_coordinates = region_coords_map[self.region_assignments].copy()
        return brain_coordinates, self.region_assignments

    def estimate_emotion_intensity(self, texts: List[str]) -> np.ndarray:
        intensities = []
        for text in texts:
            score = self.sia.polarity_scores(text)['compound']
            intensity = min(abs(score) + 1, 2.0)  # Scale to [0, 2]
            intensities.append(intensity)
        return np.array(intensities)


def run_emotion_mapping_analysis(title: str, conversation: List[str], dim_reduction_method='pca', n_components=3):
    np.random.seed(42)
    embeddings = get_embeddings(conversation)
    if embeddings is None:
        return None, None, None, None, None

    mapper = EmotionBrainMapper(dim_reduction_method=dim_reduction_method, n_components=n_components)
    brain_coords, region_assignments = mapper.fit_transform_embeddings(embeddings)
    emotion_intensities = mapper.estimate_emotion_intensity(conversation)
    sil_score = silhouette_score(mapper.embeddings_reduced, mapper.original_cluster_labels)
    print(f"{title} Silhouette Score: {sil_score:.3f}")
    print(f"{title} Emotion Intensities: {np.round(emotion_intensities, 2)}")
    return mapper, embeddings, emotion_intensities, region_assignments, brain_coords


def bootstrap_intensity_ci(texts: List[str], n_boots: int = 100, dim_reduction_method='pca', n_components=3) -> Tuple[float, float]:
    boot_means = []
    for _ in range(n_boots):
        sample = np.random.choice(texts, len(texts), replace=True)
        mapper = EmotionBrainMapper(dim_reduction_method=dim_reduction_method, n_components=n_components)
        ints = mapper.estimate_emotion_intensity(sample)
        boot_means.append(np.mean(ints))
    mean_int = np.mean(boot_means)
    std_int = np.std(boot_means)
    ci = 1.96 * std_int
    return mean_int, ci


def run_multiple_trials(healthy_texts: List[str], depressed_texts: List[str], 
                       n_trials: int = 10, balance_strategy: str = 'undersample',
                       dim_reduction_method: str = 'pca', n_components: int = 3):
    print(f"\n=== Running {n_trials} trials with {balance_strategy} balancing ===")
    print(f"Original sizes - Healthy: {len(healthy_texts)}, Depressed: {len(depressed_texts)}")
    
    trial_results = {
        'healthy_means': [],
        'depressed_means': [],
        'healthy_cis': [],
        'depressed_cis': [],
        'p_values_chi': [],
        'p_values_t': [],
        'silhouette_healthy': [],
        'silhouette_depressed': [],
        'healthy_sizes': [],
        'depressed_sizes': []
    }
    
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        
        # Balance classes for this trial
        healthy_balanced, depressed_balanced = balance_classes(
            healthy_texts, depressed_texts, 
            strategy=balance_strategy, 
            random_state=trial
        )
        
        print(f"Balanced sizes - Healthy: {len(healthy_balanced)}, Depressed: {len(depressed_balanced)}")
        trial_results['healthy_sizes'].append(len(healthy_balanced))
        trial_results['depressed_sizes'].append(len(depressed_balanced))
        
        # Run analysis on balanced data
        pos_results = run_emotion_mapping_analysis(
            f"Healthy_T{trial+1}", healthy_balanced, 
            dim_reduction_method, n_components
        )
        neg_results = run_emotion_mapping_analysis(
            f"Depressed_T{trial+1}", depressed_balanced, 
            dim_reduction_method, n_components
        )
        
        if pos_results[0] is not None and neg_results[0] is not None:
            pos_mapper, pos_embeddings, pos_intensities, pos_assignments, pos_coords = pos_results
            neg_mapper, neg_embeddings, neg_intensities, neg_assignments, neg_coords = neg_results
            
            # Store silhouette scores
            pos_sil = silhouette_score(pos_mapper.embeddings_reduced, pos_mapper.original_cluster_labels)
            neg_sil = silhouette_score(neg_mapper.embeddings_reduced, neg_mapper.original_cluster_labels)
            trial_results['silhouette_healthy'].append(pos_sil)
            trial_results['silhouette_depressed'].append(neg_sil)
            
            # Bootstrap confidence intervals
            pos_mean, pos_ci = bootstrap_intensity_ci(healthy_balanced, n_boots=50, 
                                                    dim_reduction_method=dim_reduction_method, 
                                                    n_components=n_components)
            neg_mean, neg_ci = bootstrap_intensity_ci(depressed_balanced, n_boots=50,
                                                    dim_reduction_method=dim_reduction_method, 
                                                    n_components=n_components)
            
            trial_results['healthy_means'].append(pos_mean)
            trial_results['depressed_means'].append(neg_mean)
            trial_results['healthy_cis'].append(pos_ci)
            trial_results['depressed_cis'].append(neg_ci)
            
            # Statistical tests
            pos_counts = np.bincount(pos_assignments, minlength=pos_mapper.n_emotion_regions)
            neg_counts = np.bincount(neg_assignments, minlength=neg_mapper.n_emotion_regions)
            cont_table = np.vstack([pos_counts, neg_counts])
            chi2_stat, p_chi, dof, ex = chi2_contingency(cont_table)
            t_stat, p_t = ttest_ind(pos_intensities, neg_intensities)
            trial_results['p_values_chi'].append(p_chi)
            trial_results['p_values_t'].append(p_t)
            print(f"Trial {trial+1} - Chi-squared p: {p_chi:.4f}, T-test p: {p_t:.4f}")
            print(f"Trial {trial+1} - Healthy mean: {pos_mean:.3f}±{pos_ci:.3f}, Depressed mean: {neg_mean:.3f}±{neg_ci:.3f}")
            
    return trial_results


def analyze_trial_results(trial_results: dict):
    print(f"\n=== MULTI-TRIAL ANALYSIS SUMMARY ===")
    print(f"Number of trials: {len(trial_results['healthy_means'])}")

    healthy_means = np.array(trial_results['healthy_means'])
    depressed_means = np.array(trial_results['depressed_means'])
    p_values_chi = np.array(trial_results['p_values_chi'])
    p_values_t = np.array(trial_results['p_values_t'])

    print(f"\nHealthy Mean Intensities across trials:")
    print(f"  Mean: {np.mean(healthy_means):.4f} ± {np.std(healthy_means):.4f}")
    print(f"  Range: [{np.min(healthy_means):.4f}, {np.max(healthy_means):.4f}]")
    print(f"\nDepressed Mean Intensities across trials:")
    print(f"  Mean: {np.mean(depressed_means):.4f} ± {np.std(depressed_means):.4f}")
    print(f"  Range: [{np.min(depressed_means):.4f}, {np.max(depressed_means):.4f}]")
    
    print(f"\nEffect sizes (Cohen's d) across trials:")
    # Calculate Cohen's d for each trial (approximate)
    effect_sizes = []
    for i in range(len(healthy_means)):
        # Simple effect size approximation
        pooled_std = np.sqrt((trial_results['healthy_cis'][i]**2 + trial_results['depressed_cis'][i]**2) / 2)
        if pooled_std > 0:
            d = (healthy_means[i] - depressed_means[i]) / pooled_std
            effect_sizes.append(abs(d))
    
    if effect_sizes:
        effect_sizes = np.array(effect_sizes)
        print(f"  Mean |Cohen's d|: {np.mean(effect_sizes):.4f} ± {np.std(effect_sizes):.4f}")
        print(f"  Range: [{np.min(effect_sizes):.4f}, {np.max(effect_sizes):.4f}]")
    
    # Statistical significance across trials
    sig_chi = np.sum(p_values_chi < 0.05)
    sig_t = np.sum(p_values_t < 0.05)
    
    print(f"\nStatistical Significance across trials:")
    print(f"  Chi-squared tests significant (p < 0.05): {sig_chi}/{len(p_values_chi)} ({sig_chi/len(p_values_chi)*100:.1f}%)")
    print(f"  T-tests significant (p < 0.05): {sig_t}/{len(p_values_t)} ({sig_t/len(p_values_t)*100:.1f}%)")
    print(f"  Mean Chi-squared p-value: {np.mean(p_values_chi):.4f}")
    print(f"  Mean T-test p-value: {np.mean(p_values_t):.4f}")
    
    # Silhouette scores
    if trial_results['silhouette_healthy'] and trial_results['silhouette_depressed']:
        sil_healthy = np.array(trial_results['silhouette_healthy'])
        sil_depressed = np.array(trial_results['silhouette_depressed'])
        
        print(f"\nClustering Quality (Silhouette Scores):")
        print(f"  Healthy: {np.mean(sil_healthy):.3f} ± {np.std(sil_healthy):.3f}")
        print(f"  Depressed: {np.mean(sil_depressed):.3f} ± {np.std(sil_depressed):.3f}")

    mean_diff = np.mean(healthy_means) - np.mean(depressed_means)
    print(f"\n=== CONCLUSIONS ===")
    print(f"Average difference in emotion intensity: {mean_diff:.4f}")
    print(f"Direction: {'Healthy > Depressed' if mean_diff > 0 else 'Depressed > Healthy'}")
    if sig_t > len(p_values_t) * 0.5:
        print(f"Result: Consistently significant differences across trials")
    elif sig_t > 0:
        print(f"Result: Some significant differences, but not consistent")
    else:
        print(f"Result: No significant differences detected")


if __name__ == "__main__":
    df = pd.read_csv('BrainEmbeddings/diacwoz_data.csv', sep="\t")
    healthy_conversation = [text.replace('.', ' ') for text in df[df['label'] == 0]['text'].tolist()]
    depressed_conversation = [text.replace('.', ' ') for text in df[df['label'] == 1]['text'].tolist()]
    print(f"Original dataset sizes:")
    print(f"Healthy: {len(healthy_conversation)} samples")
    print(f"Depressed: {len(depressed_conversation)} samples")
    print(f"Imbalance ratio: {len(healthy_conversation)/len(depressed_conversation):.2f}:1")
    
    print(f"\n=== ABLATION STUDIES (on original imbalanced data) ===")
    methods = ['pca', 'tsne']
    dims = [2, 3]
    
    for method in methods:
        for n_comp in dims:
            print(f"\nAblation - Method={method}, n_components={n_comp}")
            pos_results = run_emotion_mapping_analysis("Healthy", healthy_conversation[:50], method, n_comp)
    
    strategies = ['undersample', 'oversample', 'hybrid']
    n_trials = 5
    all_strategy_results = {}
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")
        
        trial_results = run_multiple_trials(
            healthy_conversation, 
            depressed_conversation,
            n_trials=n_trials,
            balance_strategy=strategy,
            dim_reduction_method='pca',
            n_components=2
        )
        
        all_strategy_results[strategy] = trial_results
        analyze_trial_results(trial_results)
    
    print(f"\n{'='*60}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    for strategy, results in all_strategy_results.items():
        if results['p_values_t']:
            mean_p_t = np.mean(results['p_values_t'])
            sig_count = np.sum(np.array(results['p_values_t']) < 0.05)
            total_trials = len(results['p_values_t'])
            mean_healthy = np.mean(results['healthy_means'])
            mean_depressed = np.mean(results['depressed_means'])
            
            print(f"\n{strategy.upper()}:")
            print(f"  Mean T-test p-value: {mean_p_t:.4f}")
            print(f"  Significant trials: {sig_count}/{total_trials} ({sig_count/total_trials*100:.1f}%)")
            print(f"  Mean intensities: Healthy={mean_healthy:.4f}, Depressed={mean_depressed:.4f}")
            print(f"  Difference: {mean_healthy - mean_depressed:.4f}")
    