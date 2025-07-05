# Gideon Vos 2025
# James Cook University
# www.linkedin.com/in/gideonvos

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter
import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
from sklearn.utils import resample


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def save_word2vec_model(filepath: str, W1: np.ndarray, W2: np.ndarray, vocab: Dict[str, int]):
    """Saves the Word2Vec model components to a file."""
    np.savez_compressed(filepath, W1=W1, W2=W2, vocab=vocab)
    print(f"Word2Vec model saved to {filepath}")


def load_word2vec_model(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
    """Loads the Word2Vec model components from a file."""
    if not os.path.exists(filepath):
        return None
    try:
        data = np.load(filepath, allow_pickle=True)
        W1 = data['W1']
        W2 = data['W2']
        vocab = data['vocab'].item()
        print(f"Word2Vec model loaded from {filepath}")
        return W1, W2, vocab
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None


def download_nrc_lexicon(url: str, filepath: str) -> bool:
    """Downloads the NRC lexicon if it doesn't already exist."""
    if os.path.exists(filepath):
        print(f"Lexicon already exists at {filepath}")
        return True
    try:
        print(f"Downloading NRC Emotion Intensity Lexicon from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading lexicon: {e}")
        return False


def load_nrc_lexicon(filepath: str) -> Optional[pd.DataFrame]:
    """Loads the NRC lexicon into a pandas DataFrame."""
    try:
        lexicon_df = pd.read_csv(filepath,
                                 names=["word", "emotion", "intensity_score"],
                                 sep='\t', skiprows=1)
        return lexicon_df
    except FileNotFoundError:
        print(f"Error: Lexicon file not found at {filepath}")
        return None


def load_and_combine_data(diacwoz_path: str, emotions_path: str) -> Optional[pd.DataFrame]:
    """Loads and combines the two CSV files into a single DataFrame."""
    try:
        diacwoz_df = pd.read_csv(diacwoz_path, sep="\t")
        diacwoz_df['label'] = diacwoz_df['label'].map({0: 'healthy', 1: 'depressed'})

        # include emotions only when training embeddings model
        #emotions_df = pd.read_csv(emotions_path, sep='\t')
        #emotions_df.columns = ["text","label","id"]
        #emotions_df = emotions_df[['text', 'label']]

        #combined_df = pd.concat([diacwoz_df, emotions_df], ignore_index=True)
        print("Data loaded and combined successfully.")
        return diacwoz_df #combined_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_text(text: str) -> List[str]:
    """Simple text preprocessing: lowercase and tokenize."""
    return text.lower().split()


def create_vocabulary(corpus: List[List[str]], min_freq: int = 5) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Creates a vocabulary from the corpus."""
    word_counts = Counter(word for sentence in corpus for word in sentence)
    vocab = {word: i for i, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    ix_to_word = {i: word for word, i in vocab.items()}
    return vocab, ix_to_word


def generate_context_target_pairs(corpus: List[List[str]], vocab: Dict[str, int], window_size: int = 2) -> List[Tuple[int, int]]:
    """Generates context-target pairs for Word2Vec."""
    pairs = []
    for sentence in corpus:
        indices = [vocab.get(word) for word in sentence if word in vocab]
        for i, target_word_ix in enumerate(indices):
            for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                if i != j:
                    pairs.append((target_word_ix, indices[j]))
    return pairs


class Word2VecDataset(Dataset):
    """PyTorch Dataset for Word2Vec training pairs."""
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class Word2VecModel(nn.Module):
    """GPU-accelerated Word2Vec model using PyTorch."""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Word2VecModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embeddings (W1)
        self.embeddings_input = nn.Embedding(vocab_size, embedding_dim)
        # Output embeddings (W2)
        self.embeddings_output = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize weights
        self.embeddings_input.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.embeddings_output.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, target_word, context_word):
        target_embed = self.embeddings_input(target_word)
        context_embed = self.embeddings_output(context_word)
        score = torch.sum(target_embed * context_embed, dim=1)
        return score
    
    def get_embeddings(self):
        """Returns the trained embeddings as numpy arrays."""
        W1 = self.embeddings_input.weight.data.cpu().numpy().T
        W2 = self.embeddings_output.weight.data.cpu().numpy()
        return W1, W2


def train_word2vec_gpu(pairs: List[Tuple[int, int]], vocab_size: int, embedding_dim: int, 
                      learning_rate: float, epochs: int, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Trains a Word2Vec model using GPU acceleration."""
    
    dataset = Word2VecDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = Word2VecModel(vocab_size, embedding_dim).to(device)
    
    # Use negative sampling loss (more efficient than full softmax)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Training on {len(pairs)} pairs with batch size {batch_size}")
    print(f"Total batches per epoch: {len(dataloader)}")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (target_words, context_words) in enumerate(dataloader):
            target_words = target_words.to(device)
            context_words = context_words.to(device)
            pos_scores = model(target_words, context_words)
            pos_labels = torch.ones_like(pos_scores)
            neg_context = torch.randint(0, vocab_size, context_words.shape, device=device)
            neg_scores = model(target_words, neg_context)
            neg_labels = torch.zeros_like(neg_scores)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([pos_labels, neg_labels])
            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}')
    
    W1, W2 = model.get_embeddings()
    return W1, W2


def train_word2vec(pairs: List[Tuple[int, int]], vocab_size: int, embedding_dim: int, learning_rate: float, epochs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper function that uses GPU training if available, otherwise falls back to CPU."""
    if torch.cuda.is_available():
        print("Using GPU-accelerated training...")
        return train_word2vec_gpu(pairs, vocab_size, embedding_dim, learning_rate, epochs)
    else:
        print("CUDA not available. Using CPU training...")
        return train_word2vec_cpu(pairs, vocab_size, embedding_dim, learning_rate, epochs)


def train_word2vec_cpu(pairs: List[Tuple[int, int]], vocab_size: int, embedding_dim: int, learning_rate: float, epochs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Original CPU-based Word2Vec training (fallback)."""
    def softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    W1 = np.random.randn(embedding_dim, vocab_size)
    W2 = np.random.randn(vocab_size, embedding_dim)

    for epoch in range(epochs):
        loss = 0
        for target_ix, context_ix in pairs:
            h = W1[:, target_ix]
            u = W2 @ h
            y_pred = softmax(u)
            e = -y_pred
            e[context_ix] += 1
            dW2 = np.outer(e, h)
            dW1_col = W2.T @ e
            W1[:, target_ix] += learning_rate * dW1_col
            W2 += learning_rate * dW2
            loss += -np.log(y_pred[context_ix])
        print(f'Epoch: {epoch + 1}, Loss: {loss / len(pairs)}')
    return W1, W2


def get_word_embeddings(text: str, W1: np.ndarray, W2: np.ndarray, vocab: Dict[str, int]) -> Optional[np.ndarray]:
    """Gets the average embedding for a given text."""
    words = preprocess_text(text)
    embeddings = []
    for word in words:
        if word in vocab:
            word_ix = vocab[word]
            embeddings.append((W1[:, word_ix] + W2[word_ix]) / 2)
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)


class EmotionBrainMapper:
    """
    Handles the logic for mapping text embeddings to brain regions.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the mapper and defines emotion-to-brain-region coordinates.
        """
        self.emotion_regions = self.define_emotion_regions()

    def define_emotion_regions(self) -> Dict[str, List[float]]:
        """
        Defines key emotional brain regions with their MNI coordinates based on scientific literature.
        """
        return {
            # Core Affect & General Emotion Processing
            'amygdala': [0, -5, -18],
            'anterior_cingulate_cortex': [0, 25, 25],
            'insula': [0, 8, 0],
            'orbitofrontal_cortex': [0, 40, -15],
            'ventromedial_prefrontal_cortex': [0, 50, -10],
            'periaqueductal_gray': [0, -32, -12],
            'nucleus_accumbens': [0, 10, -8],
            'hippocampus': [0, -15, -20],
            'thalamus': [0, -15, 10],

            # Specific Emotions
            'admiration': [-10, 52, 4],
            'amusement': [-2, 54, 26],
            'anger': [-5, 25, 25],
            'annoyance': [-18, -105, -3],
            'approval': [-2, 54, 26],
            'caring': [9, -13, 7],
            'confusion': [-42, 26, 28],
            'curiosity': [-36, 52, 14],
            'desire': [0, 40, -15],
            'disappointment': [-18, -105, -3],
            'disapproval': [44, 20, 36],
            'disgust': [-38, 14, 2],
            'embarrassment': [-46, 10, 28],
            'excitement': [42, -14, 8],
            'fear': [-27, -4, -20],
            'gratitude': [0, 52, 4],
            'grief': [0, 53, 27],
            'joy': [-33, -82, -14],
            'love': [-6, 18, 33],
            'nervousness': [-6, 22, 38],
            'optimism': [26, 55, 21],
            'pride': [-10, 34, 32],
            'realization': [54, -55, 22],
            'relief': [0, 50, -10],
            'remorse': [16, -18, 62],
            'sadness': [47, 9, 21],
            'surprise': [54, -55, 22],
            'neutral': [0, 0, 0],
            'depressed': [-10, 48, -4],
            'healthy': [0, 50, -10],
        }


def estimate_emotion_intensity(text: str, nrc_lexicon: pd.DataFrame) -> float:
    """
    Estimates the emotional intensity of a text based on linguistic features
    and the NRC Emotion Intensity Lexicon.
    """
    intensity_score = 1.0
    
    # 1. Intensifiers
    intensifiers = ['very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally', 'so', 'much']
    for word in intensifiers:
        if word in text.lower():
            intensity_score += 0.2

    # 2. Punctuation
    intensity_score += text.count('!') * 0.3
    intensity_score += text.count('?') * 0.1
    
    # 3. Capitalization
    if text.isupper():
        intensity_score += 0.5
        
    # 4. NRC Emotion Intensity Lexicon
    words = preprocess_text(text)
    lexicon_intensity = 0.0
    word_count = 0
    for word in words:
        if word in nrc_lexicon['word'].values:
            scores = nrc_lexicon[nrc_lexicon['word'] == word]['intensity_score']
            if not scores.empty:
                lexicon_intensity += scores.mean()
                word_count += 1
                
    if word_count > 0:
        intensity_score += (lexicon_intensity / word_count)
    return intensity_score


def run_emotion_mapping_analysis(emotion_name: str, emotion_texts: List[str], W1: np.ndarray, W2: np.ndarray, vocab: Dict[str, int], mapper: EmotionBrainMapper, nrc_lexicon: pd.DataFrame) -> Optional[Dict]:
    """Runs the emotion mapping analysis for a given emotion."""
    
    text_embeddings = [get_word_embeddings(text, W1, W2, vocab) for text in emotion_texts]
    text_embeddings = [emb for emb in text_embeddings if emb is not None]

    if not text_embeddings:
        return None
    
    emotion_name_embedding = get_word_embeddings(emotion_name, W1, W2, vocab)
    if emotion_name_embedding is None:
        return None

    region_mapping = {}
    total_intensity = 0
    
    for region, _ in mapper.emotion_regions.items():
        region_embedding_proxy = get_word_embeddings(region.replace('_', ' '), W1, W2, vocab)
        
        if region_embedding_proxy is not None:
            similarity = np.dot(emotion_name_embedding, region_embedding_proxy) / (np.linalg.norm(emotion_name_embedding) * np.linalg.norm(region_embedding_proxy))
            
            intensity = np.mean([estimate_emotion_intensity(text, nrc_lexicon) for text in emotion_texts])
            total_intensity += intensity
            
            region_mapping[region] = {
                'avg_intensity': intensity * (1 + similarity)
            }

    if not region_mapping:
        return None

    return {
        'avg_intensity': total_intensity / len(emotion_texts),
        'region_mapping': region_mapping,
    }


def perform_statistical_tests(pos_intensities, pos_assignments, neg_intensities, neg_assignments, region_names):
    """
    Performs statistical significance tests between healthy and depressed groups.
    Uses oversampling to handle class imbalance.
    """
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS (HEALTHY vs. DEPRESSED)")
    print("="*60)
    
    # Oversample the minority class to balance the datasets for comparison
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
    elif len(neg_intensities) < len(pos_intensities):
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
    else:
        pos_intensities_resampled = pos_intensities
        pos_assignments_resampled = pos_assignments
        neg_intensities_resampled = neg_intensities
        neg_assignments_resampled = neg_assignments

    print(f"Original sizes - Healthy: {len(pos_intensities)}, Depressed: {len(neg_intensities)}")
    print(f"Resampled sizes - Healthy: {len(pos_intensities_resampled)}, Depressed: {len(neg_intensities_resampled)}")
    u_stat, p_value = stats.mannwhitneyu(pos_intensities_resampled, neg_intensities_resampled, alternative='two-sided')
    print(f"\nOVERALL INTENSITY COMPARISON:")
    print(f"Healthy mean intensity: {np.mean(pos_intensities_resampled):.4f} ± {np.std(pos_intensities_resampled):.4f}")
    print(f"Depressed mean intensity: {np.mean(neg_intensities_resampled):.4f} ± {np.std(neg_intensities_resampled):.4f}")
    print(f"Mann-Whitney U statistic: {u_stat:.4f}, p-value: {p_value:.6f}")
    print(f"Significant difference (α = 0.05): {'YES' if p_value < 0.05 else 'NO'}")
    print(f"\nPER-REGION INTENSITY ANALYSIS:")
    significant_regions = []
    
    for region_idx, region_name_str in enumerate(region_names):
        pos_region_mask = pos_assignments_resampled == region_idx
        neg_region_mask = neg_assignments_resampled == region_idx
        pos_region_intensities = pos_intensities_resampled[pos_region_mask]
        neg_region_intensities = neg_intensities_resampled[neg_region_mask]
        if len(pos_region_intensities) > 1 and len(neg_region_intensities) > 1:
            try:
                u_stat_region, p_value_region = stats.mannwhitneyu(pos_region_intensities, neg_region_intensities, alternative='two-sided')
                region_name_title = region_name_str.replace('_', ' ').title()
                print(f"  {region_name_title}:")
                print(f"    Healthy  : {len(pos_region_intensities)} texts, mean intensity: {np.mean(pos_region_intensities):.4f}")
                print(f"    Depressed: {len(neg_region_intensities)} texts, mean intensity: {np.mean(neg_region_intensities):.4f}")
                print(f"    Mann-Whitney U: {u_stat_region:.4f}, p-value: {p_value_region:.6f}")
                if p_value_region < 0.05:
                    significant_regions.append(region_name_title)
                    print(f"    *** SIGNIFICANT DIFFERENCE DETECTED ***")
            except ValueError:
                continue
    print(f"\nSUMMARY OF SIGNIFICANT FINDINGS:")
    if significant_regions:
        print(f"A significant difference in emotional intensity between Healthy and Depressed groups was found in {len(significant_regions)} region(s):")
        for region in significant_regions:
            print(f"  - {region}")
    else:
        print("No significant differences were found in regional emotional intensity between the two groups.")
    return pos_intensities_resampled, neg_intensities_resampled


if __name__ == '__main__':
    diacwoz_file = 'BrainEmbeddings/diacwoz_data.csv'
    emotions_file = 'BrainEmbeddings/emotions27.tsv'
    nrc_url = 'https://raw.githubusercontent.com/ishikasingh/Affective-text-gen/master/NRC-Emotion-Intensity-Lexicon-v1.txt'
    nrc_filepath = 'BrainEmbeddings/NRC-Emotion-Intensity-Lexicon-v1.txt'
    model_filepath = 'BrainEmbeddings/word2vec_model.npz'

    if download_nrc_lexicon(nrc_url, nrc_filepath):
        nrc_lexicon = load_nrc_lexicon(nrc_filepath)
        if nrc_lexicon is not None:
            df = load_and_combine_data(diacwoz_file, emotions_file)

            if df is not None:
                model_data = load_word2vec_model(model_filepath)
                if model_data:
                    W1, W2, vocab = model_data
                else:
                    print("No pre-trained model found. Training from scratch...")
                    corpus = [preprocess_text(text) for text in df['text'].astype(str)]
                    vocab, ix_to_word = create_vocabulary(corpus, min_freq=1)
                    pairs = generate_context_target_pairs(corpus, vocab, window_size=2)
                    embedding_dim = 100
                    learning_rate = 0.001
                    epochs = 10
                    
                    W1, W2 = train_word2vec(pairs, len(vocab), embedding_dim, learning_rate, epochs)
                    print("Word2Vec model training complete.")
                    save_word2vec_model(model_filepath, W1, W2, vocab)

                mapper = EmotionBrainMapper()
                all_results = {}
                emotion_labels = [
                    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                    'relief', 'remorse', 'sadness', 'surprise', 'neutral', 'depressed', 'healthy'
                ]
                for emotion_name in emotion_labels:
                    if emotion_name in df['label'].unique():
                        emotion_texts = df[df["label"] == emotion_name]["text"].astype(str).tolist()
                        results = run_emotion_mapping_analysis(emotion_name, emotion_texts, W1, W2, vocab, mapper, nrc_lexicon)
                        if results:
                            all_results[emotion_name] = results
                    
                print("\n\n" + "="*20 + " COMPARATIVE ANALYSIS REPORT " + "="*20)
                print("\n--- Brain Region Activity Across Emotions ---")
                brain_region_activity = {region: [] for region in mapper.emotion_regions.keys()}
                for emotion, data in all_results.items():
                    if 'region_mapping' in data:
                        for region, region_data in data['region_mapping'].items():
                            brain_region_activity[region].append(f"{emotion} (intensity: {region_data['avg_intensity']:.2f})")
                
                for region, mapped_emotions in sorted(brain_region_activity.items()):
                    if mapped_emotions:
                        print(f"{region.replace('_', ' ').title()}:")
                        print(f"  - Associated with: {', '.join(mapped_emotions)}")

                print("\n\n--- Overall Emotion Intensity Summary ---")
                if all_results:
                    sorted_intensities = sorted(all_results.items(), key=lambda item: item[1]['avg_intensity'], reverse=True)
                    
                    print("Emotion                       | Average Intensity")
                    print("------------------------------|-------------------")
                    for emotion, data in sorted_intensities:
                        print(f"{emotion:<29} | {data['avg_intensity']:.2f}")
                else:
                    print("No results to display for emotion intensity.")

                print("\nPreparing data for statistical analysis...")
                healthy_intensities = []
                healthy_assignments = []
                depressed_intensities = []
                depressed_assignments = []
                region_names = list(mapper.emotion_regions.keys())
                region_name_to_idx = {name: i for i, name in enumerate(region_names)}
                region_proxy_embeddings = {
                    region: get_word_embeddings(region.replace('_', ' '), W1, W2, vocab)
                    for region in region_names
                }

                region_proxy_embeddings = {k: v for k, v in region_proxy_embeddings.items() if v is not None}

                for label_type, intensity_list, assignment_list in [('healthy', healthy_intensities, healthy_assignments), ('depressed', depressed_intensities, depressed_assignments)]:
                    texts = df[df['label'] == label_type]['text'].astype(str).tolist()
                    for text in texts:
                        text_emb = get_word_embeddings(text, W1, W2, vocab)
                        if text_emb is None:
                            continue
                        intensity = estimate_emotion_intensity(text, nrc_lexicon)
                        similarities = {
                            region: np.dot(text_emb, proxy_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(proxy_emb))
                            for region, proxy_emb in region_proxy_embeddings.items()
                        }
                        if similarities:
                            best_region_name = max(similarities, key=similarities.get)
                            best_region_idx = region_name_to_idx[best_region_name]
                            
                            intensity_list.append(intensity)
                            assignment_list.append(best_region_idx)

                perform_statistical_tests(
                    np.array(healthy_intensities), 
                    np.array(healthy_assignments),
                    np.array(depressed_intensities), 
                    np.array(depressed_assignments),
                    region_names
                )