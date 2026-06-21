# Deep Audio Ensembles for Music Genre Classification
An end-to-end machine learning and deep learning repository implementing classical DSP features, Convolutional Recurrent Neural Networks (CRNNs), and state-of-the-art transformer ensembles for classifying audio tracks into ten distinct music genres.

Deployed Application: [Music Genre Classification](https://huggingface.co/spaces/shukhaRani/MUSIC-GENRE)

---

## Executive Summary
This repository captures a rigorous, five-milestone development lifecycle that advances from basic audio processing to a production-grade deep learning model. The final deployment leverages an ensemble of Audio Spectrogram Transformers (AST), HuBERT, and Wav2Vec2 models to classify raw audio files into ten musical genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

---

## Project Architecture & Milestone Progression

### Milestone 1: Audio Processing & Pipeline Ingestion
- Ingested raw stems (drums, vocals, bass, other) from the GTZAN-style dataset structure.
- Audited file integrity, filtering corrupted files and identifying non-standard sizes.
- Programmed a silence detection utility using Librosa to split audio clips at specific dB thresholds, identifying the duration and position (start, middle, end) of silent intervals.
- Stacked and mixed raw stems (with volume normalization and scaling) to produce deterministic 5-second and 30-second audio mixes.

### Milestone 2: Feature Engineering & Baseline Classifier
- Extracted hand-crafted digital signal processing (DSP) features from raw audio signals:
  - Tempo (Beats Per Minute)
  - Spectral Centroid (timbre representation)
  - Zero-Crossing Rate (rough noisiness measure)
  - Spectral Rolloff (spectral shape description)
- Formed a tabular dataset and trained a Decision Tree Classifier (max depth 5) as an early baseline, evaluating validation macro F1-score, precision, recall, and confusion matrix metrics.

### Milestone 3: Deep Learning Foundations for Audio
- Conceptualized multi-dimensional audio representations, converting raw 1D time-series signals to 2D Mel-Spectrograms.
- Structured PyTorch operations for 2D convolutional networks:
  - Computed output spatial dimensions after Conv2d layers and MaxPool2d pooling layers.
  - Calculated parameters for fully connected Linear layers.
- Formulated loss functions (CrossEntropyLoss) and gradient updates (SGD, Adam).
- Set up training metrics tracking using Weights and Biases (wandb).

### Milestone 4: Synthetic Dataset & Deep Recurrent CNNs
- Built a synthetic dataset generator to overlay background environment noises (ESC-50 dataset) onto mixed audio stems.
- Converted raw audio waveforms into log-mel spectrogram tensors in decibels (dB).
- Designed and trained a Convolutional Recurrent Neural Network (CRNN) in PyTorch:
  - CNN Block 1: Conv2d (32 filters, 3x3) + BatchNorm + ReLU + MaxPool2d
  - CNN Block 2: Conv2d (64 filters, 3x3) + BatchNorm + ReLU + MaxPool2d
  - Recurrent Block: Bidirectional Long Short-Term Memory (LSTM) with hidden size 64
  - Classifier: Linear output projection to 10 classes
- Maintained strict reproducible seeds and executed GPU training.

### Milestone 5: Audio Spectrogram Transformers (AST)
- Transitioned to attention-based vision models for audio classification.
- Adapted `MIT/ast-finetuned-audioset-10-10-0.4593` for 10-class genre classification.
- Froze the transformer base and trained the classification head using Cosine Annealing with Warmup learning rate scheduling.
- Fine-tuned the entire network using high-throughput configurations (mixed precision training, gradient accumulation, and persistent worker data loaders).

### Production Ensemble Pipeline (AST + HuBERT + Wav2Vec2)
The final production system combines three separate deep learning architectures:
1. **AST (Audio Spectrogram Transformer)**: Captures global spectral context by patching 2D spectrograms.
2. **HuBERT (Hidden-Unit BERT)**: Extract self-supervised representation from raw 1D audio.
3. **Wav2Vec2**: Learns robust contextual speech-and-audio features directly from waveform inputs.

The pipeline integrates Test-Time Augmentation (TTA), combining predictions from random temporal crop shifts to increase classification robustness against noise and outliers.

---

## Technical Performance & Details

| Phase / Model | Key Components | Feature Space | Validation Metric | Target Classes |
| --- | --- | --- | --- | --- |
| **Milestone 2 Baseline** | Decision Tree | Tempo, Centroid, ZCR, Rolloff | Macro F1: ~0.35 | 10 Genres |
| **Milestone 4 Deep Learning** | CRNN (CNN + Bi-LSTM) | 128-bin Mel Spectrograms (dB) | Accuracy: 64.00% | 10 Genres |
| **Milestone 5 SOTA Model** | AST (Transformers) | 128x1024 Spectrogram Patches | F1 Score: 85%+ | 10 Genres |
| **Production Ensemble** | AST + HuBERT + Wav2Vec2 | Hybrid (Raw Waveform + Mel-spec) | F1 Score: ~89% | 10 Genres |

---

## Implementation Details & Codebase Structure

- `24f2007655-notebook-t12026.ipynb`: LightGBM-based NLP classifier notebook used in related comment classification experiments.
- `DL-24f2007655-notebook-t12026`: Main Kaggle export notebook containing the AST + HuBERT + Wav2Vec2 ensemble pipeline.
- `Milestone1`: Script/notebook solving data loading, silence analysis, and audio mixing challenges.
- `milestone-2`: Script/notebook implementing Librosa feature extraction and Decision Tree classification.
- `Milestone-3`: Script/notebook outlining PyTorch spectrogram dimension operations.
- `Milestone-4`: Notebook generating the noisy synthetic mashups and training the CRNN model.
- `Milestone-5`: Notebook implementing AST model loading and parameter tuning.
- `kaggle-notebook.ipynb` / `kaggle-notebook`: Supplementary execution environments.

---

## Getting Started

### Prerequisites
Install the required system dependencies and python libraries:
```bash
pip install -q transformers librosa torch torchaudio soundfile torchmetrics matplotlib tqdm
```

### Feature Extraction Example
To extract Mel-spectrograms from raw audio and save them as PyTorch tensors:
```python
import torch
import torchaudio

def extract_mel_spectrogram(audio_path, target_sr=22050):
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=2048, hop_length=512, n_mels=128
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    mel_spec = mel_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    return mel_spec_db
```

### Model Classification Example
To run inference using the fine-tuned Audio Spectrogram Transformer:
```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=10,
    ignore_mismatched_sizes=True
)
```

---

## Deployment
The model is fully deployed and accessible via Hugging Face Spaces. The web interface allows users to upload raw audio files (WAV, MP3, etc.) and receive real-time classification probabilities across the ten target genres. 

Access the deployment here: [Music Genre Classification](https://huggingface.co/spaces/shukhaRani/MUSIC-GENRE)
