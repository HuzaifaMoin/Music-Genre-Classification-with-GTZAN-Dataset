# Music-Genre-Classification-with-GTZAN-Dataset
Overview

This project implements a music genre classification system using the GTZAN dataset
. The goal is to classify audio clips into 10 genres (e.g., Blues, Classical, Rock) using two approaches:

Tabular Approach – Uses pre-extracted audio features (MFCCs, chroma, tempo) with a dense neural network.

Image-Based Approach – Converts audio clips into mel-spectrograms and trains a Convolutional Neural Network (CNN).

The project compares the performance of both models and explores transfer learning as an optional enhancement.

Dataset

Audio Files: 1000 WAV files (30 seconds each) across 10 genres in the genres_original folder.

Pre-extracted Features: CSV file features_30_sec.csv containing MFCCs, chroma, tempo, and other audio descriptors.

Tools & Libraries

Python 3

Librosa – Audio processing (MFCCs, spectrograms)

Scikit-learn – Data preprocessing and evaluation metrics

TensorFlow / Keras – Neural network models

Matplotlib / Seaborn / Plotly – Visualizations

NumPy / Pandas – Data handling

Features & Methods
Tabular Approach

Dense Neural Networks with multiple architectures (Model_1 to Model_4)

Early stopping callback for faster convergence

StandardScaler for feature normalization

Image-Based Approach

Mel-spectrogram generation from WAV files

Optional data augmentation via time/frequency masking

CNN architecture with convolutional, pooling, and dropout layers

Trained with Adam optimizer and early stopping

Exploratory Data Analysis (EDA)

Correlation Heatmaps – To examine relationships between audio features

Tempo Boxplots – To compare BPM distribution across genres

Visual inspection of waveforms for sample audio files

Model Training

Dense Neural Network (Model_4) achieved the highest accuracy (~94–95%) on test data.

Simple CNN achieved competitive accuracy (~85–88%) and better generalization.

Performance Metrics:

Accuracy, precision, recall, F1-score

Confusion matrices

Per-class accuracy analysis

Usage

Clone this repository:

git clone <repo_url>
cd music-genre-classification


Install dependencies:

pip install -r requirements.txt


Run the notebook for:

Feature-based model training

Spectrogram generation and CNN training

Visualization of results and performance comparison

Results

Dense Neural Network performs better for high-precision classification.

CNN shows more stable validation performance and generalizes better on unseen data.

Confusion matrices highlight genre-specific misclassifications.

Sample Visualizations:

Correlation Heatmaps

BPM Boxplots

Training Accuracy & Loss curves

Confusion Matrices

Per-class accuracy analysis

Conclusion

For high-accuracy classification on structured features, Dense Neural Networks are preferred.

For tasks requiring better generalization and smaller datasets, CNN-based spectrogram models are effective.

Further improvements:

Data augmentation for CNN

Regularization for Dense NN

Transfer learning using pretrained models (e.g., VGG16)
