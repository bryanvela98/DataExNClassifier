# Text Classification and Preprocessing

A NLP project implementing text preprocessing pipelines and machine learning classifiers for spam detection. This project demonstrates fundamental natural language processing techniques including tokenization, stopword removal, feature extraction, and text classification.

## Project Overview

This project consists of two main parts:

1. **Text Preprocessing and Exploration**: Implementation of a custom tokenization pipeline and text visualization using Project Gutenberg books
2. **SMS Spam Classification**: Training and evaluating machine learning models for binary text classification

## Features

### Part 1: Text Preprocessing and Visualization

- **Custom Tokenization Engine**
  - Handles contractions (can't, I'm, won't)
  - Preserves decimal numbers (3.14, 2.5)
  - Separates punctuation marks and brackets
  - Protects ellipsis (...) as single token
  - Tokenizes arithmetic operations

- **Stopword Removal**
  - Filters common words to focus on meaningful content
  - Uses custom stopword list (33 words)

- **Special Character Handling**
  - Retains alphabetic, numeric, and alphanumeric tokens
  - Preserves decimal numbers

- **Text Visualization**
  - Frequency-based word distribution
  - TF-IDF weighted word importance
  - Genre-specific analysis (Animals and Detective Fiction)

### Part 2: SMS Spam Classification

- **Feature Extraction**
  - Bag-of-Words (Count-based)
  - TF-IDF weighted features

- **Machine Learning Models**
  - Logistic Regression
  - Multi-Layer Perceptron (MLP) Neural Network

- **Model Optimization**
  - Hyperparameter tuning for MLP
  - Cross-validation with stratified sampling
  - Performance metrics: Precision, Recall, F1-Score

## Project Structure

```
Assignment1/
├── Assignment1.ipynb          # Main Jupyter notebook
├── data/
│   ├── stopwords.txt         # Stopword list
│   ├── books/
│   │   ├── animals/          # Animal genre books
│   │   └── detective_fiction/ # Detective fiction books
│   └── sms/
│       └── sms_data.csv      # SMS spam dataset
├── scripts/
│   ├── __init__.py
│   ├── part1_functions.py    # Preprocessing functions
│   ├── pg_clean_text.py      # Project Gutenberg text cleaner
│   └── utils.py              # Utility functions
└── README.md
```

## Requirements

- Python >= 3.9
- Jupyter Notebook
- scikit-learn >= 1.5
- pandas
- matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/bryanvela98/DataExNClassifier.git
cd DataExNClassifier

# Install required packages
pip install jupyter scikit-learn pandas matplotlib
```

## Usage

### Running the Notebook

```bash
jupyter notebook Assignment1.ipynb
```

### Preprocessing Text

```python
from scripts.part1_functions import preprocess, read_and_concat_books

# Load and preprocess books
text = read_and_concat_books("./data/books/animals")
tokens = preprocess(text)
```

### Training Spam Classifier

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Create feature vectors
vectorizer = CountVectorizer(tokenizer=preprocess_sms)
X = vectorizer.fit_transform(df['text'])

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
```

## Methodology

### Tokenization Rules

1. **Contraction Handling**: Splits contractions mechanically (can't → ['ca', "n't"])
2. **Punctuation Separation**: Treats each punctuation mark as separate token
3. **Decimal Preservation**: Maintains decimal numbers intact (3.14)
4. **Ellipsis Protection**: Treats '...' as single token

### Text Classification Pipeline

1. **Data Preprocessing**: Tokenization, lowercasing, stopword removal
2. **Feature Extraction**: Count-based or TF-IDF vectorization
3. **Model Training**: Logistic Regression and MLP classifiers
4. **Hyperparameter Tuning**: Grid search for optimal MLP configuration
5. **Evaluation**: Precision, Recall, and F1-Score metrics

## Results

### Model Performance

The project evaluates four model configurations:

- Logistic Regression with Count features
- MLP with Count features
- Logistic Regression with TF-IDF features
- MLP with TF-IDF features

Hyperparameter tuning explores:

- Hidden layer sizes: (50,), (100,)
- Learning rates: 0.001, 0.01

### Visualization

- Top 10 most frequent words by genre
- Top 10 words by TF-IDF score
- Model performance comparison charts

## Key Implementation Details

### Tokenization Algorithm

Uses regular expressions to:

- Add spaces around contractions for proper splitting
- Protect ellipsis with placeholder
- Separate punctuation and operators
- Handle decimal points with negative lookahead/lookbehind

### Model Training

- Train/test split: 80/20
- Random state: 42 (for reproducibility)
- Stratified sampling: Maintains spam/ham ratio
- Positive label: 'spam' for metric calculation

## Dataset

- **Project Gutenberg Books**: Text corpus from two genres (Animals, Detective Fiction)
- **SMS Spam Collection**: Binary classification dataset with 'text' and 'label' columns

## Author

Bryan Vela

- GitHub: [@bryanvela98](https://github.com/bryanvela98)
