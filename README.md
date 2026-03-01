# 🧠 Sentiment Analysis on Drug Reviews

### A Comparative Study of Classical Machine Learning and Transformer-Based Models

---

## 📌 Abstract

This study investigates sentiment classification on the Drugs.com review dataset using both traditional machine learning methods and transformer-based deep learning architectures.

We conduct:

* Binary sentiment classification using TF-IDF and Word2Vec features with classical ML and deep neural networks.
* Three-class sentiment classification using a fine-tuned Google Research BERT model.

The objective is to analyze the representational power of sparse vs dense embeddings and compare traditional models with contextualized transformers in a real-world healthcare-related NLP task.

---

## 📂 Dataset

* Source: Drugs.com Drug Review Dataset
* User-generated drug reviews
* Associated numerical ratings (1–10)

### Label Construction

Binary classification:

* Positive (rating ≥ threshold)
* Negative (rating < threshold)

Three-class classification:

* Negative
* Neutral
* Positive

Text preprocessing included:

* Lowercasing
* Punctuation removal
* Stopword filtering
* Lemmatization

---

## 🧪 Methodology

### 1️⃣ Feature Engineering (Binary Classification)

#### TF-IDF Representation

Sparse high-dimensional representation using term frequency-inverse document frequency weighting.

#### Word2Vec Embeddings

Dense distributed word representations capturing semantic similarity.

---

### 2️⃣ Classical Machine Learning Models

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors

All models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

### 3️⃣ Deep Learning Models (Binary)

* Fully Connected Neural Network (TF-IDF input)
* LSTM-based model (Word2Vec input)

---

### 4️⃣ Transformer-Based Model (Three-Class)

We fine-tuned **BERT-base-uncased** using the HuggingFace Transformers library.

Architecture:

* Base: 12-layer Transformer encoder
* Hidden size: 768
* Attention heads: 12
* Output layer adapted for 3-class classification

Training Setup:

* Optimizer: AdamW
* Loss: CrossEntropyLoss
* Learning rate scheduling
* Early stopping

---

## 📊 Experimental Results

### Binary Classification

| Model               | Feature  | Accuracy | F1 |

| Logistic Regression | TF-IDF   | 0.82     | 0.81 |

| Naive Bayes         | TF-IDF   | 0.78     | 0.77 |

| Random Forest       | TF-IDF   | 0.81     | 0.79 |

| LSTM                | Word2Vec | 0.83     | 0.83 |

---

### Three-Class BERT Model

Fine-tuned BERT achieved:

* Higher macro-F1 compared to classical models
* Better generalization on neutral class
* Improved semantic understanding due to contextual embeddings

---

## 🤗 HuggingFace Model

Fine-tuned model available at:

```
https://huggingface.co/erdemyavuz/drug-review-sentiment-analysis-bert
```

Inference example:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="erdemyavuz/drug-review-sentiment-analysis-bert"
)

classifier("This medication completely changed my life.")
```

---

## 🔍 Comparative Insights

1. TF-IDF + Linear models perform surprisingly strong in binary classification.
2. Word2Vec improves semantic clustering but requires deeper models to outperform TF-IDF.
3. BERT significantly improves:

   * Context modeling
   * Neutral sentiment detection
   * Domain-sensitive expressions

---

## 🧠 Research Contributions

* Empirical comparison of sparse vs dense representations
* Classical ML vs Deep Learning vs Transformer benchmarking
* Sentiment classification in healthcare domain
* Publicly available fine-tuned transformer model

---

## 🛠 Tech Stack

* Python
* Scikit-learn
* TensorFlow / PyTorch
* HuggingFace Transformers
* Word2Vec
* TF-IDF

---

