# 💊 Drug Review Sentiment Analysis using BERT

This repository contains an end-to-end Natural Language Processing (NLP) pipeline that classifies customer reviews of various drugs into sentiment categories (Negative, Neutral, Positive) using a Fine-Tuned BERT model. 

## 🚀 Live Demo
You can try the model directly in your browser without installing anything: 
**[👉 Click here to open the Web App](https://huggingface.co/spaces/erdemyavuz/drug-sentiment-analysis)**

---

### 🧪 Sample Reviews for Testing
If you want to test the model but don't have a review in mind, you can copy and paste the examples below into the application:

**🟢 Positive Review:**
> *"This medication completely changed my life. Within a week, all my symptoms disappeared and I experienced absolutely no side effects. Highly recommended!"*

**🔴 Negative Review:**
> *"Absolutely terrible experience. It made me feel extremely nauseous all day and didn't help with the pain at all. I had to stop taking it immediately."*

**🟡 Mixed / Neutral Review:**
> *"It works okay for the pain, but it makes me feel very sleepy during the day. I might ask my doctor for an alternative."*

---

## 📂 Project Structure

The project is divided into four main Jupyter Notebooks, documenting the entire machine learning lifecycle:

* **`1.Data Cleaning and Preprocessing.ipynb`**: Handles the raw data from Kaggle (Drugs.com dataset). Includes HTML unescaping, lowercasing, tokenization, removing non-alphanumerics, removing stopwords, and lemmatization using NLTK/WordNet.
* **`2.ML and DL Models.ipynb`**: Establishes baseline models including traditional Machine Learning and Deep Learning (CNN) architectures.
* **`3.Bert Fine Tuning.ipynb`**: The core of the project. Fine-tunes the `bert-base-uncased` model for sequence classification on 3 labels (0: Negative, 1: Neutral, 2: Positive).
* **`4.Bert Model Import - Extracting the False Predictions.ipynb`**: Focuses on error analysis by extracting and investigating the false predictions made by the BERT model to understand its limitations and improve future iterations.

## 🛠️ Tech Stack
* **Language:** Python
* **NLP & Deep Learning:** Hugging Face `transformers`, `PyTorch`, `NLTK`
* **Machine Learning:** `scikit-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Web App Framework:** `Streamlit` (Deployed on Hugging Face Spaces)

## 📊 Evaluation Metrics

Evaluation was conducted on a held-out validation set.

Metrics reported:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)
* Confusion Matrix

### Example

| Model               | Feature  | Accuracy | F1 |

| Logistic Regression | TF-IDF   | 0.82     | 0.81 |

| Naive Bayes         | TF-IDF   | 0.78     | 0.77 |

| Random Forest       | TF-IDF   | 0.81     | 0.79 |

| LSTM                | Word2Vec | 0.83     | 0.83 |

---

## 🔍 Key Findings

* Transformer-based contextual embeddings outperform classical TF-IDF based models in multi-class sentiment detection.
* The neutral class benefits significantly from contextual modeling.
* Domain-specific vocabulary (medical terminology) is better captured by BERT representations.

---

## 🧪 Example Inference

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="erdemyavuz55/drug-review-sentiment-analysis-bert"
)

classifier("This medication worked great with minimal side effects.")
```

Example Output:

```
[{'label': 'POSITIVE', 'score': 0.97}]
```

---

## 📈 Comparison with Classical Models

In the broader study, we compared:

* TF-IDF + Logistic Regression
* TF-IDF + SVM
* Word2Vec + LSTM
* Random Forest
* Gradient Boosting

The fine-tuned BERT model achieved superior macro-F1 in the three-class setting.

---

## ⚠️ Limitations

* Trained only on English reviews
* Domain-specific slang may reduce accuracy
* Ratings-based labeling may introduce subjectivity bias
* No domain-adaptive pretraining performed

---

## 🚀 Future Work

* Domain-adaptive pretraining on medical corpora
* Model interpretability (SHAP, attention visualization)
* Deployment as REST API
* Cross-dataset generalization experiments

## ⚙️ Installation & Usage

If you want to run this project locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/ErdemYavuz55/drug-review-sentiment-analysis.git
cd drug-review-sentiment-analysis

2.Install the required dependencies:
pip install -r requirements.txt

3.(Optional) Run the Streamlit app locally:
streamlit run app.py
```
## 📝 License

This project is licensed under the MIT License.

