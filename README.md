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

📝 License
This project is licensed under the MIT License.
