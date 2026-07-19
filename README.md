# Text Sentiment Analysis 🧠📝

[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg)](https://opensource.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)]()
[![NLP Tooling](https://img.shields.io/badge/NLP-NLTK%20%7C%20SpaCy-blue.svg)]()

An end-to-end Machine Learning and **NLP framework** that uses **Convolutional Neural Networks (CNNs)** to extract and classify emotional tone from unstructured text, moving beyond basic lexicon-based methods.

---

## 📸 System Architecture & Visual Pipeline

```text
[Raw Text Input] 
       │
       ▼
[Advanced NLP Preprocessing] (Tokenization ➔ Punctuation/Number Removal ➔ Stemming/Lemmatization)
       │
       ▼
[Vector Embedding Layer] (Dense Word Translation Array using TF IDF)
       │
       ▼
[1D Convolutional Network (CNN)] (Local Feature Extraction & Feature Maps)
       │
       ▼
[Global Max Pooling & Dense Matrix] ➔ [Softmax Classifier] ➔ [Predicted Sentiment Output]
```

---

## 🛠️ Core Technical Specs & Tags

*   **Core Frameworks:** NumPy, Pandas, PyTorch
*   **NLP Tooling Stack:** NLTK (Natural Language Toolkit), SpaCy, Scikit-Learn
*   **Architecture Mapping:** Convolutional Neural Networks (CNN-1D)
*   **Project Timeline:** Independent Research Sprint (< 1 month)

---

## 📈 Model Performance & Metrics

The integration of advanced preprocessing paired with local CNN feature maps outperformed traditional recurrent (LSTM) baseline structures in both training velocity and parameter stability:

| Pipeline Variant        | Preprocessing Level                 | Accuracy Score | Compute Training Speed | Overfitting Drift         |
| :---------------------- | :---------------------------------- | :------------- | :--------------------- | :------------------------ |
| **Baseline Lexicon**    | None                                | 62.4%          | Instant                | N/A                       |
| **Basic RNN**           | Tokenization Only                   | 76.8%          | Slow (High Epochs)     | Moderate                  |
| **Deep CNN V1**         | Stemming + Noise Clean              | 84.2%          | Very Fast              | High                      |
| **Deep CNN V2 (Final)** | **Full Lemmatization + CNN Matrix** | **91.5%**      | **Fast & Optimized**   | **Minimal (Regularized)** |

---

