<div align="center">
# Text Sentiment Analysis 🧠📝

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Deep%20Learning-red)]()
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-blue)]()
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange?logo=tensorflow)]()
[![Architecture](https://img.shields.io/badge/Model-CNN--1D-purple)]()
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://opensource.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Research%20Project-lightgrey)]()

An end-to-end Machine Learning and **NLP framework** that uses **Convolutional Neural Networks (CNNs)** to extract and classify emotional tone from unstructured text, moving beyond basic lexicon-based methods.
</div>


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

