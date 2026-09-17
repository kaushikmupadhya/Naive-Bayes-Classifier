# Naive Bayes Spam–Ham Classifier
An NLP pipeline that classifies SMS messages as **spam** or **ham** using a **Multinomial Naive Bayes** model with TF-IDF features.

Same notebook is also on Kaggle:
https://www.kaggle.com/code/kaushikmanjunatha/natural-language-processing-spam-ham-classifier

---

## Overview
Short messages are loaded from multiple labelled training files, cleaned with NLTK, converted to TF-IDF vectors, and classified with scikit-learn’s MultinomialNB. The model is validated on a held-out split, tested on the labelled SMS Spam Collection, then used to label an unlabelled test set. A small “cheat the classifier” experiment checks how the model behaves when spam and ham wording are mixed.
Validation and labelled-test accuracy are in the ~99% range.

---

## What’s in this repo
| File | Role |
|---|---|
| Naive_Bayes_Classifier.ipynb | Full walkthrough (recommended) |
| naive_bayes_classifier.py | Same pipeline as a Python script |
| TrainDataset1.csv / TrainDataset2.csv / TrainDataset3.txt | Labelled training data (type, text) |
| SMSSpamCollection.txt | Labelled test set (tab-separated) |
| TestDataset.csv | Unlabelled messages for prediction |
| LICENSE | MIT |

---

## Pipeline
1. Load three training sources and concatenate them.
2. Preprocess each message: strip punctuation, drop English stopwords, Porter stem, WordNet lemmatize.
3. Explore class balance (ham vs spam) with bar and pie charts.
4. Train TfidfVectorizer + MultinomialNB in a scikit-learn Pipeline (spam = 1, ham = 0; 80/20 split, random_state=5).
5. Validate with accuracy, classification report, and a seaborn confusion-matrix heatmap.
6. Test on SMSSpamCollection.txt.
7. Predict labels for TestDataset.csv and plot the predicted mix.
8. Stress-test with mixed spam/ham wording to see where the classifier fails.

---
## Setup
```bash
git clone https://github.com/kaushikmupadhya/Naive-Bayes-Classifier.git
cd Naive-Bayes-Classifier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn nltk
Download NLTK data once:
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---
## Run
Notebook:
```
jupyter notebook Naive_Bayes_Classifier.ipynb
Script (run from the repo root so data files resolve):
python naive_bayes_classifier.py
```
Or open the Kaggle notebook and click Copy & Edit:
`https://www.kaggle.com/code/kaushikmanjunatha/natural-language-processing-spam-ham-classifier`

---

## Model
```
Pipeline([
    ("vectorizer", TfidfVectorizer(analyzer=preprocess)),
    ("classifier", MultinomialNB()),
])
```
Naive Bayes works well on bag-of-words SMS data: it is fast, handles high-dimensional sparse text, and is a strong baseline for spam filtering.

---

## Dataset
Training and evaluation use SMS-style messages labelled ham (legitimate) or spam. 
The labelled collection follows the UCI SMS Spam Collection format:
`https://archive.ics.uci.edu/ml/datasets/sms+spam+collection`
On Kaggle the input is `SMSSpamCollectionDataset`.

---

## License
MIT. See LICENSE.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes..

---

## Contact
If you have any questions or suggestions, feel free to contact me at [Kaushik Manjunatha](https://www.linkedin.com/in/kaushik-manjunatha/)
