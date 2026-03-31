# Fake News Identification using NLP

## Overview

This project focuses on **Fake News Identification** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.
The notebook performs:

* **Exploratory Data Analysis (EDA)**
* **Part-of-Speech (POS) Tagging**
* **Named Entity Recognition (NER)**
* **Text Preprocessing**
* **Sentiment Analysis**
* **Topic Modeling (LDA & LSA)**
* **Fake vs Factual News Classification**

The goal is to analyze textual patterns in news articles and build a model that can classify whether a news article is:

* **Fake News**
* **Factual News**

---

## Features

This project includes:

* News data visualization
* POS tagging using **spaCy**
* Named entity analysis
* Text cleaning and preprocessing
* Sentiment analysis using **VADER**
* Topic modeling using:

  * **LDA (Latent Dirichlet Allocation)**
  * **LSA (Latent Semantic Analysis)**
* Classification using:

  * **Logistic Regression**
  * **SGD Classifier**

---

## Technologies Used

### Programming Language

* Python

### Libraries

* pandas
* matplotlib
* seaborn
* spaCy
* nltk
* vaderSentiment
* gensim
* scikit-learn

---

## Project Structure

```bash
fake-news-identification/
│
├── fake_news_identification.ipynb   # Main Jupyter Notebook
├── fake_news_data.csv               # Dataset
└── README.md                        # Project Documentation
```

---

## Dataset

The project uses a CSV dataset named:

```bash
fake_news_data.csv
```

### Expected Important Columns

* `text` → News article content
* `fake_or_factual` → Label indicating whether the article is:

  * `Fake News`
  * `Factual News`

---

## Workflow

### 1. Data Loading

The dataset is loaded using pandas:

```python
data = pd.read_csv("fake_news_data.csv")
```

### 2. Exploratory Data Analysis

Basic inspection of dataset:

* Preview records
* Check missing values / datatypes
* Visualize class distribution

Example:

```python
data.info()
data.head()
```

### 3. POS Tagging

Using **spaCy**, the project extracts:

* Tokens
* Entity types
* POS tags

This helps understand grammatical patterns in fake and factual news.

Example:

```python
def extract_token_tags(doc):
    return [(i.text, i.ent_type_, i.pos_) for i in doc]
```

### 4. Named Entity Recognition (NER)

The project identifies common named entities such as:

* Persons
* Organizations
* Locations
* Dates

This helps compare what kinds of entities appear more often in fake vs factual news.

### 5. Text Preprocessing

The news text is cleaned using the following steps:

* Remove unwanted prefixes
* Convert to lowercase
* Remove punctuation
* Remove stopwords
* Tokenization
* Lemmatization

### 6. N-gram Analysis

The notebook extracts:

* **Unigrams**
* **Bigrams**

This helps identify the most frequent words and phrases after preprocessing.

### 7. Sentiment Analysis

Sentiment analysis is performed using **VADER Sentiment Analyzer**.

Each article gets:

* Sentiment score
* Sentiment label:

  * Positive
  * Neutral
  * Negative

### 8. Topic Modeling

#### LDA (Latent Dirichlet Allocation)

Used to discover hidden topics in fake news articles.

#### LSA (Latent Semantic Analysis)

Used with TF-IDF to analyze semantic patterns and topics.

### 9. Feature Extraction

The cleaned text is converted into numerical features using:

* **Bag of Words**
* **CountVectorizer**
* **TF-IDF** (for topic modeling)

Example:

```python
countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(X)
```

### 10. Model Training

The project trains classification models to predict whether news is fake or factual.

#### Models Used

* **Logistic Regression**
* **SGD Classifier**

Train-test split is used for evaluation.

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, Y, test_size=0.3)
```

### 11. Evaluation

The models are evaluated using:

* Accuracy Score

Example:

```python
accuracy_score(y_pred_lr, y_test)
accuracy_score(y_pred_svm, y_test)
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-identification.git
cd fake-news-identification
```

### 2. Install Dependencies

```bash
pip install pandas matplotlib seaborn spacy nltk vaderSentiment gensim scikit-learn
```

### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Run Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```bash
fake_news_identification.ipynb
```

---

## Required NLTK Downloads

The notebook uses some NLTK resources. If not already installed, run:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

> Note: If `punkt_tab` gives issues, replace it with `punkt`.

---

## Example Output

The project provides:

* Class distribution plots
* POS tag analysis
* Named entity visualizations
* Sentiment comparison charts
* Topic modeling results
* Fake/Factual news classification accuracy

---

## Future Improvements

Possible improvements for this project:

* Use **TF-IDF** for classification instead of only Bag of Words
* Add more ML models:

  * Naive Bayes
  * Random Forest
  * XGBoost
* Use **Deep Learning models**:

  * LSTM
  * GRU
  * BERT
* Improve preprocessing pipeline
* Add confusion matrix and F1-score
* Deploy as a web app using **Streamlit** or **Flask**

---

## Learning Outcomes

This project helps in understanding:

* NLP preprocessing pipeline
* POS tagging and NER
* Sentiment analysis
* Topic modeling
* Text vectorization
* Fake news classification using ML

---

## Author

**Ayush Kansliwal**

---

## License

This project is for **educational and academic purposes**.
