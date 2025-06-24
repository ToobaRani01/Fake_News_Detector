# 📰 Fake News Detection with NLP & Streamlit

This project is a fake news classifier built using Python, Natural Language Processing (NLP), machine learning, and a simple Streamlit-based web interface. It identifies whether a given news article is *Fake* or *Real*.

---

## 📂 Dataset

The dataset is **not uploaded** to the repository. You can download both CSV files (`fake.csv`, `true.csv`) from this Kaggle source:

🔗 [Fake and Real News Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection).

After downloading, place both files (`fake.csv` and `true.csv`) in the root folder of your project.

---

## 📊 Project Workflow

### Step 1: Dataset Loading & Exploration
- Load both `fake.csv` and `true.csv` using Pandas
- Add a label column (0 = Fake, 1 = Real)
- Merge the two datasets
- Perform basic checks like `.head()`, `.info()`, `.describe()`

### Step 2: Preprocessing
- Check for missing values
- Visualize label balance using Seaborn
- Combine text columns (`title`, `subject`, `text`) into one: `content`

### Step 3: NLP Preprocessing
- Lowercasing, punctuation removal, removing links, digits, etc.
- Tokenization with `word_tokenize`
- Stopword removal using NLTK
- Lemmatization using `WordNetLemmatizer`

### Step 4: Feature Extraction
- Apply TF-IDF with max 3000 features on `processed_content`

### Step 5: Train-Test Split
- Use `train_test_split` (80/20 ratio)

### Step 6: Model Training
- Train a Logistic Regression model (`max_iter=1000`)

### Step 7: Evaluation
- Accuracy, Precision, Recall, F1-score
- Visualize confusion matrix with heatmap

### Step 8: Model Saving for Deployment

Once the model is trained and evaluated, we save the important components using `joblib`:

- `lr_model.pkl` → Trained Logistic Regression model
- `tfidf_vectorizer.pkl` → TF-IDF vectorizer used to convert text to numeric form

These files will be reused in the Streamlit app to load the model instantly and avoid retraining.



## 📦 Required Libraries

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib streamlit joblib
