import streamlit as st
st.set_page_config(page_title="Fake News Detector", layout="centered")

import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# --- NLTK Downloads and Path Setup ---
# It's good practice to manage NLTK data path for deployment.
# We'll use a local folder named 'nltk_data' in the same directory as this script.
nltk_data_path = "nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path) # Add to NLTK's search path

try:
    # Use quiet=True for deployment to avoid verbose output in Streamlit console
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
    # print("NLTK resources verified/downloaded for Streamlit.") # Uncomment for debugging
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}. Please ensure you have internet access or download manually.")
    st.stop() # Stop the app if NLTK data is critical and missing

# --- Load Models and Vectorizer ---
@st.cache_resource # This decorator caches the function result to run only once
def load_resources():
    try:
        lr_model = joblib.load('lr_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return lr_model, tfidf_vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer files not found. Make sure 'lr_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        st.stop()

lr_model, tfidf_vectorizer = load_resources()

# --- NLP Preprocessing Function (Exactly copied from your notebook) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', ' ', text) # Replace newline characters with spaces
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers

    tokens = word_tokenize(text) # Tokenize text into words
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()] # Remove stopwords and non-alphabetic tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lemmatize tokens

    return " ".join(tokens)

st.title("📰 Fake News Detector")
st.markdown("---")
st.write("Enter the details of a news article below to find out if it's likely real or fake.")

# Input fields
news_title = st.text_input("News Title:", placeholder="Enter the headline of the news article")
news_subject = st.text_input("News Subject:", placeholder="e.g., Politics, World News, Health")
news_text = st.text_area("News Article Content:", height=250, placeholder="Paste the full text of the news article here...")

# Predict button
if st.button("Analyze News"):
    if not news_title and not news_subject and not news_text:
        st.warning("Please enter some text in at least one field to analyze.")
    else:
        # Combine inputs into a single string, similar to training data
        full_news_content = str(news_title) + " " + str(news_subject) + " " + str(news_text)

        # Preprocess the combined text
        processed_input = preprocess_text(full_news_content)

        if not processed_input.strip():
            st.warning("The entered text was too short or became empty after processing. Please provide more meaningful content.")
        else:
            # Transform the processed text using the loaded TF-IDF vectorizer
            # Note: tfidf_vectorizer.transform expects an iterable (like a list)
            input_vectorized = tfidf_vectorizer.transform([processed_input])

            # Make prediction
            prediction = lr_model.predict(input_vectorized)
            prediction_proba = lr_model.predict_proba(input_vectorized)

            st.markdown("---")
            st.subheader("Prediction Result:")

            if prediction[0] == 1: # 1 means Real
                st.success("✅ This news is likely **REAL**!")
                st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
            else: # 0 means Fake
                st.error("❌ This news is likely **FAKE**!")
                st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")

            st.markdown("---")
            st.info("Disclaimer: This is an AI-powered prediction based on trained data. Always cross-reference information with multiple reputable sources for absolute verification.")





# import streamlit as st
# # --- Streamlit User Interface (UI) ---
# st.set_page_config(page_title="Fake News Detector", layout="centered")
# # --- Import Required Libraries ---
# import joblib
# import re
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import os

# # --- NLTK Downloads and Path Setup ---
# nltk_data_path = "nltk_data"
# if not os.path.exists(nltk_data_path):
#     os.makedirs(nltk_data_path)
# nltk.data.path.append(nltk_data_path)

# try:
#     nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
#     nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
#     nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
# except Exception as e:
#     st.error(f"Error downloading NLTK resources: {e}. Please ensure you have internet access or download manually.")
#     st.stop()

# # --- Load Models and Vectorizer ---
# @st.cache_resource
# def load_resources():
#     try:
#         lr_model = joblib.load('lr_model.pkl')
#         tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
#         return lr_model, tfidf_vectorizer
#     except FileNotFoundError:
#         st.error("Error: Model or Vectorizer files not found. Make sure 'lr_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as this script.")
#         st.stop()
#     except Exception as e:
#         st.error(f"Error loading model/vectorizer: {e}")
#         st.stop()

# lr_model, tfidf_vectorizer = load_resources()

# # --- NLP Preprocessing Function ---
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""

#     text = text.lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>+', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\n', ' ', text)
#     text = re.sub(r'\w*\d\w*', '', text)

#     tokens = word_tokenize(text)
#     tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]

#     return " ".join(tokens)



# st.title("📰 Fake News Detector")
# st.markdown("---")
# st.write("Paste the news article content (or just the headline) below to find out if it's likely real or fake.")

# # --- ONLY ONE INPUT FIELD NOW ---
# news_content_input = st.text_area("News Article Text:", height=100, placeholder="Paste the full news article or just the headline here...")

# # Predict button
# if st.button("Analyze News"):
#     if not news_content_input: # Check if the single input field is empty
#         st.warning("Please enter some text to analyze.")
#     else:
#         # Use the single input directly for preprocessing
#         full_news_content = str(news_content_input) # Ensure it's a string

#         # Preprocess the combined text
#         processed_input = preprocess_text(full_news_content)

#         if not processed_input.strip():
#             st.warning("The entered text was too short or became empty after processing. Please provide more meaningful content.")
#         else:
#             # Transform the processed text using the loaded TF-IDF vectorizer
#             input_vectorized = tfidf_vectorizer.transform([processed_input])

#             # Make prediction
#             prediction = lr_model.predict(input_vectorized)
#             prediction_proba = lr_model.predict_proba(input_vectorized)

#             st.markdown("---")
#             st.subheader("Prediction Result:")

#             if prediction[0] == 1: # 1 means Real
#                 st.success("✅ This news is likely **REAL**!")
#                 st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
#             else: # 0 means Fake
#                 st.error("❌ This news is likely **FAKE**!")
#                 st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")

#             st.markdown("---")
#             st.info("Disclaimer: This is an AI-powered prediction based on trained data. Always cross-reference information with multiple reputable sources for absolute verification.")