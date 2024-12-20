import streamlit as st
import joblib
import nltk

# Load the saved model and vectorizer from disk
naive_bayes_model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to preprocess input text (same as in training)
def preprocess_text(text):
    text = text.lower()  # Lower case conversion
    tokens = nltk.word_tokenize(text)  # Tokenization
    tokens = [PorterStemmer().stem(word) for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(tokens)

# Streamlit App Layout
st.title("Spam Classifier")
st.write("Enter an email to check if it's spam or not.")

# Input field for user to enter an email to classify as spam or not spam
email_input = st.text_area("Enter the email content here:")

if st.button("Classify"):
    if email_input:
        processed_email = preprocess_text(email_input)
        email_vectorized = vectorizer.transform([processed_email])
        prediction = naive_bayes_model.predict(email_vectorized)

        if prediction[0] == 1:
            st.success("This email is classified as **Spam**.")
        else:
            st.success("This email is classified as **Not Spam**.")
