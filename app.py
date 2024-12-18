import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
with open("classifierRF.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("vectorizerRF.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_sentiment(text):
    # Transform input text menggunakan vectorizer
    text_vectorized = vectorizer.transform([text])
    # Prediksi menggunakan model Random Forest
    prediction = classifier.predict(text_vectorized)
    return prediction[0]

def main():
    st.title("Prediksi Sentimen Terhadap Kenaikan PPN 12%")
    st.subheader("Input teks untuk mendapatkan prediksi sentimen: positif atau negatif")

    # Input text dari user
    user_input = st.text_input("Masukkan teks:", placeholder="Contoh: Kenaikan PPN ini terlalu tinggi!")

    if st.button("Prediksi"):
        if user_input.strip() == "":
            st.error("Teks tidak boleh kosong!")
        else:
            sentiment = predict_sentiment(user_input)
            st.success(f"Hasil prediksi: **{sentiment}**")

if __name__ == "__main__":
    main()
