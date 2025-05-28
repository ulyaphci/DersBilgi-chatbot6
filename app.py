# app.py
import streamlit as st
import pandas as pd
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
from datetime import datetime

# NLTK setup (only stopwords now)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)

# Load Excel data
column_names = [
    "sınıf", "ders_adı", "hoca-adı", "gün1", "saat1", "derslik1",
    "gün2", "saat2", "derslik2", "vizetarihi", "saat.1",
    "finaltarihi", "saat.2", "butunlemetarihi", "saat.3"
]
df = pd.read_excel("ders_bilgi.xlsx", names=column_names, header=1)

# Preprocessing
stop_words = set(stopwords.words("turkish"))

def preprocess_text(text):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    return " ".join([w for w in tokens if w not in stop_words])

df["document"] = df.apply(lambda row: f"{row['sınıf']} {row['ders_adı']} {row['hoca-adı']} "
                                      f"{row['gün1']} {row['saat1']} {row['derslik1']} "
                                      f"{row['gün2']} {row['saat2']} {row['derslik2']} "
                                      f"{row['vizetarihi']} {row['saat.1']} "
                                      f"{row['finaltarihi']} {row['saat.2']} "
                                      f"{row['butunlemetarihi']} {row['saat.3']}", axis=1)

df["processed"] = df["document"].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["processed"])

def find_best_match(user_input):
    processed_input = preprocess_text(user_input)
    user_vec = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_idx = similarities.argmax()
    return df.iloc[best_idx]

def extract_info(question, row):
    q = question.lower()

    if re.search(r'\b(selam|merhaba|günaydın|iyi akşamlar|nasılsın)\b', q):
        return "Merhaba! Size nasıl yardımcı olabilirim? 😊"

    if "final" in q:
        return f"{row['ders_adı']} dersi finali: {row['finaltarihi']} Saat: {row['saat.2']}"
    if "vize" in q:
        return f"{row['ders_adı']} dersi vizesi: {row['vizetarihi']} Saat: {row['saat.1']}"
    if "büt" in q or "bütünleme" in q:
        return f"{row['ders_adı']} dersi bütünlemesi: {row['butunlemetarihi']} Saat: {row['saat.3']}"

    sinif_match = re.search(r'(\d)\.?\s*sınıf', q)
    if "hangi sınıf" in q or "kaçıncı sınıf" in q:
        return f"{row['ders_adı']} dersi {row['sınıf']}. sınıf dersidir."
    elif sinif_match:
        sinif = int(sinif_match.group(1))
        dersler = df[df["sınıf"] == sinif]["ders_adı"].unique()
        return f"{sinif}. sınıf dersleri:\n" + "\n".join(f"- {d}" for d in dersler)

    gunler = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma"]
    for gun in gunler:
        if gun in q:
            if sinif_match:
                sinif = int(sinif_match.group(1))
                dersler = df[((df["gün1"].str.lower() == gun) | (df["gün2"].str.lower() == gun)) & (df["sınıf"] == sinif)]
            else:
                dersler = df[(df["gün1"].str.lower() == gun) | (df["gün2"].str.lower() == gun)]
            if not dersler.empty:
                return f"{gun.capitalize()} günü dersler:\n" + "\n".join(f"- {d}" for d in dersler["ders_adı"].unique())
            return f"{gun.capitalize()} günü için ders bulunamadı."

    if "bugün" in q:
        weekday = datetime.today().strftime("%A").lower()
        gun_map = {"monday": "pazartesi", "tuesday": "salı", "wednesday": "çarşamba",
                   "thursday": "perşembe", "friday": "cuma"}
        gun = gun_map.get(weekday, "")
        dersler = df[(df["gün1"].str.lower() == gun) | (df["gün2"].str.lower() == gun)]
        return f"Bugün ({gun}) olan dersler:\n" + "\n".join(f"- {d}" for d in dersler["ders_adı"].unique()) if not dersler.empty else "Bugün ders yok."

    if "derslik" in q or "nerede" in q:
        return f"{row['ders_adı']} dersi:\n- {row['gün1']}: {row['derslik1']}\n- {row['gün2']}: {row['derslik2']}"

    for hoca in df["hoca-adı"].dropna().unique():
        ad = hoca.split()[0].lower()
        if ad in q:
            hoca_dersleri = df[df["hoca-adı"].str.lower().str.contains(ad)]["ders_adı"].unique()
            return f"{hoca} hocanın verdiği dersler:\n" + "\n".join(f"- {d}" for d in hoca_dersleri)

    date_match = re.search(r"\d{2}\.\d{2}\.\d{4}", question)
    if date_match:
        date_str = date_match.group()
        matched_rows = df[
            (df["vizetarihi"].astype(str) == date_str) |
            (df["finaltarihi"].astype(str) == date_str) |
            (df["butunlemetarihi"].astype(str) == date_str)
        ]
        if not matched_rows.empty:
            response = f"{date_str} tarihinde yapılan sınav(lar):\n"
            for _, r in matched_rows.iterrows():
                if str(r["vizetarihi"]) == date_str:
                    response += f"- {r['ders_adı']} (Vize)\n"
                if str(r["finaltarihi"]) == date_str:
                    response += f"- {r['ders_adı']} (Final)\n"
                if str(r["butunlemetarihi"]) == date_str:
                    response += f"- {r['ders_adı']} (Bütünleme)\n"
            return response
        return f"{date_str} tarihinde herhangi bir sınav bulunamadı."

    return f"{row['ders_adı']} dersi hakkında bilgi: {row['sınıf']}. sınıf, Vize: {row['vizetarihi']}, Final: {row['finaltarihi']}"

# Streamlit UI
st.title("📚 Ders Bilgi Chatbotu")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Sorunuzu yazınız...")

if user_input:
    st.session_state.history.append(("user", user_input))
    match_row = find_best_match(user_input)
    response = extract_info(user_input, match_row)
    st.session_state.history.append(("assistant", response))

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)
