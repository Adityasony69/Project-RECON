import joblib

# Load the trained model
model = joblib.load("models/fake_news_model.pkl")

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("=" * 50)
print("        PROJECT RECON ")
print("=" * 50)

# Get news from user
news = input("\n📰 Paste a news article:\n\n> ")