import joblib
import os 
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key =os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Load the trained model
model = joblib.load("models/fake_news_model.pkl")

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("=" * 50)
print("        PROJECT RECON ")
print("=" * 50)

# Get news from user
news = input("\n Paste a news article:\n\n> ")

#convert the nws article info TF-IDF feature 
news_vector = vectorizer.transform([news])

#predicting using the trained model
prediction = model.predict(news_vector)
confidence = model.predict_proba(news_vector)

#converting result into redable text 
result = 'REAL' if prediction[0]==1 else 'FAKE'

#heighest confidence perfcentage 
confidence_score = max(confidence[0])*100


print("\n" + "=" * 50)
print(f"Prediction : {result}")
print(f"Confidence : {confidence_score:.2f}%")
print("=" * 50)
