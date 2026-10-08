
from fastapi import FastAPI
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="Batch Sentiment Analysis API")

analyzer = SentimentIntensityAnalyzer()


class SentimentRequest(BaseModel):
    sentences: list[str]


def classify_sentiment(sentence: str) -> str:
    # Handle empty or whitespace-only sentences
    if not sentence.strip():
        return "neutral"

    scores = analyzer.polarity_scores(sentence)
    compound = scores["compound"]

    # VADER standard thresholds
    if compound >= 0.05:
        return "happy"
    elif compound <= -0.05:
        return "sad"
    else:
        return "neutral"


@app.post("/sentiment")
def analyze_sentiment(request: SentimentRequest):
    results = []

    for sentence in request.sentences:
        sentiment = classify_sentiment(sentence)

        results.append({
            "sentence": sentence,
            "sentiment": sentiment
        })

    return {"results": results}


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}
