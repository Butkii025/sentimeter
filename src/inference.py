"""
Unified inference wrapper. Works today with the classic TF-IDF + LogReg model,
and switches to the fine-tuned DistilBERT model the moment BACKEND="transformer"
and HF_REPO points at your pushed model.
"""

import os
import joblib

BACKEND = os.environ.get("SENTIMENT_BACKEND", "classic")   # "classic" or "transformer"
HF_REPO = os.environ.get("HF_REPO", "your-username/distilbert-sentiment-imdb")

# Project root = parent of this file's directory (src/), so model paths work
# no matter what directory you run the script from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class SentimentAnalyzer:
    def __init__(self, backend=BACKEND):
        self.backend = backend
        if backend == "classic":
            self.clf = joblib.load(os.path.join(MODELS_DIR, "logreg_sentiment.pkl"))
            self.vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        elif backend == "transformer":
            from transformers import pipeline
            self.pipe = pipeline(
                "text-classification",
                model=HF_REPO,
                top_k=None,        # return scores for both classes
                truncation=True,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict(self, text: str) -> dict:
        """Returns {'label': 'positive'|'negative', 'scores': {'positive': p, 'negative': n}}"""
        if not text or not text.strip():
            return {"label": None, "scores": {}}

        if self.backend == "classic":
            import re
            clean = re.sub(r"<.*?>", " ", text)
            clean = re.sub(r"[^a-zA-Z\s]", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip().lower()

            X = self.vectorizer.transform([clean])
            pred = self.clf.predict(X)[0]
            # decision_function gives a margin, not a probability -- squash it for a
            # confidence-like number using a sigmoid
            import math
            margin = self.clf.decision_function(X)[0]
            pos_score = 1 / (1 + math.exp(-margin))
            return {
                "label": "positive" if pred == 1 else "negative",
                "scores": {"positive": round(pos_score, 4), "negative": round(1 - pos_score, 4)},
            }

        else:  # transformer
            results = self.pipe(text)[0]
            scores = {r["label"].lower(): round(r["score"], 4) for r in results}
            label = max(scores, key=scores.get)
            return {"label": label, "scores": scores}


if __name__ == "__main__":
    sa = SentimentAnalyzer(backend="classic")
    for s in [
        "This movie completely blew me away, incredible acting.",
        "Absolute waste of two hours, terrible pacing throughout.",
    ]:
        print(s, "->", sa.predict(s))