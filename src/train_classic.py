"""
Classic ML baseline: TF-IDF + Logistic Regression.
Also fits a LinearSVC for comparison since it's nearly free once TF-IDF is built.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])
y_train, y_test = train_df["label"], test_df["label"]
print(f"TF-IDF matrix: {X_train.shape}")

print("\n--- Logistic Regression (grid search over C) ---")
param_grid = {"C": [0.1, 1, 3, 10]}
lr_search = GridSearchCV(
    LogisticRegression(max_iter=1000), param_grid, cv=3, scoring="f1", n_jobs=-1
)
lr_search.fit(X_train, y_train)
print("Best C:", lr_search.best_params_)

lr_preds = lr_search.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds)
print(f"Logistic Regression -> accuracy: {lr_acc:.4f}, f1: {lr_f1:.4f}")
print(classification_report(y_test, lr_preds, target_names=["negative", "positive"]))
print("Confusion matrix:\n", confusion_matrix(y_test, lr_preds))

print("\n--- LinearSVC (for comparison) ---")
svc = LinearSVC(C=1.0, max_iter=5000)
svc.fit(X_train, y_train)
svc_preds = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_preds)
svc_f1 = f1_score(y_test, svc_preds)
print(f"LinearSVC -> accuracy: {svc_acc:.4f}, f1: {svc_f1:.4f}")

# keep whichever model wins
best_model, best_name = (lr_search.best_estimator_, "logreg") if lr_f1 >= svc_f1 else (svc, "svc")
print(f"\nBest model: {best_name} (f1={max(lr_f1, svc_f1):.4f})")

joblib.dump(best_model, f"models/{best_name}_sentiment.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print(f"Saved models/{best_name}_sentiment.pkl and models/tfidf_vectorizer.pkl")

# a handful of misclassified examples, worth eyeballing
print("\n--- Sample misclassified reviews (Logistic Regression) ---")
wrong_idx = (lr_preds != y_test.values).nonzero()[0][:5]
for i in wrong_idx:
    true_label = "positive" if y_test.values[i] == 1 else "negative"
    pred_label = "positive" if lr_preds[i] == 1 else "negative"
    text = test_df.iloc[i]["clean_text"][:200]
    print(f"[true={true_label}, pred={pred_label}] {text}...")
