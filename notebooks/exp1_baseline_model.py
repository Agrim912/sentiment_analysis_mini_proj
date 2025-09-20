import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
import mlflow
mlflow.set_experiment("Sentiment Analysis Experiment using baseline model")

mlflow.set_tracking_uri("")
df =pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Twitter-Sentiment-Analysis/refs/heads/master/train_tweet.csv")
print("Shape: ",df.shape)
print(df['label'].value_counts())

X=df.drop(columns=['id','label'])
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply normalization to the tweet text
X_train['tweet'] = X_train['tweet'].apply(lambda x: normalize(x, use_stemming=False))
X_test['tweet'] = X_test['tweet'].apply(lambda x: normalize(x, use_stemming=False))

# Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train['tweet']).toarray()
X_test_tfidf = vectorizer.transform(X_test['tweet']).toarray()

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
print("Evaluating model on test set...")
y_pred = model.predict(X_test_tfidf)


accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Get a comprehensive report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


with mlflow.start_run():
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("max_features", 500)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # set tags
    mlflow.set_tag("developer", "Agrim")
    

    # Log the model
    mlflow.sklearn.log_model(model, "logistic_regression_model")

    # log data
    mlflow.log_artifact("notebooks/exp1_baseline_model.py")
    mlflow.log_artifact("notebooks/preprocessing.py")