# ============================================
# PHISHING EMAIL DETECTION MODEL
# ============================================

# Install Required Libraries First:
# pip install pandas scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------
# SAMPLE DATASET
# --------------------------------------------

data = {
    "email": [
        "Congratulations! You won a free iPhone. Click here now!",
        "Your bank account has been suspended. Verify immediately.",
        "Meeting scheduled for tomorrow at 10 AM.",
        "Project report attached. Please review.",
        "Claim your lottery prize now!!!",
        "Update your password using this urgent link.",
        "Lunch at 1 PM today?",
        "Can you send me the assignment notes?",
        "Limited offer! Win cash rewards instantly.",
        "Your Amazon account needs verification."
    ],

    "label": [
        "Phishing",
        "Phishing",
        "Safe",
        "Safe",
        "Phishing",
        "Phishing",
        "Safe",
        "Safe",
        "Phishing",
        "Phishing"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# --------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["email"])
y = df["label"]

# --------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# --------------------------------------------
# TRAIN MODEL
# --------------------------------------------

model = MultinomialNB()
model.fit(X_train, y_train)

# --------------------------------------------
# PREDICTION
# --------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------
# EVALUATION
# --------------------------------------------

print("\n===================================")
print(" PHISHING EMAIL DETECTION RESULTS ")
print("===================================\n")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------
# TEST CUSTOM EMAIL
# --------------------------------------------

print("\n===================================")
print(" TEST YOUR OWN EMAIL ")
print("===================================\n")

custom_email = input("Enter an email message:\n")

custom_features = vectorizer.transform([custom_email])

prediction = model.predict(custom_features)

print("\nPrediction:", prediction[0])

if prediction[0] == "Phishing":
    print("⚠ Warning: This email looks suspicious!")
else:
    print("✅ This email appears safe.")