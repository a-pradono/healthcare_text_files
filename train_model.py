# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report

import warnings
warnings.filterwarnings("ignore")

# %%
# Load data frame
df = pd.read_json('df_final.json')
df.head()

# %%
# Check unique label
df['label'].unique()

# %%
# Drop label unknown before training
df = df[df['label'] != 'unknown'].reset_index(drop=True)
df.head()

# %%
# Check imbalance class
df['label'].value_counts(normalize=True)

# %% [markdown]
# ### Model training
# TF-IDF vectorizer and supervised ML models

# %%
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

# %%
# Split the data set into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(df.text_clean.to_numpy(), df.label.to_numpy(), test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Label encoding
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Apply SMOTE for imbalance label
smote = SMOTE(random_state=42, k_neighbors=1)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# %%
# Set models
models = {"LogisticRegression": {"model": LogisticRegression(random_state=42),
                 "params": {"C": [0.001, 0.01, 0.1, 1]}},
          
          "KNN": {"model": KNeighborsClassifier(),
                  "params": {"n_neighbors": [5, 10, 20, 100], 
                             "weights": ["uniform", "distance"]}},
          
          "Random Forest": {"model": RandomForestClassifier(random_state=2806),
                            "params": {"n_estimators": [10, 100],
                                       "max_depth": [100, 200, None],
                                       }},
          
          "NB": {"model": MultinomialNB(),
                  "params": {"alpha": [0.001, 0.01, 0.1, 1]}}}

# %%
# Define grid search
def GridSearch(models, X_train, Y_train, X_test, Y_test):
    scores = {"model":[], "Test F1": [], "best params": []}
    for name, m in models.items():
        gscv = GridSearchCV(m["model"], m["params"], verbose=2, n_jobs=2)
        gscv.fit(X_train, Y_train)
        scores["model"].append(name)
        scores["Test F1"].append(f1_score(gscv.predict(X_test), Y_test, average="macro"))
        scores["best params"].append(gscv.best_params_)
    return scores

model_scores = GridSearch(models, X_train, Y_train, X_test, Y_test)

# %%
# Display the model scores
pd.DataFrame(model_scores)

# %%
# Generating test report for logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)
print(classification_report(Y_test, model.predict(X_test)))

# %%
# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(Y_test, y_pred)

# Visualize
classes = ["general illness", "joint illness", "chronic illness"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")
plt.show()

# %%
# Save tf-idf and model results
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "logistic_regression_model.pkl")


