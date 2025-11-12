# %%
import os
import pandas as pd

# %%
# Load data frame
data = []

for dirname, _, filenames in os.walk(r'C:\Users\Aditya\Desktop\Project GitHub\nucleusbi_assignment\archive'):
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(dirname, filename)
            with open(file_path, 'r') as f:
                text = f.read()
            data.append({'file_name': filename, 'text': text})

df = pd.DataFrame(data)
df.head()

# %%
# Check rows and cols
df.shape

# %%
# See the example of text column
print(df.loc[0, 'text'])

# %% [markdown]
# ### Text preprocessing
# Text cleaning and normalization

# %%
import re
import string
from nltk.corpus import stopwords
import nltk

# %%
# Get stopwords
#nltk.download('stopwords')

# Create a function to process the text
def preprocess(text):
    # Remove speaker labels
    text = re.sub(r'\b(D|P)\s*:\s*', '', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove custom words
    text = re.sub(r'\b(hi|hey|yeah|ok|okay|um|uh|uhm|umm|like|well||im|really|hasnt|think|ive|hes|shes|anything|anybody|alright|dont|could)\b', '', text, flags=re.IGNORECASE)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove whitespace
    text = re.sub(r'\s+', ' ', text)
    # Tokenize
    tokens = text.split()
    # Include the custom words
    stop_words = set(stopwords.words('english'))
    # Filter tokens
    tokens = [t for t in tokens if t not in stop_words]
    # Remove consecutive duplicate words
    cleaned_tokens = []
    for i, t in enumerate(tokens):
        if i == 0 or t != tokens[i - 1]:
            cleaned_tokens.append(t)

    return ' '.join(cleaned_tokens)

df['text_clean'] = df['text'].apply(preprocess)
df.head()

# %%
# Store df to json
df.to_json('df_preprocessed.json')

# %% [markdown]
# ### Find topic
# TF-IDF vectorizer and latent semantic analysis (LSA)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# %%
# Drop NaN in text_clean column
texts = df['text_clean'].dropna().tolist() 

# Convert to TF-IDF representation
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(texts)

# Perform SVD on the TF-IDF matrix
topic_size = 3 
lsa = TruncatedSVD(n_components=topic_size, random_state=42)
lsa_vectors = lsa.fit_transform(tfidf)

# %%
# Print the topics learned by the LSA model
topics = []

terms = tfidf_vectorizer.get_feature_names_out()

for i, comp in enumerate(lsa.components_):
    row = {}
    row['topic_num'] = i
    row['topics'] = [terms[j] for j in comp.argsort()[:-6:-1]]  
    print(f"topic {i}: {', '.join(row['topics'])}")
    topics.append(row)

# %%
# Convert to data frame
df_topic = pd.DataFrame(topics)

# Add label column
df_topic['label'] = 0

# Save to csv for manual label
df_topic.to_csv('text_topic_label.csv', index=False)

print('Now label manually!')
print(
    {
        'general illness': 'general symptoms probably short-term illness',
        'joint illness': 'joint or musculoskeletal injuries',
        'chronic illness': 'serious symptomps probably long-term illness'
    }
)

# %%



