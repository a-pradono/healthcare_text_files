# %%
import pandas as pd
import ast

# %%
# Load topic data frame
df_topic_label = pd.read_csv('text_topic_label.csv')

# Load json preprocessed data frame
df = pd.read_json('df_preprocessed.json')

# %%
df_topic_label

# %% [markdown]
# ### Find topic and label
# Topic from text and then label from topic
# 

# %%
# Convert topic strings to lists
if isinstance(df_topic_label['topics'].iloc[0], str):
    df_topic_label['topics'] = df_topic_label['topics'].apply(ast.literal_eval)

# %%
# Function to find topic from text
def find_topic(text):
    selected_topic = None
    best_score = 0

    if not isinstance(text, str) or not text.strip():
        return None

    text_words = set(text.lower().split())

    for _, row in df_topic_label.iterrows():
        topic_words = set(w.lower() for w in row['topics'])
        overlap = len(text_words & topic_words)

        if overlap > best_score:
            best_score = overlap
            selected_topic = row['topic_num']

    return selected_topic

# %%
# Function to find label from topic
def find_label(topic_num):
    if pd.isna(topic_num):
        return 'unknown'

    match = df_topic_label[df_topic_label['topic_num'] == topic_num]
    if len(match) == 0:
        return 'unknown'
    return match.iloc[0]['label']

# %%
# Apply to data frame
df['topic'] = df['text_clean'].apply(find_topic)
df['label'] = df['topic'].apply(find_label)
df.head()

# %%
# Check null values
df[df['topic'].isnull()]

# %%
# Rows and cols
df.shape

# %%
# Save df to json
df.to_json('df_final.json')

# %%



