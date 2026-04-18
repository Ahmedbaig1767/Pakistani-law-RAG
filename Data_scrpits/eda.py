"""
============================================================
ADVANCED EDA FOR RAG LEGAL DATASET
============================================================
This script analyzes your dataset to understand:
- text size
- word usage
- chunk quality
- similarity between chunks
- clustering patterns
============================================================
"""

# import libraries
import json
import glob
import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# load data
def load_data(path="data/processed/*.json"):
    """
    Load all JSON files and combine into one DataFrame
    """
    files = glob.glob(path)
    dfs = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df["source"] = file.split("/")[-1]
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


df = load_data()


df["text"] = df["text"].fillna("")   # add this at line 67
df["char_length"] = df["text"].apply(len)  # this is your existing line 68
df["tokens"] = df["text"].apply(word_tokenize)
df["token_count"] = df["tokens"].apply(len)
df["word_length"] = df["text"].apply(lambda x: len(x.split()))
stop_words = set(stopwords.words("english"))
df["tokens_no_stop"] = df["tokens"].apply(
    lambda tokens: [t for t in tokens if t not in stop_words]
)


# -------------------------------------------------------------
# TEXT ANALYSIS
# -------------------------------------------------------------

plt.figure()
sns.histplot(df["word_length"], bins=40, kde=True)
plt.title("Word Length Distribution")
plt.xlabel("Words per Chunk")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.histplot(df["char_length"], bins=40, kde=True)
plt.title("Character Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.boxplot(x=df["token_count"])
plt.title("Token Count Distribution (Boxplot)")
plt.show()

all_tokens = [t for tokens in df["tokens"] for t in tokens]
vocab_size = len(set(all_tokens))
print(f"\n📌 Vocabulary Size: {vocab_size}")

freq = Counter(all_tokens)
common_words = freq.most_common(20)
words, counts = zip(*common_words)

plt.figure()
sns.barplot(x=list(words), y=list(counts))
plt.xticks(rotation=60)
plt.title("Top 20 Words (Before Stopword Removal)")
plt.show()

all_tokens_clean = [t for tokens in df["tokens_no_stop"] for t in tokens]
freq_clean = Counter(all_tokens_clean)
common_words_clean = freq_clean.most_common(20)
words, counts = zip(*common_words_clean)

plt.figure()
sns.barplot(x=list(words), y=list(counts))
plt.xticks(rotation=60)
plt.title("Top 20 Words (After Stopword Removal)")
plt.show()


# N-GRAMS
def plot_ngrams(tokens, n=2, top_k=15, title="N-Grams"):
    ngram_list = list(ngrams(tokens, n))
    freq = Counter(ngram_list).most_common(top_k)
    labels = [" ".join(x[0]) for x in freq]
    values = [x[1] for x in freq]
    plt.figure()
    sns.barplot(x=values, y=labels)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.show()


plot_ngrams(all_tokens_clean, n=2, title="Top Bigrams")
plot_ngrams(all_tokens_clean, n=3, title="Top Trigrams")

wc = WordCloud(width=800, height=400, background_color="white").generate(
    " ".join(all_tokens_clean)
)
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.show()


# -------------------------------------------------------------
# CHUNKING ANALYSIS
# -------------------------------------------------------------

plt.figure()
sns.histplot(df["token_count"], bins=40, kde=True)
plt.title("Chunk Size Distribution")
plt.xlabel("Tokens")
plt.show()

plt.figure()
sns.boxplot(x=df["token_count"])
plt.title("Chunk Size Consistency")
plt.show()

df["too_short"] = df["token_count"] < 50
df["too_long"] = df["token_count"] > 400

counts = [
    df["too_short"].sum(),
    len(df) - df["too_short"].sum() - df["too_long"].sum(),
    df["too_long"].sum()
]

plt.figure()
sns.barplot(x=["Too Short", "Good", "Too Long"], y=counts)
plt.title("Chunk Quality")
plt.show()


# -------------------------------------------------------------
# SEMANTIC INSIGHTS
# -------------------------------------------------------------

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["text"])

tfidf_scores = np.mean(X.toarray(), axis=0)
terms = vectorizer.get_feature_names_out()
top_idx = np.argsort(tfidf_scores)[-20:]

plt.figure()
sns.barplot(x=tfidf_scores[top_idx], y=terms[top_idx])
plt.title("Top TF-IDF Keywords")
plt.show()

sample = X[:50]
sim_matrix = cosine_similarity(sample)

plt.figure()
sns.heatmap(sim_matrix, cmap="coolwarm")
plt.title("Cosine Similarity Heatmap")
plt.show()


# -------------------------------------------------------------
# CLUSTERING
# -------------------------------------------------------------

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray())

plt.figure()
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=clusters, palette="tab10")
plt.title("Cluster Visualization (PCA)")
plt.legend()
plt.show()


# -------------------------------------------------------------
# EMBEDDING READINESS
# -------------------------------------------------------------

plt.figure()
sns.scatterplot(x=df["token_count"], y=df["char_length"])
plt.title("Token vs Character Length")
plt.xlabel("Tokens")
plt.ylabel("Characters")
plt.show()


# -------------------------------------------------------------
# FINAL INSIGHTS
# -------------------------------------------------------------

print("\n================= KEY INSIGHTS =================")
print(f"Total Chunks     : {len(df)}")
print(f"Avg Tokens/Chunk : {df['token_count'].mean():.2f}")
print(f"Vocabulary Size  : {vocab_size}")
print(f"Duplicate Rows   : {df.duplicated().sum()}")
print(f"Too Short Chunks : {df['too_short'].sum()}")
print(f"Too Long Chunks  : {df['too_long'].sum()}")