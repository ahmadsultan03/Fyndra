# copyrights @ Fyndra v1.0.0 - Muhammad Ahmad Sultan, Shandana Iftikhar, Samar Qaiser

# Required Python Libraries for Fyndra

import requests
from bs4 import BeautifulSoup
import re
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Downloading the required NLTK data

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Globals
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Simple Web Crawler v1.0.0
def crawl_website(url, limit=50):
    pages = []
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') and url.split('//')[1].split('/')[0] in href:
                pages.append(href)
            if len(pages) >= limit:
                break
    except:
        pass
    return list(set(pages))

def extract_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else ""
        body = soup.get_text(separator=' ')
        return title + " " + body
    except Exception as e:
        print(f"Failed to extract text from {url}: {e}")
        return ""

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Indexing
def index_documents(docs):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    return X, vectorizer

# Query Handling
def handle_query(query, vectorizer, doc_vectors):
    query_processed = preprocess(query)
    query_vector = vectorizer.transform([query_processed])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1]  # Rank by similarity score

    relevant_indices = ranked_indices  
    top_k = 5  # Define top_k here if you want to limit the number of results
    
    # Pass top_k directly to the evaluate function
    precision, recall, f1 = evaluate(query, ranked_indices, relevant_indices, top_k)
    print(f"\nEvaluation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return ranked_indices, similarities


def get_relevant_indices_for_query(query, doc_vectors, vectorizer, threshold):
    # Preprocess the query and transform it into vector space
    query_processed = preprocess(query)
    query_vector = vectorizer.transform([query_processed])
    
    # Calculate cosine similarity between query and document vectors
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Filter documents with similarity above the threshold
    relevant_indices = [i for i, similarity in enumerate(similarities) if similarity >= threshold]
    
    return relevant_indices

def evaluate(query, retrieved_indices, relevant_indices, top_k):
    # Limit the evaluation to top_k results from the retrieved documents
    top_k_retrieved = retrieved_indices[:top_k]
    
    # Ground truth: relevant documents
    true_positives = len([idx for idx in top_k_retrieved if idx in relevant_indices])
    false_positives = len([idx for idx in top_k_retrieved if idx not in relevant_indices])
    false_negatives = len([idx for idx in relevant_indices if idx not in top_k_retrieved])

    # Precision = TP / (TP + FP)
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    # Recall = TP / (TP + FN)
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # Return precision, recall, and F1 score for all results in top_k
    print(f"Evaluation for Query: '{query}'")
    print(f"Top {top_k} results:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return precision, recall, f1



# Visualization
def visualize_vectors(doc_vectors, urls):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(doc_vectors.toarray())
    
    # Ensure the number of URLs matches the number of reduced points
    if len(urls) != len(reduced):
        print(f"Warning: Mismatch between URLs ({len(urls)}) and reduced points ({len(reduced)})")

    # Scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=np.arange(len(reduced)), cmap='viridis', edgecolor='k', s=100)
    
    # Add labels to the points
    for i, url in enumerate(urls):
        plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, f'{i+1}', fontsize=9, ha='center')
    
    # Add color bar to show the mapping of colors to document indices
    plt.colorbar(scatter, label="Document Index")
    
    # Add title and labels
    plt.title("Document Clustering - PCA Projection", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    
    # Show grid for better readability
    plt.grid(True)
    
    # Show plot
    plt.show()


# <------------------ Main Execution ------------------>

if __name__ == "__main__":
    base_urls = [
        "https://www.daraz.pk",
        "https://www.priceoye.pk",
        "https://www.goto.com.pk",
        "https://www.telemart.pk",
        "https://www.yayvo.com"
    ]

    print("[+] Crawling and extracting data...")
    urls = []
    for base in base_urls:
        urls.extend(crawl_website(base, limit=20))

    raw_docs = [extract_text(url) for url in urls]
    
    # Only keep valid documents (non-empty ones)
    processed_docs = [preprocess(doc) for doc in raw_docs if doc.strip() != ""]

    # Ensure the processed_docs and urls are the same length
    urls = [url for url, doc in zip(urls, raw_docs) if doc.strip() != ""]

    print("[+] Indexing documents...")
    doc_vectors, vectorizer = index_documents(processed_docs)

    print("[+] Ready for search queries.")
    while True:
        query = input("Enter search query (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        ranked_indices, similarities = handle_query(query, vectorizer, doc_vectors)
        print("\nTop 5 relevant documents:")
        for i in ranked_indices[:5]:
            print(f"- URL: {urls[i]}\n  Score: {similarities[i]:.4f}\n")

    # Visualizing document vectors
    visualize_vectors(doc_vectors, urls)
    
    # Closing Brackets Properly