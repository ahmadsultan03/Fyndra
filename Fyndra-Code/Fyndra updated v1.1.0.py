# copyrights @ Fyndra v1.1.0 - Muhammad Ahmad Sultan, Shandana Iftikhar, Samar Qaiser

import requests  # For making HTTP requests to fetch web page content.
from bs4 import BeautifulSoup  # For parsing HTML and XML documents, extracting data from web pages.
import re  # For regular expression operations, useful for text cleaning and pattern matching.
import nltk  # (Natural Language Toolkit) For various NLP tasks like tokenization, stemming, lemmatization.
import string  # Provides a collection of string constants.
import numpy as np  # For numerical operations, especially array manipulation.
from sklearn.feature_extraction.text import TfidfVectorizer  # For conversion to a matrix of TF-IDF features.
from sklearn.metrics.pairwise import cosine_similarity  # To compute cosine similarity between vectors.
from sklearn.metrics import precision_score, recall_score, f1_score  # For evaluating the performance of the retrieval system.
from sklearn.decomposition import PCA  # (Principal Component Analysis) For dimensionality reduction.
import matplotlib.pyplot as plt  # For creating static, interactive, and animated visualizations in Python.
from rank_bm25 import BM25Okapi  # For implementing the BM25 ranking algorithm, a probabilistic retrieval model.

# Downloading the required NLTK data
nltk.download('punkt')  # Downloads the Punkt sentence tokenizer models for splitting text into sentences/words.
nltk.download('stopwords')  # Downloads the list of common stop words.
nltk.download('wordnet')  # Downloads WordNet, a large lexical database of English, used for lemmatization.
nltk.download('averaged_perceptron_tagger')  # Downloads the Perceptron Tagger model for Part-of-Speech (POS) tagging.

from nltk.corpus import stopwords  # To access the list of NLTK stop words.
from nltk.tokenize import word_tokenize  # To split text into individual words (tokens).
# For lemmatization (reducing words to base form) and stemming (reducing words to root form).
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag  # To perform Part-of-Speech tagging on tokens.
# To access WordNet functionalities, e.g., for POS tag conversion in lemmatization.
from nltk.corpus import wordnet


# Globals
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Enhanced Web Crawler [v1.1.0]
def crawl_website(url, limit=20):
    pages = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}  # Add headers to avoid blocking
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Prioritize links that look like product pages
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            if not href.startswith('http'):
                if href.startswith('/'):
                    href = url + href
                else:
                    href = url + '/' + href
            
            # Better URL filtering
            if (url.split('//')[1].split('/')[0] in href and 
                not any(ext in href.lower() for ext in ['.jpg', '.png', '.pdf', '.zip'])):
                pages.append(href)
                if len(pages) >= limit * 2:  # Crawl more initially to filter later
                    break
    except Exception as e:
        print(f"Error crawling {url}: {e}")
    return list(set(pages))[:limit]  # Return unique URLs up to limit

def extract_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        title = soup.title.string if soup.title else ""
        
        # Get text from main content areas
        body = ""
        for tag in ['main', 'article', 'div', 'section']:
            elements = soup.find_all(tag)
            for element in elements:
                body += element.get_text(separator=' ', strip=True) + " "
        
        # Fallback to all text if no specific content found
        if len(body.strip()) < 100:
            body = soup.get_text(separator=' ', strip=True)
            
        return title + " " + body
    except Exception as e:
        print(f"Failed to extract text from {url}: {e}")
        return ""

# Advanced Preprocessing [v1.1.0]
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def preprocess(text):
    # Handle contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am", "’re": " are", "’s": " is"
    }
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    
    # Remove special characters but keep basic punctuation for sentence boundaries
    text = re.sub(r'[^\w\s.,;!?]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    
    # Tokenize with punctuation
    tokens = word_tokenize(text)
    
    # POS tagging for better lemmatization
    pos_tags = pos_tag(tokens)
    
    processed_tokens = []
    for word, tag in pos_tags:
        if len(word) <= 2 or word in stop_words or not word.isalpha():
            continue
            
        # Get WordNet POS tag
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        stem = stemmer.stem(lemma)
        processed_tokens.append(stem)
    
    return ' '.join(processed_tokens)

# Improved Indexing with BM25 [v1.1.0]
def index_documents(docs):
    # Tokenize documents for BM25
    tokenized_docs = [doc.split() for doc in docs]
    
    # Create both BM25 and TF-IDF indices
    bm25 = BM25Okapi(tokenized_docs)
    
    # Also keep TF-IDF for visualization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )
    tfidf_vectors = vectorizer.fit_transform(docs)
    
    return {
        'bm25': bm25,
        'tfidf': tfidf_vectors,
        'vectorizer': vectorizer,
        'tokenized_docs': tokenized_docs
    }

# Enhanced Query Handling with Query Expansion [v1.1.0]
def expand_query(query):
    # Simple synonym expansion
    synonyms = {
        "phone": ["mobile", "cellphone", "smartphone", "device"],
        "cheap": ["inexpensive", "affordable", "low-cost", "budget"],
        "good": ["great", "excellent", "quality", "superior"],
        "laptop": ["notebook", "computer", "macbook", "ultrabook"],
        "price": ["cost", "rate", "value", "amount"],
        "buy": ["purchase", "order", "acquire", "shop"]
    }
    
    expanded = query.lower()
    for word, syns in synonyms.items():
        if word in expanded:
            expanded += " " + " ".join(syns)
    
    return expanded

def handle_query(query, index, urls, processed_docs):
    # Expand query
    expanded_query = expand_query(query)
    query_processed = preprocess(expanded_query).split()
    
    # Get scores from BM25
    doc_scores = index['bm25'].get_scores(query_processed)
    ranked_indices = np.argsort(doc_scores)[::-1]
    
    # More lenient threshold - only filter out very poor matches
    min_score = np.percentile(doc_scores, 30)  # Keep top 70% of scores
    filtered_indices = [i for i in ranked_indices if doc_scores[i] > min_score]
    
    # Get relevant documents (for evaluation)
    relevant_indices = get_relevant_indices_for_query(query_processed, processed_docs)
    
    # Debug output
    print(f"\nQuery terms: {query_processed}")
    print(f"Number of relevant documents found: {len(relevant_indices)}")
    
    # Evaluate at different k values
    print("\nSearch Results Evaluation:")
    for k in [3, 5, 10]:
        current_k = min(k, len(filtered_indices))  # Don't exceed available results
        if current_k == 0:
            print(f"Top {k} results - No results to evaluate")
            continue
            
        precision, recall, f1 = evaluate(query, filtered_indices, relevant_indices, current_k)
        print(f"Top {current_k} results - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    
    return filtered_indices, doc_scores

def get_relevant_indices_for_query(query_tokens, processed_docs):
    relevant_indices = []
    if not query_tokens:  # Handle empty queries
        return relevant_indices
    
    query_set = set(query_tokens)
    
    for idx, doc in enumerate(processed_docs):
        doc_tokens = set(doc.split())
        intersection = query_set & doc_tokens
        
        # More lenient relevance criteria:
        # 1. At least one term matches (minimum requirement)
        # 2. Higher scores for better matches
        if len(intersection) > 0:
            relevant_indices.append(idx)
    
    return relevant_indices

# Robust Evaluation [v1.1.0]
def evaluate(query, retrieved_indices, relevant_indices, k):
    if not retrieved_indices or k == 0:
        return 0.0, 0.0, 0.0
    
    # Ensure we don't try to evaluate more results than we have
    k = min(k, len(retrieved_indices))
    
    retrieved_top_k = retrieved_indices[:k]
    relevant_set = set(relevant_indices)
    
    # If no relevant documents exist, precision is undefined (return 0)
    if not relevant_set:
        return 0.0, 0.0, 0.0
    
    # Calculate true positives, false positives, false negatives
    tp = len([i for i in retrieved_top_k if i in relevant_set])
    fp = len([i for i in retrieved_top_k if i not in relevant_set])
    fn = len([i for i in relevant_set if i not in retrieved_top_k])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

# Visualization with Cluster Labels [v1.1.0]
def visualize_vectors(doc_vectors, urls, cluster_labels=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(doc_vectors.toarray())
    
    plt.figure(figsize=(12, 10))
    
    if cluster_labels is not None:
        # Color by cluster
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, 
                            cmap='viridis', alpha=0.7, edgecolor='k', s=100)
        plt.colorbar(scatter, label='Cluster')
    else:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=np.arange(len(reduced)), 
                            cmap='viridis', alpha=0.7, edgecolor='k', s=100)
        plt.colorbar(scatter, label='Document Index')
    
    # Annotate some points with their URLs (not all to avoid clutter)
    for i in range(0, len(urls), max(1, len(urls)//10)):
        plt.annotate(f"{i}", (reduced[i, 0], reduced[i, 1]), 
                    textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.title("Document Vector Space (PCA Reduced)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
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

    print("[+] Crawling and extracting data from e-commerce sites...")
    urls = []
    for base in base_urls:
        print(f"Crawling: {base}")
        urls.extend(crawl_website(base, limit=15))  # Get 15 URLs per site
    
    print(f"\n[+] Found {len(urls)} unique URLs. Extracting content...")
    raw_docs = []
    valid_urls = []
    
    for url in urls:
        content = extract_text(url)
        if content.strip() and len(content.split()) > 50:  # Only keep documents with meaningful content
            raw_docs.append(content)
            valid_urls.append(url)
    
    print(f"[+] Keeping {len(raw_docs)} valid documents after filtering")
    
    print("[+] Preprocessing documents...")
    processed_docs = [preprocess(doc) for doc in raw_docs]
    
    print("[+] Indexing documents with BM25 and TF-IDF...")
    index = index_documents(processed_docs)
    
    print("\n[+] Search engine ready. Enter your queries below:")
    while True:
        query = input("\nSearch query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
            
        ranked_indices, scores = handle_query(query, index, valid_urls, processed_docs)
        
        print("\nTop Results:")
        for i, idx in enumerate(ranked_indices[:5]):  # Show top 5 results
            print(f"\n{i+1}. URL: {valid_urls[idx]}")
            print(f"   Score: {scores[idx]:.4f}")
            
            # Show snippet of content
            snippet = ' '.join(raw_docs[idx].split()[:30]) + "..."
            print(f"   Content: {snippet}")
    
    # Final visualization
    print("\n[+] Generating document visualization...")
    visualize_vectors(index['tfidf'], valid_urls)
    
    print("\n[+] Search session ended.")
    
    # Closing Brackets Properly