import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download resources (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def to_lowercase(text: str) -> str:
    """Convert text to lowercase"""
    return text.lower()

def remove_special_characters(text: str) -> str:
    """Remove special characters, keep only letters and numbers"""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_stopwords(tokens: list) -> list:
    """Remove stopwords from tokenized words"""
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize words"""
    return [lemmatizer.lemmatize(word) for word in tokens]

def stem_tokens(tokens: list) -> list:
    """Stem words"""
    return [stemmer.stem(word) for word in tokens]

def normalize(text: str, use_stemming=False) -> str:
    """
    Normalize text by:
    1. Lowercasing
    2. Removing special characters
    3. Tokenizing (simple split)
    4. Removing stopwords
    5. Lemmatization (or stemming if use_stemming=True)
    """
    text = to_lowercase(text)
    text = remove_special_characters(text)
    tokens = text.split()  # simple tokenizer
    tokens = remove_stopwords(tokens)
    
    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)
    
    return " ".join(tokens)
