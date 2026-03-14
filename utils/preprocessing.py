"""
utils/preprocessing.py
-----------------------
NLP text preprocessing pipeline.
Uses NLTK when available, falls back to a lightweight built-in implementation.
"""

import re
import string

# ── Try to load NLTK (optional) ───────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    for res in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ── Built-in fallback stopwords ───────────────────────────────────────────────
BUILTIN_STOPWORDS = {
    'a','an','the','is','it','in','on','at','to','for','of','and','or','but',
    'this','that','these','those','my','me','i','you','your','we','our','they',
    'their','be','been','was','were','are','am','have','has','had','do','does',
    'did','will','would','could','should','may','might','shall','with','from',
    'by','as','up','about','into','through','during','before','after','above',
    'below','between','out','off','over','under','again','then','there','here',
    'when','where','which','who','whom','how','all','both','each','few','more',
    'most','other','some','such','no','not','only','own','same','so','than',
    'too','very','just','because','if','while','although','though'
}

# Keep these even if they appear in stopword lists (intent-relevant)
KEEP_WORDS = {'where', 'need', 'want', 'help', 'near', 'can', 'get', 'find', 'go'}

# ── Simple lemmatizer rules ───────────────────────────────────────────────────
LEMMA_MAP = {
    'studying':'study','studies':'study','studied':'study',
    'eating':'eat','eats':'eat','ate':'eat',
    'going':'go','goes':'go','went':'go',
    'finding':'find','finds':'find','found':'find',
    'needing':'need','needs':'need','needed':'need',
    'wanting':'want','wants':'want','wanted':'want',
    'getting':'get','gets':'get','got':'get',
    'helping':'help','helps':'help','helped':'help',
    'looking':'look','looks':'look','looked':'look',
    'rooms':'room','labs':'lab','libraries':'library',
    'offices':'office','facilities':'facility',
    'buildings':'building','places':'place','areas':'area',
    'students':'student','doctors':'doctor','nurses':'nurse',
    'hospitals':'hospital','clinics':'clinic','hostels':'hostel',
    'medicines':'medicine','medications':'medication',
}

def _simple_lemma(word):
    return LEMMA_MAP.get(word, word)

# ── Main preprocessing function ───────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove punctuation / special characters
    3. Tokenize
    4. Remove stopwords (keeping intent-relevant words)
    5. Lemmatize
    Returns a clean string of tokens joined by spaces.
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)

    if NLTK_AVAILABLE:
        # NLTK pipeline
        tokens = word_tokenize(text)
        stops  = _stop_words - KEEP_WORDS
        tokens = [t for t in tokens if t not in stops]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    else:
        # Built-in fallback pipeline
        tokens = text.split()
        stops  = BUILTIN_STOPWORDS - KEEP_WORDS
        tokens = [t for t in tokens if t not in stops]
        tokens = [_simple_lemma(t) for t in tokens]

    return ' '.join(tokens)


def batch_preprocess(texts: list) -> list:
    """Preprocess a list of text strings."""
    return [preprocess_text(t) for t in texts]
