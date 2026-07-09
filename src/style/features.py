import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.style.state import CharacterStyleState

def compute_similarity_scores(quote_text: str, candidates: list[str], style_state: CharacterStyleState, min_quotes: int = 5) -> dict[str, float]:
    quote_text = str(quote_text).strip()
    scores = {c: 0.0 for c in candidates}
    if not quote_text:
        return scores
        
    valid_cands = []
    histories = []
    
    for c in candidates:
        if c in style_state.state.fingerprints:
            fp = style_state.state.fingerprints[c]
            if fp.quotes_seen >= min_quotes and fp.texts:
                valid_cands.append(c)
                histories.append(" ".join(fp.texts))
                
    if not valid_cands:
        return scores
        
    corpus = [quote_text] + histories
    try:
        # Default tokenizer, lowercase=True, no stop word removal, no lemmatization, no n-grams
        vectorizer = TfidfVectorizer(lowercase=True)
        vecs = vectorizer.fit_transform(corpus)
        # Cosine similarity between quote (vecs[0]) and candidate histories (vecs[1:])
        sims = cosine_similarity(vecs[0:1], vecs[1:]).flatten()
        for i, c in enumerate(valid_cands):
            scores[c] = float(sims[i])
    except ValueError:
        pass
        
    return scores
