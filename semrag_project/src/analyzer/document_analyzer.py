# src/analyzer/document_analyzer.py
from collections import Counter
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

class DocumentAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def generate_summary(self, chunks: List[str], type: str = "concise") -> str:
        """Generate document summary"""
        full_text = " ".join(chunks)
        
        if type == "concise":
            # First 1000 chars + key sentences
            sentences = nltk.sent_tokenize(full_text)
            summary = sentences[0] if sentences else ""
            if len(summary) < 200 and len(sentences) > 1:
                summary += " " + sentences[1]
            return summary[:500] + "..."
        
        elif type == "bullet":
            # Extract key points
            sentences = nltk.sent_tokenize(full_text)
            important = self._get_important_sentences(sentences, 5)
            return "\n".join([f"• {s}" for s in important])
        
        else:  # detailed
            return full_text[:2000] + "..."
    
    def extract_keywords(self, chunks: List[str], top_n: int = 20) -> List[str]:
        """Extract important keywords"""
        full_text = " ".join(chunks)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', full_text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]
    
    def get_statistics(self, chunks: List[str]) -> Dict:
        """Get document statistics"""
        full_text = " ".join(chunks)
        sentences = nltk.sent_tokenize(full_text)
        
        return {
            "total_chunks": len(chunks),
            "total_words": len(full_text.split()),
            "total_characters": len(full_text),
            "total_sentences": len(sentences),
            "avg_chunk_size": sum(len(c.split()) for c in chunks) / len(chunks),
            "unique_words": len(set(full_text.lower().split()))
        }
    
    def extract_topics(self, chunks: List[str], n_topics: int = 5) -> List[List[str]]:
        """Extract main topics using LDA"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf = vectorizer.fit_transform(chunks)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append(top_words)
        
        return topics
    
    def _get_important_sentences(self, sentences: List[str], n: int) -> List[str]:
        """Get most important sentences based on word frequency"""
        if not sentences:
            return []
        
        words = [w.lower() for s in sentences for w in nltk.word_tokenize(s) 
                if w.isalpha() and w.lower() not in self.stop_words]
        
        freq = Counter(words)
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            score = sum(freq.get(word.lower(), 0) for word in nltk.word_tokenize(sentence))
            sentence_scores[i] = score
        
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]
        return [sentences[i] for i in sorted(top_indices)]