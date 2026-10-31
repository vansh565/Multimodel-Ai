from collections import defaultdict
from typing import List, Dict
import numpy as np
from datetime import datetime, timedelta

class FeedbackLearner:
    """Learn from user feedback to improve responses"""
    
    def __init__(self, config):
        self.config = config
        self.feedback_history = defaultdict(list)
        self.term_weights = defaultdict(float)
        self.pattern_scores = defaultdict(float)
        
    def add_feedback(self, session_id: str, query: str, response: str, rating: float, helpful: bool):
        """Store and process feedback"""
        feedback_entry = {
            'query': query,
            'response': response,
            'rating': rating,
            'helpful': helpful,
            'timestamp': datetime.now()
        }
        
        self.feedback_history[session_id].append(feedback_entry)
        
        # Update term weights based on feedback
        self._update_term_weights(query, response, rating)
        
        # Update pattern scores
        self._update_pattern_scores(query, response, rating)
        
        # Prune old feedback
        self._prune_old_feedback(session_id)
    
    def _update_term_weights(self, query: str, response: str, rating: float):
        """Update weights for important terms"""
        terms = query.lower().split()
        
        for term in terms:
            if len(term) > 3:  # Ignore short terms
                current_weight = self.term_weights[term]
                # Adjust based on rating (positive or negative)
                adjustment = (rating - 0.5) * 0.1
                self.term_weights[term] = max(0, min(1, current_weight + adjustment))
    
    def _update_pattern_scores(self, query: str, response: str, rating: float):
        """Update scores for response patterns"""
        # Simple pattern extraction
        if 'symptom' in query.lower() and rating > 0.7:
            self.pattern_scores['symptom_response'] += 0.1
        elif 'treatment' in query.lower() and rating > 0.7:
            self.pattern_scores['treatment_response'] += 0.1
        elif rating < 0.3:
            if 'symptom' in query.lower():
                self.pattern_scores['symptom_response'] = max(0, self.pattern_scores['symptom_response'] - 0.1)
    
    def get_important_terms(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get most important terms based on feedback"""
        sorted_terms = sorted(self.term_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:top_k]
    
    def get_pattern_boost(self, query: str) -> float:
        """Get boost factor based on query patterns"""
        boost = 1.0
        
        if 'symptom' in query.lower():
            boost *= (1 + self.pattern_scores['symptom_response'])
        if 'treatment' in query.lower():
            boost *= (1 + self.pattern_scores['treatment_response'])
        
        return min(boost, 2.0)  # Cap at 2x boost
    
    def get_session_performance(self, session_id: str) -> Dict:
        """Get performance metrics for a session"""
        feedbacks = self.feedback_history[session_id]
        
        if not feedbacks:
            return {'avg_rating': 0, 'helpful_rate': 0, 'count': 0}
        
        ratings = [f['rating'] for f in feedbacks]
        helpful_count = sum(1 for f in feedbacks if f['helpful'])
        
        return {
            'avg_rating': np.mean(ratings),
            'helpful_rate': helpful_count / len(feedbacks),
            'count': len(feedbacks),
            'recent_trend': self._calculate_trend(feedbacks[-5:])
        }
    
    def _calculate_trend(self, recent_feedbacks: List) -> float:
        """Calculate performance trend"""
        if len(recent_feedbacks) < 2:
            return 0
        
        ratings = [f['rating'] for f in recent_feedbacks]
        return ratings[-1] - ratings[0]
    
    def _prune_old_feedback(self, session_id: str, max_age_days: int = 30):
        """Remove feedback older than max_age_days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        self.feedback_history[session_id] = [
            f for f in self.feedback_history[session_id]
            if f['timestamp'] > cutoff
        ]