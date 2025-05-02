# word_with_nlp.py

from nltk.util import ngrams
import re
import math
import nltk
from collections import Counter

class NLPClass:
    def __init__(self):
        # Load English dictionary words
        self.english_words = set(nltk.corpus.words.words())
        # Set up n-gram model from Brown corpus
        self.train_text = ' '.join(nltk.corpus.brown.words())
        self.character_pairs = Counter(ngrams(self.train_text.lower(), 2))
        self.total_pairs = sum(self.character_pairs.values())
        
    def calculate_entropy(self, text):
        """Calculate Shannon entropy of a string"""
        text = text.lower()
        probabilities = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in probabilities)
    
    def contains_dictionary_word(self, domain):
        """Check if domain contains recognizable dictionary words"""
        # Remove TLD (.com, .org, etc.)
        domain_name = re.sub(r'\.[a-z]{2,}$', '', domain.lower())
        
        # Check for whole domain match
        if domain_name in self.english_words:
            return True
            
        # Check for partial matches (words of 3+ letters)
        for word in self.english_words:
            if len(word) >= 3 and word in domain_name:
                return True
                
        return False
    
    def calculate_ngram_score(self, text):
        """Calculate how natural the character transitions are"""
        text = text.lower()
        pairs = list(ngrams(text, 2))
        
        if not pairs:
            return 0
            
        # Calculate probability of this sequence based on training data
        score = 0
        for pair in pairs:
            pair_count = self.character_pairs[pair]
            # Smooth probabilities to avoid zeros
            probability = (pair_count + 1) / (self.total_pairs + len(self.character_pairs))
            score += math.log(probability)
            
        return score / len(pairs)  # Normalize by length
    
    def check_word_random(self, domain):
        """Main function to determine if a domain appears random"""
        # Remove TLD for analysis
        domain_name = re.sub(r'\.[a-z]{2,}$', '', domain.lower())
        
        # Skip very short domains as they're hard to classify
        if len(domain_name) < 3:
            return 0
        
        # Calculate various features
        entropy = self.calculate_entropy(domain_name)
        has_dictionary_word = self.contains_dictionary_word(domain_name)
        ngram_score = self.calculate_ngram_score(domain_name)
        
        # Decision logic
        random_score = 0
        reasons = []
        
        # High entropy suggests randomness
        if entropy > 3.0:
            random_score += 0.3
            reasons.append(f"High character entropy ({entropy:.2f})")
        
        # Dictionary words suggest non-randomness
        if has_dictionary_word:
            random_score -= 0.4
            reasons.append("Contains recognizable word(s)")
        
        # Low n-gram score suggests randomness
        if ngram_score < -3.5:
            random_score += 0.4
            reasons.append("Unusual character combinations")
        
        # Make final determination
        is_random = random_score > 0.2
        confidence = min(0.9, abs(random_score) + 0.5)  # Scale to reasonable confidence

        # Verdict
        return 1 if (is_random) else 0
        
        '''
        return {
            "is_random": is_random,
            "confidence": confidence,
            "reason": ", ".join(reasons),
            "entropy": entropy,
            "ngram_score": ngram_score,
            "has_dictionary_word": has_dictionary_word
        }
        '''