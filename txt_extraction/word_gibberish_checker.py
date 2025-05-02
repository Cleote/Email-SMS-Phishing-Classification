# word_gibberish_checker.py

import math
import nltk
from nltk.util import ngrams
from collections import Counter

class WordGibberishDetector:
    def __init__(self):
        # Load English dictionary words
        self.english_words = set(nltk.corpus.words.words())

        # Add common proper names, more if needed
        self.english_words.update({'cory', 'odonnell', 'smith', 'johnson', 'williams', 'jones',
                                   'brown', 'davis', 'miller', 'wilson', 'moore', 'taylor',
                                   'anderson', 'thomas', 'jackson', 'white'})

        sample_text = ' '.join(nltk.corpus.brown.words()[:50000])  # Limit corpus size
        self.char_pairs = Counter(ngrams(sample_text.lower(), 2))
        self.total_char_pairs = sum(self.char_pairs.values())

    def is_dictionary_word(self, word):
        """Check if word exists in dictionary"""
        return word.lower() in self.english_words
    
    def calculate_entropy(self, word):
        """Calculate Shannon entropy of a word's characters"""
        word = word.lower()
        if len(word) < 2:
            return 0
        probabilities = [word.count(c) / len(word) for c in set(word)]
        return -sum(p * math.log2(p) for p in probabilities)
    
    def calculate_char_ngram_score(self, word):
        """Calculate character transition probability within a word"""
        if len(word) < 2:
            return 0
            
        pairs = list(ngrams(word.lower(), 2))
        score = 0
        for pair in pairs:
            pair_count = self.char_pairs[pair]
            probability = (pair_count + 1) / (self.total_char_pairs + len(self.char_pairs))
            score += math.log(probability)
        return score / len(pairs)

    def detect_gibberish(self, tokens):
        """Optimized version that returns a list of 1/0 for gibberish detection."""
        results = []
        
        for word in tokens:
            if not any(c.isalpha() for c in word):
                continue
                
            # Fast checks in priority order
            if self.is_dictionary_word(word):
                results.append(0)
                continue
                
            entropy = self.calculate_entropy(word)
            char_score = self.calculate_char_ngram_score(word)
            
            if entropy > 3.0 or char_score < -5.0:
                results.append(1)
            else:
                results.append(0)
        
        return results # 1 for gibberish, 0 for not gibberish