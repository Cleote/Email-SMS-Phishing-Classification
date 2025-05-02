# text-body_features.py

# 0 stands for legitimate
# 1 stands for phishing

import re
import nltk
import emoji
import unicodedata
import phonenumbers
from collections import Counter
from urllib.request import urlopen
# from underthesea import lang_detect
from langdetect import detect, LangDetectException
from txt_extraction.word_gibberish_checker import WordGibberishDetector
# from underthesea import word_tokenize as uts_word_tokenize  --> These two are deprecated until underthesea is compatible with
# from underthesea import text_normalize as uts_text_normalize  newer versions of python.
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from phonenumbers import NumberParseException

# A function to fetch and load a confusables map from Unicode.org or a cached version for homoglyphs
def load_confusables():
    """Load Unicode confusables mapping from Unicode.org or use a cached version."""
    try:
        # Try to load from Unicode.org (you may want to cache this locally in production)
        url = "https://www.unicode.org/Public/security/latest/confusables.txt"
        with urlopen(url) as response:
            confusables_data = response.read().decode('utf-8')
        
        # Parse the confusables data
        confusables = {}
        for line in confusables_data.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(';')
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[1].split('#')[0].strip()
                    if source and target:
                        # Convert hex representations to actual characters
                        source_char = chr(int(source, 16))
                        # We don't need the target character, just need to know it's a confusable
                        confusables[source_char] = True
        
        return confusables
    
    except Exception:
        # Fallback to a small built-in list if we can't fetch the data
        return {
            'а': True, 'е': True, 'о': True, 'р': True, 'с': True,  # Cyrillic
            'α': True, 'β': True, 'ε': True, 'μ': True, 'ο': True,  # Greek
            'ı': True, 'ł': True, 'ń': True, 'ѕ': True              # Other
        }

# Preload confusables for future use
CONFUSABLES = load_confusables()

def safe_lang_detect(text):
    """Safe wrapper for lang_detect that handles newlines"""
    # Replace newlines with spaces for language detection
    single_line_text = text.replace('\n', ' ')
    try:
        return detect(single_line_text)
    except LangDetectException:
        # Fallback to English if language detection fails
        return 'en'

def get_default_region(text): # --> Used for phone region
    """Determine default region based on detected language."""
    lang = safe_lang_detect(text)
    return 'VN' if lang == 'vi' else 'US'
        
# Returns tokenized text based of detected language (vi|en)
def tokenize_text(text):
    """
    Tokenizes text into words based on detected language.
    Currently supports Vietnamese and defaults to English/other languages.
    
    Args:
        text (str): The text to tokenize
        
    Returns:
        list: A list of tokenized words
    """
    # lang = safe_lang_detect(text)
    # return uts_word_tokenize(text) if lang == 'vi' else nltk_word_tokenize(text)
    return nltk_word_tokenize(text)

# Returns normalized text based of detected language (vi|en)
def normalize_text(text):
    """
    Normalizes text into words based on detected language.
    Currently supports Vietnamese and defaults to English/other languages.
    
    Args:
        text (str): The text to normalize
        
    Returns:
        list: A list of normalized words
    """
    # lang = safe_lang_detect(text)
    # return uts_text_normalize(text) if lang == 'vi' else unicodedata.normalize("NFKC", text)
    return unicodedata.normalize("NFKC", text)

OBFUSCATED_URL_PATTERN = re.compile(
    r'(?:h[^\w]*t[^\w]*t[^\w]*p[^\w]*s?[^\w]*:[^\w]*/[^\w]*/|w[^\w]*w[^\w]*w[^\w]*\.)[^\s\.,!?)]+',
    re.IGNORECASE
)

def extract_obfuscated_urls(text):
    """Extracts and normalizes obfuscated URLs, filtering invalid ones."""
    urls = []
    for match in OBFUSCATED_URL_PATTERN.finditer(text):
        normalized = re.sub(r'\s+', '', match.group())
        if "." in normalized:  # Basic validation (has a domain)
            urls.append(normalized)
    return urls

# Initialize WordGibberishDetector
detector = WordGibberishDetector()  # Create once
english_words = set(nltk.corpus.words.words())  # Load once

phishhints_txt = open("word_collections/phishing_hints.txt", "r")
imperatives_txt = open("word_collections/imperatives.txt", "r")
allbrands_txt = open("word_collections/allbrands.txt", "r")

def __txt_to_list(txt_object):
    list = []
    for line in txt_object:
        list.append(line.strip())
    txt_object.close()
    return list
    
HINTS = __txt_to_list(phishhints_txt)
IMPERATIVES = __txt_to_list(imperatives_txt)
BRANDS = __txt_to_list(allbrands_txt)

GREETING_PATTERNS = [
    
    # A collection of common English generic greeting patterns seen in phishing
    re.compile(
        r"\b(dear|attention|hello|hi|greetings|respected)\s+"
        r"(customer|user|friend|valued customer|sir|madam|member|client|account holder|subscriber|shopper|recipient|"
        r"valued member|valued client|registered user|email user|account holder|beneficiary|valued partner|applicant|"
        r"policyholder)\b",
        re.IGNORECASE
    ),
    
    # A collection of common Vietnamese generic greeting patterns seen in phishing
    re.compile(
        r"\b(kính gửi|kính thưa|thân gửi|chào|xin chào|quý khách)\s+"
        r"(quý khách|quý khách hàng|anh/chị|bạn thân mến)\b",
        re.IGNORECASE
    )
]

# Pre-compile all regex patterns
URL_PATTERN = re.compile(
    r'(?:(?:https?|ftp)://|www\.)\S+|'                            # Standard protocols
    r'(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?|'                       # Domain patterns
    r'\b(?:h[^\w]*t[^\w]*t[^\w]*p[^\w]*s?[^\w]*:[^\w]*/[^\w]*/|'  # Obfuscated http://
    r'w[^\w]*w[^\w]*w[^\w]*\.)',                                  # Obfuscated www.
    re.IGNORECASE
)
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_PATTERN = re.compile(
    r'(?<!\w)'  # Negative lookbehind to prevent partial matches
    r'(\+\d{1,3}[-.\s]?)?'  # Country code
    r'(?:\(\d{1,4}\)|\d{1,4})'  # Area code
    r'[-.\s]?\d{1,4}[-.\s]?\d{1,4}'  # Local number
    r'(?!\w)'  # Negative lookahead to prevent partial matches
)
TOLL_FREE_PATTERN = re.compile(r'\b(\+?\d{1,3}[-.\s]?)?((800|888|877|866|855|844|833|822|880|881|882|883|884|885|886|887|889))[-.\s]?\d{3}[-.\s]?\d{4}\b', re.IGNORECASE)
_CLEAN_REGEX = re.compile(r'[^a-zA-Z]')

STANDARD_CATEGORIES = {
        'Lu', 'Ll', 'Lt', 'Lm', 'Lo',
        'Nd', 'Nl', 'No',
        'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po',
        'Zs'
    }

def clean_word(word):
    """Faster cleaning with precompiled regex"""
    return _CLEAN_REGEX.sub('', word)

def is_valid_phone_number(number_str, text=None):
    """Enhanced validation with automatic region detection."""
    try:
        region = get_default_region(text) if text else None
        parsed = phonenumbers.parse(number_str, region)
        return phonenumbers.is_valid_number(parsed)
    except NumberParseException:
        return False


#################################################################################################################################
#               1. Check if text-body has a generic gretting
#################################################################################################################################

def has_generic_greeting(text):
    """Returns True if the text contains a generic phishing-style greeting, otherwise False."""
    for pattern in GREETING_PATTERNS:
        if pattern.search(text):
            return 1
    return 0

#################################################################################################################################
#               2. Check if text-body has a url
#################################################################################################################################

def has_url(text):
    """Detects both standard and obfuscated URLs."""
    return 1 if URL_PATTERN.search(text) else 0
    
#################################################################################################################################
#               3. Check if text-body has an email
#################################################################################################################################

def has_email(text):
    """Returns True if the text contains an email address, otherwise False."""
    return 1 if EMAIL_PATTERN.search(text) else 0

#################################################################################################################################
#               4. Check if text-body has a phone number
#################################################################################################################################

def has_phone_number(text):
    """Fast check with language-aware region defaults."""
    if not PHONE_PATTERN.search(text):
        return 0
        
    # Only validate one example if fast check passes
    for match in PHONE_PATTERN.finditer(text):
        if is_valid_phone_number(match.group(), text):
            return 1
    return 0

# def has_phone_number(text):
#     """Returns True if the text contains a phone number, otherwise False."""
#     # Matches various phone number formats including (xxx) xxx-xxxx, xxx-xxx-xxxx, etc.
#     return 1 if PHONE_PATTERN.search(text) else 0

#################################################################################################################################
#               5. Check if text-body has a toll free phone number
#################################################################################################################################

def has_toll_free_number(text):
    """Returns True if the text contains a toll-free phone number, otherwise False."""
    # Expanded list of toll-free prefixes (US/Canada + some international)
    return 1 if TOLL_FREE_PATTERN.search(text) else 0

# def has_toll_free_number(text):
#     """Returns True if the text contains a toll-free phone number, otherwise False."""
#     # Expanded list of toll-free prefixes (US/Canada + some international)
#     toll_free_prefixes = r'800|888|877|866|855|844|833|822|880|881|882|883|884|885|886|887|889'
    
#     # Regex pattern matches:
#     # 1. Optional international prefix (+1, 001, etc.)
#     # 2. One of the toll-free prefixes
#     # 3. Separators (optional): spaces, hyphens, dots, or none
#     # 4. Remaining 7 digits (with optional separators)
#     pattern = (
#         r'\b(\+?\d{1,3}[-.\s]?)?'  # Optional international prefix
#         f'({toll_free_prefixes})'    # Toll-free prefix
#         r'[-.\s]?\d{3}[-.\s]?\d{4}\b'  # Remaining digits
#     )
    
#     return 1 if(re.search(pattern, text, re.IGNORECASE)) else 0
    
#################################################################################################################################
#               6. Check if text-body contains an unofuscated brand
#################################################################################################################################

def has_brand(text):
    """
    Returns 1 if any brand appears in the text, ignoring case.
    Allows common separators (e.g., '-', '_', '/', '.') within brand names, and punctuation after.
    """
    # Allowed internal separators between brand words
    separator = r'[-_/\.]?'
    
    # Generate flexible patterns for multi-word brands: 'american express' → matches 'american[-_/\.]?express'
    brand_patterns = [
        separator.join(map(re.escape, brand.split()))
        for brand in BRANDS
    ]
    
    pattern = re.compile(
        r'(?<!\w)(' + '|'.join(brand_patterns) + r')\W*(?!\w)',
        flags=re.IGNORECASE
    )
    
    # Just need to know if there's at least one match
    return 1 if pattern.search(text) else 0

#################################################################################################################################
#               7. Check if text-body contains gibberish words
#################################################################################################################################

def has_gibberish(text):
    """Early exit on first gibberish word."""
    tokens = tokenize_text(text)
    for word in tokens:
        cleaned = clean_word(word)
        if cleaned and detector.detect_gibberish([cleaned]):
            return 1
    return 0

#################################################################################################################################
#               8. Check if text-body has signs of obfuscation techniques
#################################################################################################################################

def has_obfuscation(text, symbol_threshold=0.4, unique_ngram_threshold=0.7, n=5):
    """Returns 1 if text has signs of homoglyphs, typosquatting, or leetspeak, otherwise 0."""
    # Normalize text based on language
    normalized_text = normalize_text(text)
    if not normalized_text.strip():
        return 0 # Skip empty or whitespace-only input

    # Character frequency analysis
    char_counts = Counter(normalized_text)
    total_chars = sum(char_counts.values())
    if total_chars == 0:
        return 0

    # Symbol ratio check (detects excessive symbols like '@', '$', '0' for 'o')
    symbol_chars = [c for c in normalized_text if not c.isalnum()]
    symbol_ratio = len(symbol_chars) / total_chars

    # Detect mixed scripts (Latin + non-Latin in the same word)
    has_mixed_scripts = bool(re.search(r'[a-zA-Z][\u0400-\u04FF]|[\u0400-\u04FF][a-zA-Z]', normalized_text))

    # N-gram uniqueness check
    ngrams = [normalized_text[i:i+n] for i in range(len(normalized_text) - n + 1)]
    if not ngrams:
        unique_ngram_ratio = 0
    else:
        unique_ngram_ratio = len(set(ngrams)) / len(ngrams)

    # Final Heuristic Criteria
    is_symbol_heavy = symbol_ratio > symbol_threshold
    is_too_unique = unique_ngram_ratio > unique_ngram_threshold

    # Combined criteria (require at least two flags)
    flags = sum([is_symbol_heavy, is_too_unique, has_mixed_scripts])
    return 1 if flags >= 2 else 0

#################################################################################################################################
#               9. Count the number of characters in the text-body
#################################################################################################################################

def count_chars(text):
    """Returns the number of characters in the given text, including homoglyphs and special symbols."""
    # return len([c for c in text if unicodedata.category(c)[0] != 'C'])
    return sum(unicodedata.category(c)[0] != 'C' for c in text)

#################################################################################################################################
#               10 + 11. Count the number of words / lexical words in the text-body
#################################################################################################################################

def count_words(text, lex=False):
    """Returns the number of words or lexical words if lex=True (case-insensitive)."""
    # Remove non-letter characters, convert to lowercase, and filter empty strings
    words = (clean_word(word).lower() for word in tokenize_text(text))
    
    if lex:
        return sum(1 for word in words if word and word in english_words)
    
    else:
        return sum(1 for word in words if word)

#################################################################################################################################
#               12. Count the number of capitalized letters in the text-body
#################################################################################################################################

def count_capitalized_letters(text):
    """Returns the number of capitalized letters"""
    return sum(char.isupper() for char in text)
    
    # """Returns the number of capitalized letters: A-Z (ASCII 65-90), ignoring other characters."""
    # return sum(65 <= ord(c) <= 90 for c in text)

#################################################################################################################################
#               13 + 14. Count the number of all caps words in the text-body
#################################################################################################################################

def count_all_caps_words(text, lex=False):
    """Returns the number of words that are entirely in uppercase (minimum 2 characters)."""
    words = (clean_word(word) for word in tokenize_text(text))
    
    if lex:
        # Count lexical words that are all uppercase and at least 2 characters long
        return sum(1 for word in words if word.isupper() and len(word) >= 2 and (w := word.lower()) in english_words)
    
    else:
        # Count words that are all uppercase and at least 2 characters long
        return sum(1 for word in words if word.isupper() and len(word) >= 2)
    
#################################################################################################################################
#               15. Count the number of numerical characters (0-9) in the text-body
#################################################################################################################################

def count_numerical_characters(text):
    """Returns the number of numerical characters (0-9) in the text."""
    return sum(1 for char in text if char.isdigit())

#################################################################################################################################
#               16 + 17. Count the number of unique words in the text-body
#################################################################################################################################

def count_unique_words(text, lex=False):
    """Returns the number of unique words in the text, case-insensitive."""
    # Remove non-letter characters, convert to lowercase, and filter empty strings
    words = (clean_word(word).lower() for word in tokenize_text(text))

    if lex:
        # Create a set of unique non-empty, lexical words
        unique_words = {word for word in words if word in english_words} 
        
    else:
        # Create a set of unique non-empty words
        unique_words = {word for word in words if word}
        
    return len(unique_words)

#################################################################################################################################
#               18. Count the number of URLs in the text-body
#################################################################################################################################

def count_urls(text):
    """Returns the number of unique standard + obfuscated URLs."""
    return len(set(URL_PATTERN.findall(text)) | set(extract_obfuscated_urls(text)))

#################################################################################################################################
#               19. Count the number of phone numbers in the text-body
#################################################################################################################################

def count_phone_number(text):
    """Returns the number of valid phone numbers using regex + library validation."""
    matches = PHONE_PATTERN.finditer(text)
    return sum(1 for match in matches if is_valid_phone_number(match.group(), text))

# def hybrid_count(text):
#     """Returns the number of phone numbers in the text."""
#     # Step 1: Fast regex pre-filter
#     candidates = re.findall(
#         r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b',
#         text
#     )
    
#     # Step 2: Validate only regex matches
#     count = 0
#     for num in candidates:
#         try:
#             parsed = phonenumbers.parse(num, "US")
#             if is_valid_number(parsed):
#                 count += 1
#         except:
#             pass
#     return count

#################################################################################################################################
#               20. Count the number of symbols/special characters in the text-body
#################################################################################################################################

def count_special_characters(text):
    """Returns the number of symbols/special characters in the text."""
    # Count characters that are not alphanumeric and not whitespace
    return sum(1 for char in text if not (char.isalnum() or char.isspace()))

#################################################################################################################################
#               21. Count the number of unsual symbols/characters in the text-body (emojis, homoglyphs)
#################################################################################################################################

# def count_unusual_symbols(text):
#     """Returns the count of unusual symbols including emojis and homoglyphs."""
#     # Count emojis
#     emoji_count = sum(1 for char in text if char in emoji.EMOJI_DATA)
    
#     # Get confusables mapping for homoglyph detection
#     confusables = load_confusables()
    
#     # Count homoglyphs
#     homoglyph_count = sum(1 for char in text if char in confusables)
    
#     # Count other unusual Unicode characters
#     unusual_unicode_count = 0
#     standard_categories = {'Lu', 'Ll', 'Lt', 'Lm', 'Lo',
#                            'Nd', 'Nl', 'No',
#                            'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po',
#                            'Zs'}
    
#     for char in text:
#         # Skip if it's already counted as emoji or homoglyph
#         if char in emoji.EMOJI_DATA or char in confusables:
#             continue
        
#         # Check Unicode category
#         category = unicodedata.category(char)
#         if category not in standard_categories:
#             unusual_unicode_count += 1
            
#     # Return detailed counts
#     return {
#         "total_unusual": emoji_count + homoglyph_count + unusual_unicode_count,
#         "emoji_count": emoji_count,
#         "homoglyph_count": homoglyph_count,
#         "other_unusual_count": unusual_unicode_count
#     }

def count_unusual_symbols(text):
    """Returns the number of unusual symbols including emojis and homoglyphs/unusual unicode."""
    total = 0
    EMOJI_DATA = emoji.EMOJI_DATA

    for char in text:
        if char in EMOJI_DATA or char in CONFUSABLES:
            total += 1
        else:
            category = unicodedata.category(char)
            if category not in STANDARD_CATEGORIES:
                total += 1

    return total

#################################################################################################################################
#               22. Count the number of gibberish in the text-body
#################################################################################################################################

def count_gibberish(text):
    """Batch processing optimized."""
    tokens = tokenize_text(text)
    words = [clean_word(word) for word in tokens]
    words = [word for word in words if word]
    
    return sum(detector.detect_gibberish(words))

# def count_gibberish(text):
#     """Counts the number of gibberish words in the text."""
#     # Tokenize the text
#     tokens = tokenize_text(text)
#     words = [clean_word(word) for word in tokens]
#     words = [word for word in words if word]
    
#     # Count gibberish words
#     count = 0
#     for word in words:
#         if detector.detect_gibberish([word]):
#             count += 1
    
#     return count

#################################################################################################################################
#               23. Count the number of phish-hints in the text-body
#################################################################################################################################

def count_phish_hints(text):
    """
    Counts the number of phishing hints in the text, ignoring case.
    Allows common separators (e.g., '-', '_', '/', '.') within hints, and punctuation after.
    """
    # Allowed internal separators between hint words
    separator = r'[-_/\.]?'

    # Generate flexible patterns: 'log in' → matches 'log[-_/\.]?in'
    hint_patterns = [
        separator.join(map(re.escape, hint.split()))
        for hint in HINTS
    ]

    pattern = re.compile(
        r'(?<!\w)(' + '|'.join(hint_patterns) + r')\W*(?!\w)',
        flags=re.IGNORECASE
    )

    matched_ranges = []
    for match in pattern.finditer(text):
        start, end = match.span()
        # Avoid overlaps
        if all(end <= s or start >= e for s, e in matched_ranges):
            matched_ranges.append((start, end))
    
    return len(matched_ranges)

#################################################################################################################################
#               24. Count the number of imperatives in the text-body
#################################################################################################################################

def count_imperatives(text):
    """
    Counts the number of imperatives in the text, ignoring case.
    Allows common separators (e.g., '-', '_', '/', '.') within hints, and punctuation after.
    """
    # Allowed internal separators between hint words
    separator = r'[-_/\.]?'

    # Generate flexible patterns: 'log in' → matches 'log[-_/\.]?in'
    imperative_patterns = [
        separator.join(map(re.escape, imperative.split()))
        for imperative in IMPERATIVES
    ]

    pattern = re.compile(
        r'(?<!\w)(' + '|'.join(imperative_patterns) + r')\W*(?!\w)',
        flags=re.IGNORECASE
    )

    matched_ranges = []
    for match in pattern.finditer(text):
        start, end = match.span()
        # Avoid overlaps
        if all(end <= s or start >= e for s, e in matched_ranges):
            matched_ranges.append((start, end))

    return len(matched_ranges)

#################################################################################################################################
#               25 + 26. Calculate the average length of a word in the text-body
#################################################################################################################################

def avg_word_length(char_count, word_count):
    """Calculate the average length of a word in the text."""
    return (char_count / word_count) if word_count and char_count else 0

#################################################################################################################################
#               25 + 26. Calculate the ratio of numerics compared to characters in the text-body
#################################################################################################################################

def ratio_digits(numerics_count, char_count):
    """Calculate the ratio of numbers compared to characters in the text."""
    return (numerics_count / char_count) if char_count and numerics_count else 0

#################################################################################################################################
#               27. Calculate the ratio of gibberish compared to words in the text-body
#################################################################################################################################

def ratio_gibberish_words(gibberish_count, word_count):
    """Calculate the ratio of gibberish words compared to words in the text."""
    return (gibberish_count / word_count) if word_count and gibberish_count else 0

#################################################################################################################################
#               28. Calculate the ratio of gibberish compared to words in the text-body
#################################################################################################################################

def ratio_lex_words(lex_word_count, word_count):
    """Calculate the ratio of gibberish words compared to words in the text."""
    return (lex_word_count / word_count) if word_count and lex_word_count else 0

#################################################################################################################################
#               29. Calculate the ratio of phish-hints compared to words in the text-body
#################################################################################################################################

def ratio_phish_words(phish_count, word_count):
    """Calculate the ratio of phish-hint words compared to words in the text."""
    return (phish_count / word_count) if word_count and phish_count else 0

#################################################################################################################################
#               30. Calculate the ratio of imperatives compared to words in the text-body
#################################################################################################################################

def ratio_imperative_words(imperative_count, word_count):
    """Calculate the ratio of imperative words compared to words in the text."""
    return (imperative_count / word_count) if word_count and imperative_count else 0

#################################################################################################################################
#               31. Check if the ratio of gibberish is over a certain threshold
#################################################################################################################################

def check_gibberish_frequency(ratio_gibberish, threshold=0.02):
    """
    Checks if the ratio of gibberish exceeds the threshold.
    Returns True if the ratio is higher than the threshold.
    
    Default threshold is 20% (0.2) of total words.
    """
    return 1 if (ratio_gibberish > threshold) else 0

#################################################################################################################################
#               32. Check if the ratio of phish-hints is over a certain threshold
#################################################################################################################################

def check_phish_frequency(ratio_phish, threshold=0.02):
    """
    Checks if the ratio of phishing hints exceeds the threshold.
    Returns True if the ratio is higher than the threshold.
    
    Default threshold is 2% (0.02) of total words.
    """
    return 1 if (ratio_phish > threshold) else 0

#################################################################################################################################
#               33. Check if the frequency of imperatives is over a certain threshold
#################################################################################################################################

def check_imperative_frequency(ratio_imperative, threshold=0.03):
    """
    Checks if the frequency of imperatives exceeds the threshold.
    Returns True if the frequency is higher than the threshold.
    
    Default threshold is 3% (0.03) of total words.
    """
    return 1 if (ratio_imperative > threshold) else 0

#################################################################################################################################
#               34 + 35. Calculate the ratio of nb_words_caps compared to nb_words in the text-body
#################################################################################################################################

def ratio_caps_words(caps_count, word_count):
    """Calculate the ratio of capitalized words compared to words in the text."""
    return (caps_count / word_count) if word_count and caps_count else 0

#################################################################################################################################
#               36 + 37. Calculate the ratio of nb_unique_words compared to nb_words in the text-body
#################################################################################################################################

def ratio_richness(unique_word_count, word_count):
    """Calculate the ratio of unique words compared to words in the text."""
    return (unique_word_count / word_count) if word_count and unique_word_count else 0

#################################################################################################################################
#               38. Calculate the ratio of nb_symbols + nb_unsual_symbols compared to nb_characters in the text-body
#################################################################################################################################

def ratio_symbols(special_characters_count, unusual_symbols_count, char_count):
    """Calculate the ratio of imperatives compared to words in the text."""
    symbols_count = special_characters_count + unusual_symbols_count
    return (symbols_count / char_count) if char_count and symbols_count else 0
    