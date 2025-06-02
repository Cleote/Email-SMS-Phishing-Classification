# url_features.py

# 0 stands for legitimate
# 1 stands for phishing

import re

#LOCALHOST_PATH = "/var/www/html/"
HINTS = ['wp', 'login', 'includes', 'admin', 'content', 'site', 'images', 'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins', 'signin', 'view']

allbrand_txt = open("word_collections/allbrands.txt", "r")

def __txt_to_list(txt_object):
    list = []
    for line in txt_object:
        list.append(line.strip())
    txt_object.close()
    return list
    
allbrand = __txt_to_list(allbrand_txt)

#################################################################################################################################
#               Return URL or URL hostname length
#################################################################################################################################

def url_length(url):
    """Returns the length of the given URL"""
    return len(url)
    
#################################################################################################################################
#               Check if URL is an IP Address
#################################################################################################################################

def having_ip_address(url):
    """Returns 1 if the URL contains an IP address (IPv4, IPv4 in hexadecimal, or IPv6),
       otherwise returns 0"""
    # Check for IP related patterns
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '[0-9a-fA-F]{7}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

#################################################################################################################################
#              Count the number of dots ('.') in the URL
#################################################################################################################################

def count_dots(url):
    """Returns the number of dots in the given URL"""
    return url.count('.')

#################################################################################################################################
#               Count the number of hyphens ('-') in the URL
#################################################################################################################################

def count_hyphens(url):
    """Returns the number of hyphens in the given URL"""
    return url.count('-')

#################################################################################################################################
#               Count the number of at ('@') symbols in the URL
#################################################################################################################################

def count_at(url):
    """Returns the number of at(?) symbols in the given URL"""
    return url.count('@')

#################################################################################################################################
#               Count the number of question marks ('?') in the URL
#################################################################################################################################

def count_qm(url):
    """Returns the number of question marks in the given URL"""
    return url.count('?')

#################################################################################################################################
#               Count the number of and ('&') symbols in the URL
#################################################################################################################################

def count_and(url):
    """Returns the number of and symbols in the given URL"""
    return url.count('&')

#################################################################################################################################
#               Count the number of equal ('=') symbols in the URL
#################################################################################################################################

def count_equal(url):
    """Returns the number of equal symbols in the given URL"""
    return url.count('=')

#################################################################################################################################
#               Count the number of underscores ('_') in the URL
#################################################################################################################################

def count_underscore(url):
    """Returns the number of underscores in the given URL"""
    return url.count('_')

#################################################################################################################################
#              Checks if the tilde ('_') symbol exists in the URL
#################################################################################################################################

def check_tilde(url):
    """Returns 1 if '~' is in the URL, otherwise 0"""
    return 1 if '~' in url else 0

#################################################################################################################################
#               Count the number of percent ('%') symbols in the URL
#################################################################################################################################

def count_percent(url):
    """Returns the number of percent symbols in the given URL"""
    return url.count('%')

#################################################################################################################################
#               Count the number of slashes ('/') in the URL
#################################################################################################################################

def count_slash(url):
    """Returns the number of slashes in the given URL"""
    return url.count('/')

#################################################################################################################################
#               Count the number of star ('*') symbols in the URL
#################################################################################################################################

def count_star(url):
    """Returns the number of star symbols in the given URL"""
    return url.count('*')

#################################################################################################################################
#              Count the number of colons (':') in the URL
#################################################################################################################################

def count_colon(url):
    """Returns the number of colons in the given URL"""
    return url.count(':')


#################################################################################################################################
#               Count the number of commas (',') in the URL
#################################################################################################################################

def count_comma(url):
    """Returns the number of commas in the given URL"""
    return url.count(',')

#################################################################################################################################
#               Count the number of semicolons (';') in the URL
#################################################################################################################################

def count_semicolon(url):
    """Returns the number of semicolons in the given URL"""
    return url.count(';')

#################################################################################################################################
#               Count the number of dollar ('$') symbols in the URL
#################################################################################################################################

def count_dollar(url):
    """Returns the number of dollar symbols in the given URL"""
    return url.count('$')

#################################################################################################################################
#               Count the number of (space, %20) in the URL
#################################################################################################################################

def count_space(url):
    """Returns the number of spaces in a URL, counting both ' ' and '%20'"""
    return url.count(' ')+url.count('%20')

#################################################################################################################################
#               Count the number of 'www' existing in all raw words of the URL
#################################################################################################################################

def check_www(words_raw):
    """Returns the count of words containing 'www'"""
    return sum(1 for word in words_raw if 'www' in word)
    
#################################################################################################################################
#               Count the number of 'com' existing in all raw words of the URL
#################################################################################################################################

def check_com(words_raw):
    """Returns the count of words containing 'com'"""
    return sum(1 for word in words_raw if 'com' in word)

#################################################################################################################################
#               Count the number of redirection (//) symbols in the url
#################################################################################################################################

def count_double_slash(url):
    """
    Counts occurrences of '//' in a URL and determines its significance.

    - Finds all positions of '//' in the given URL.
    - If no occurrences are found, returns 0.
    - If the last occurrence appears after the 6th character, returns 1.
    - Otherwise, returns the total count of '//' occurrences.

    Parameters:
        full_url (str): The input URL.

    Returns:
        int: A count based on the position and number of '//' occurrences.
    """
    # Find all occurrences of '//'
    matches = [x.start(0) for x in re.finditer('//', url)]
    
    # If there are no matches, return 0
    if not matches:
        return 0
    
    # If the last match is after position 6, return 1
    if matches[-1] > 6:
        return 1
    
    # Otherwise, return the total count
    return len(matches)

#################################################################################################################################
#               Count the number of 'http' existing in the path of the URL
#################################################################################################################################

def count_http_token(url_path):
    """Returns the number of times 'http' appears in the URL path"""
    return url_path.count('http')

#################################################################################################################################
#               Check if the scheme of the URL uses https protocol
#################################################################################################################################

def https_token(scheme):
    """Returns 0 if the scheme is 'https', otherwise returns 1"""
    if scheme == 'https':
        return 0
    return 1

#################################################################################################################################
#               Calculate the ratio of digits in the URL or URL hostname 
#################################################################################################################################

def ratio_digits(hostname):
    """Returns the ratio of digits relative to the total length of the URL or hostname"""
    return len(re.sub("[^0-9]", "", hostname))/len(hostname) if (hostname) else 0

#################################################################################################################################
#               Check if punycode is used in the URL
#################################################################################################################################

def punycode(url):
    """Returns 1 if the URL uses Punycode encoding, otherwise returns 0"""
    return 1 if url.startswith(("http://xn--", "https://xn--")) else 0

#################################################################################################################################
#               Check if a port pattern exists in the URL
#################################################################################################################################

def port(url):
    """Returns 1 if the URL contains an explicit port number, otherwise returns 0"""
    if re.search("^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)",url):
        return 1
    return 0

#################################################################################################################################
#               Check if TLD exists in the URL path 
#################################################################################################################################

def tld_in_path(tld, path):
    """Returns 1 if the TLD appears in the path, otherwise returns 0"""
    return 1 if tld.lower() in path.lower() else 0
    
#################################################################################################################################
#               Check if tld is used in the URL subdomain 
#################################################################################################################################

def tld_in_subdomain(tld, subdomain):
    """Returns 1 if the TLD appears in the subdomain, otherwise returns 0"""
    return 1 if tld.lower() in subdomain.lower() else 0

#################################################################################################################################
#               Check if subdomain is abnormal like: starting with wwww-, wwNN, etc.
#################################################################################################################################

def abnormal_subdomain(url):
    """Returns 1 if the URL contains an abnormal subdomain pattern, otherwise returns 0"""
    return 1 if re.search(r'(http[s]?://(w[w]?|\d))([w]?(\d|-))',url) else 0

#################################################################################################################################
#               Count the number of subdomains in the URL
#################################################################################################################################

def count_subdomain(url):
    # This function currently doesn't return the true amount of subdomains in a URL
    # Functionally this only roughly estimates the amount based on dots
    # This function will be changed in the future to correct this, after the dataset is re-extracted
    """Returns 1 if the URL contains one dot, 2 if it contains two dots, and 3 for three or more dots"""
    if len(re.findall(r"\.", url)) == 1:
        return 1
    elif len(re.findall(r"\.", url)) == 2:
        return 2
    else:
        return 3

# Corrected version - unimplemented until given time to re-extract all 20k rows:
'''
import tldextract

def count_subdomain(url):
    """Returns the number of subdomains in the given URL."""
    extracted = tldextract.extract(url)
    return extracted.subdomain.count(".") + 1 if extracted.subdomain else 0
'''

#################################################################################################################################
#               Check if there is a prefix suffix in the URL
#################################################################################################################################

def prefix_suffix(url):
    """Returns 1 if the URL contains a hyphen in the domain name, otherwise returns 0"""
    return 1 if re.search(r"https?://[^\-]+-[^\-]+/", url) else 0

#################################################################################################################################
#               Check if the registered domain is created with random characters
#################################################################################################################################

from url_extraction.word_with_nlp import NLPClass

def random_domain(domain):
    """Uses NLPClass to check if the given domain is created with random characters"""
    nlp_manager = NLPClass()
    return nlp_manager.check_word_random(domain)
    
#################################################################################################################################
#               Check if the URL is using a common shortening service
#################################################################################################################################

def shortening_service(url):
    """Returns 1 if the URL belongs to a known URL shortening service, otherwise returns 0"""
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      r'tr\.im|link\.zip\.net',
                      url)
    return 1 if match else 0

#################################################################################################################################
#               Return the number of words in the given list of raw words
#################################################################################################################################

def length_word_raw(words_raw):
    """Returns the number of words in the given list of raw words"""
    return len(words_raw)

#################################################################################################################################
#               Return the number of consecutive character sequences repeated in the list of raw words
#################################################################################################################################

def char_repeat(words_raw):
    """
    Counts the occurrences of repeated consecutive characters in words.

    The function checks for character sequences of lengths 2, 3, 4, and 5 
    within each word in the given list. It increments the count when all 
    characters in a sequence are the same.

    Parameters:
        words_raw (list of str): A list of words to analyze.

    Returns:
        int: The total count of repeated character sequences found.
    """
    def __all_same(items):
        return all(x == items[0] for x in items)

    repeat = {'2': 0, '3': 0, '4': 0, '5': 0}
    part = [2, 3, 4, 5]

    for word in words_raw:
        for char_repeat_count in part:
            for i in range(len(word) - char_repeat_count + 1):
                sub_word = word[i:i + char_repeat_count]
                if __all_same(sub_word):
                    repeat[str(char_repeat_count)] = repeat[str(char_repeat_count)] + 1
    return sum(list(repeat.values()))

#################################################################################################################################
#               Return the shortest word length in the raw words list
#################################################################################################################################

def shortest_word_length(words_raw):
    """Returns the shortest word length for the raw words list or 0 if the list is empty"""
    return min(len(word) for word in words_raw) if words_raw else 0

#################################################################################################################################
#               Return the longest word length in the raw words list
#################################################################################################################################

def longest_word_length(words_raw):
    """Returns the longest word length for the raw words list or 0 if the list is empty"""
    return max(len(word) for word in words_raw) if words_raw else 0

#################################################################################################################################
#               Calculate the average word length in the raw words list
#################################################################################################################################

def average_word_length(words_raw):
    """Returns the average word length for the raw words list or 0 if the list is empty"""
    return sum(len(word) for word in words_raw) / len(words_raw) if words_raw else 0

#################################################################################################################################
#               Count the number of phish-hints in the URL 
#################################################################################################################################

def phish_hints(url_path):
    """Counts occurrences of known phishing-related hints in the URL path"""
    return sum(url_path.lower().count(hint) for hint in HINTS)

#################################################################################################################################
#               Check if the URL's domain exists in the brands list
#################################################################################################################################

def brand_in_domain(domain):
    """Returns 1 if the domain matches a known brand, otherwise 0"""
    return 1 if domain in allbrand else 0

# Dev comment: this originally had a levenstein fuzzy matching so that it was a bit more accurate,
# sadly the dataset never made use of it:
''' 
import Levenshtein
def domain_in_brand1(domain):
    for d in allbrand:
        if len(Levenshtein.editops(domain.lower(), d.lower()))<2:
            return 1
    return 0
'''


#################################################################################################################################
#               Check if the URL's subdomain or path exists in the brands list but not in domain
#################################################################################################################################

def brand_in_path(domain,path):
    """Returns 1 if a known brand appears in the path but not in the domain, otherwise returns 0"""
    return 1 if any('.' + b + '.' in path and b not in domain for b in allbrand) else 0

#################################################################################################################################
#               Check if the TLD of the URL is in the Suspicious TLDs list
#################################################################################################################################

suspicious_tlds = ['fit','tk', 'gp', 'ga', 'work', 'ml', 'date', 'wang', 'men', 'icu', 'online', 'click', # Spamhaus
        'country', 'stream', 'download', 'xin', 'racing', 'jetzt',
        'ren', 'mom', 'party', 'review', 'trade', 'accountants', 
        'science', 'work', 'ninja', 'xyz', 'faith', 'zip', 'cricket', 'win',
        'accountant', 'realtor', 'top', 'christmas', 'gdn', # Shady Top-Level Domains
        'link', # Blue Coat Systems
        'asia', 'club', 'la', 'ae', 'exposed', 'pe', 'go.id', 'rs', 'k12.pa.us', 'or.kr',
        'ce.ke', 'audio', 'gob.pe', 'gov.az', 'website', 'bj', 'mx', 'media', 'sa.gov.au' # statistics
        ]

def suspicious_tld(tld):
    """Returns 1 if the TLD is in the list of suspicious TLDs, otherwise returns 0."""
    return 1 if tld in suspicious_tlds else 0
