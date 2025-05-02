import re
import tldextract
import pandas as pd
import urllib.parse
import url_extraction.url_features as urlfe
import url_extraction.external_features as extfe

def get_domain(url):
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc, tldextract.extract(url).domain, parsed_url.path

def extract_url_features(url):
    
    def words_raw_extraction(domain, subdomain, path):
        w_domain = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
        w_subdomain = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())   
        w_path = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", path.lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        raw_words = list(filter(None,raw_words))
        return raw_words, list(filter(None,w_host)), list(filter(None,w_path))

    hostname, domain, path = get_domain(url)
    extracted_domain = tldextract.extract(url)
    domain = extracted_domain.domain+'.'+extracted_domain.suffix
    subdomain = extracted_domain.subdomain
    tmp = url[url.find(extracted_domain.suffix):len(url)]
    pth = tmp.partition("/")
    path = pth[1] + pth[2]
    words_raw, words_raw_host, words_raw_path= words_raw_extraction(extracted_domain.domain, subdomain, pth[2])
    tld = extracted_domain.suffix
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme

    features = {}
    
    features['url'] = url
    
    # [url-based features]
    features['length_url'] = urlfe.url_length(url)
    features['length_hostname'] = urlfe.url_length(hostname)
    features['ip'] = urlfe.having_ip_address(url)
    features['nb_dots'] = urlfe.count_dots(url)
    features['nb_hyphens'] = urlfe.count_hyphens(url)
    features['nb_at'] = urlfe.count_at(url)
    features['nb_qm'] = urlfe.count_qm(url)
    features['nb_and'] = urlfe.count_and(url)
    features['nb_eq'] = urlfe.count_equal(url)
    features['nb_underscore'] = urlfe.count_underscore(url)
    features['tilde_in_url'] = urlfe.check_tilde(url)
    features['nb_percent'] = urlfe.count_percent(url)
    features['nb_slash'] = urlfe.count_slash(url)
    features['nb_star'] = urlfe.count_star(url)
    features['nb_colon'] = urlfe.count_colon(url)
    features['nb_comma'] = urlfe.count_comma(url)
    features['nb_semicolon'] = urlfe.count_semicolon(url)
    features['nb_dollar'] = urlfe.count_dollar(url)
    features['nb_space'] = urlfe.count_space(url)
    
    features['nb_www'] = urlfe.check_www(words_raw)
    features['nb_com'] = urlfe.check_com(words_raw)
    features['nb_dslash'] = urlfe.count_double_slash(url)
    features['http_in_path'] = urlfe.count_http_token(path)
    features['https_token'] = urlfe.https_token(scheme)
            
    features['ratio_digits_url'] = urlfe.ratio_digits(url)
    features['ratio_digits_host'] = urlfe.ratio_digits(hostname)
    features['punycode'] = urlfe.punycode(url)
    features['port'] = urlfe.port(url)
    features['tld_in_path'] = urlfe.tld_in_path(tld, path)
    features['tld_in_subdomain'] = urlfe.tld_in_subdomain(tld, subdomain)
    features['abnormal_subdomain'] = urlfe.abnormal_subdomain(url)
    features['nb_subdomains'] = urlfe.count_subdomain(url)
    features['prefix_suffix'] = urlfe.prefix_suffix(url)
    features['random_domain'] = urlfe.random_domain(domain)
    features['shortening_service'] = urlfe.shortening_service(url)
            
    features['length_words_raw'] = urlfe.length_word_raw(words_raw)
    features['char_repeat'] = urlfe.char_repeat(words_raw)
    features['shortest_words_raw'] = urlfe.shortest_word_length(words_raw)
    features['shortest_word_host'] = urlfe.shortest_word_length(words_raw_host)
    features['shortest_word_path'] = urlfe.shortest_word_length(words_raw_path)
    features['longest_words_raw'] = urlfe.longest_word_length(words_raw)
    features['longest_word_host'] = urlfe.longest_word_length(words_raw_host)
    features['longest_word_path'] = urlfe.longest_word_length(words_raw_path)
    features['avg_words_raw'] = urlfe.average_word_length(words_raw)
    features['avg_word_host'] = urlfe.average_word_length(words_raw_host)
    features['avg_word_path'] = urlfe.average_word_length(words_raw_path)
               
    features['phish_hints'] = urlfe.phish_hints(url)
    features['brand_in_domain'] = urlfe.brand_in_domain(extracted_domain.domain)
    features['brand_in_subdomain'] = urlfe.brand_in_path(extracted_domain.domain,subdomain)
    features['brand_in_path'] = urlfe.brand_in_path(extracted_domain.domain,path)
    features['suspicious_tld'] = urlfe.suspicious_tld(tld)
               
    # [third-party-based features]
    features['whois_registered_domain'] = extfe.whois_registered_domain(domain)
    features['domain_registration_length'] = extfe.domain_registration_length(domain)
    features['domain_age'] = extfe.domain_age(domain)
    
    features['status'] = None

    # Convert the features dictionary to a Pandas DataFrame
    features_df = pd.DataFrame([features])
    return features_df