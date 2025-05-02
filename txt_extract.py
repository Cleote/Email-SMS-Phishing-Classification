import pandas as pd
import txt_extraction.text_body_features as tbdfe

def extract_text_body_features(text):

    word_count = 0
    
    features = {}
    
    features['body'] = text
    
    # [text-based features]
    features['has_url'] = tbdfe.has_url(text)
    features['has_email'] = tbdfe.has_email(text)
    features['has_phone_number'] = tbdfe.has_phone_number(text)
    features['has_brand'] = tbdfe.has_brand(text)
    features['has_obfuscation'] = tbdfe.has_obfuscation(text)
    char_count = tbdfe.count_chars(text)
    word_count = tbdfe.count_words(text, False)
    lex_word_count = tbdfe.count_words(text, True)
    features['nb_letters_caps'] = tbdfe.count_capitalized_letters(text)
    caps_word_count = tbdfe.count_all_caps_words(text, False)
    features['nb_lex_words_caps'] = lex_caps_word_count = tbdfe.count_all_caps_words(text, True)
    numerics_count = tbdfe.count_numerical_characters(text)
    unique_word_count = tbdfe.count_unique_words(text, False)
    features['nb_unique_lex_words'] = unique_lex_word_count = tbdfe.count_unique_words(text, True)
    features['nb_urls'] = tbdfe.count_urls(text)
    features['nb_phone_numbers'] = tbdfe.count_phone_number(text)
    features['nb_spec_chars'] = special_characters_count = tbdfe.count_special_characters(text)
    features['nb_unusual_symbols'] = unusual_symbols_count = tbdfe.count_unusual_symbols(text)
    features['nb_gibberish_words'] = gibberish_word_count = tbdfe.count_gibberish(text)

    # [ratio-based features]
    features['avg_word_length'] = tbdfe.avg_word_length(char_count, word_count)
    features['avg_lex_word_length'] = tbdfe.avg_word_length(char_count, lex_word_count)
    features['ratio_digits'] = tbdfe.ratio_digits(numerics_count, char_count)
    features['ratio_lex_words'] = tbdfe.ratio_lex_words(lex_word_count, word_count)
    features['ratio_caps_words'] = tbdfe.ratio_caps_words(caps_word_count, word_count)
    features['ratio_lex_caps'] = tbdfe.ratio_caps_words(lex_caps_word_count, lex_word_count)
    features['ratio_richness'] = ratio_richness = tbdfe.ratio_richness(unique_word_count, word_count)
    features['ratio_lex_richness'] = ratio_lex_richness = tbdfe.ratio_richness(unique_lex_word_count, lex_word_count)
    features['ratio_lexical_skew'] = abs(ratio_richness - ratio_lex_richness)
    features['ratio_symbols'] = tbdfe.ratio_symbols(special_characters_count, unusual_symbols_count, char_count)
    
    features['status'] = None
    
    # Convert the features dictionary to a Pandas DataFrame
    features_df = pd.DataFrame([features])
    return features_df