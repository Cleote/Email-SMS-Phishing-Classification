import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def identify_feature_types(data: pd.DataFrame, feature_columns: list, identifier: str = 'blank'):
    """
    Identify feature types based on predefined rules.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the features.
        feature_columns (list): List of feature column names.
    
    Returns:
        dict: A dictionary containing categorized feature lists.
    """
    if identifier == 'blank':
        print("identify_feature_types(data: pd.DataFrame, feature_columns: list, identifier: '[missing]')")
    
    if identifier == 'url':
        binary_features = [
            col for col in feature_columns
            # The following command was used when tilde_in_url was named nb_tilde (not sure why it's nb when it's a check)
            #if (col == "nb_tilde" or not col.startswith("nb_")) and set(data[col].unique()).issubset({0, 1, 0.0, 1.0})
            if not col.startswith("nb_") and set(data[col].unique()).issubset({0, 1, 0.0, 1.0})
        ]
        
        ratio_features = ['ratio_digits_url', 'ratio_digits_host']
        
        skewed_features = ['domain_registration_length', 'domain_age']

        numerical_features = [col for col in feature_columns if col not in binary_features + ratio_features + skewed_features]
        
    if identifier == 'body':
        binary_features = [
            col for col in feature_columns
            if not col.startswith("nb_") and set(data[col].unique()).issubset({0, 1, 0.0, 1.0})
        ]
        
        ratio_features = ['avg_word_length', 'avg_lex_word_length', 'ratio_digits', 'ratio_lex_words',
                          'ratio_caps_words', 'ratio_lex_caps', 'ratio_richness', 'ratio_lex_richness',
                          'ratio_lexical_skew', 'ratio_symbols']

        skewed_features = ['nb_letters_caps', 'nb_lex_words_caps', 'nb_unique_lex_words',
                           'nb_urls', 'nb_phone_numbers', 'nb_spec_chars', 'nb_unusual_symbols',
                           'nb_gibberish_words']
        
        # In this situation all numerical features have skewness, so this list will be empty
        numerical_features = [col for col in feature_columns if col not in binary_features + ratio_features + skewed_features]

    return {
        "binary_features": binary_features,
        "ratio_features": ratio_features,
        "skewed_features": skewed_features,
        "numerical_features": numerical_features
    }
    
def scale_features(data: pd.DataFrame, feature_types: dict):
    """
    Apply appropriate scaling and transformations based on feature types.
    
    Parameters:
        data (pd.DataFrame): The dataset containing features.
        feature_types (dict): Dictionary containing categorized feature lists.
    
    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    X_scaled = data.copy()
    scalers = {}
        
    # Min-Max Scaling for ratio features
    if feature_types["ratio_features"]:
        minmax_scaler = MinMaxScaler()
        X_scaled[feature_types["ratio_features"]] = minmax_scaler.fit_transform(X_scaled[feature_types["ratio_features"]])
        scalers["minmax"] = minmax_scaler
    
    # Log transformation for skewed features
    for col in feature_types["skewed_features"]:
        if col in X_scaled.columns:
            shift_value = 3 if col == "domain_age" else 2  # Shift negative values
            X_scaled[col] = np.log1p(X_scaled[col] + shift_value)            
    
    # Standard Scaling for numerical and skewed features
    if feature_types["numerical_features"]:
        standard_scaler = StandardScaler()
        X_scaled[feature_types["skewed_features"] + feature_types["numerical_features"]] = \
            standard_scaler.fit_transform(X_scaled[feature_types["skewed_features"] + feature_types["numerical_features"]])
        scalers["standard"] = standard_scaler
    
    return X_scaled, scalers

def apply_scalers(data: pd.DataFrame, feature_types: dict, scalers: dict):
    """
    Apply pre-extracted scalers to new evaluation/test data.

    Parameters:
        data (pd.DataFrame): The dataset to transform.
        feature_types (dict): Dictionary with categorized feature lists.
        scalers (dict): Dictionary containing pre-extracted scalers.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    X_scaled = data.copy()

    # Apply MinMax scaling
    if "minmax" in scalers:
        X_scaled[feature_types["ratio_features"]] = scalers["minmax"].transform(X_scaled[feature_types["ratio_features"]])

    # Apply log transformation
    for col in feature_types["skewed_features"]:
        if col in X_scaled.columns:
            shift_value = 3 if col == "domain_age" else 2 if col == "domain_registraion_length" else 0
            X_scaled[col] = np.log1p(X_scaled[col] + shift_value)

    # Apply Standard scaling
    if "standard" in scalers:
        X_scaled[feature_types["skewed_features"] + feature_types["numerical_features"]] = \
            scalers["standard"].transform(X_scaled[feature_types["skewed_features"] + feature_types["numerical_features"]])

    return X_scaled

def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for binary and categorial columns.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    return np.sqrt(phi2 / min((k-1), (r-1)))

def cramers_v_matrix(df, columns):
    """
    Compute the Cramér's V correlation matrix for a list of binary or categorical features.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): A list of column names to compute Cramér's V for.

    Returns:
    - pd.DataFrame: A DataFrame containing the Cramér's V values.
    """
    num_cols = len(columns)
    result_matrix = pd.DataFrame(np.zeros((num_cols, num_cols)), 
                                 index=columns, 
                                 columns=columns)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                result_matrix.iloc[i, j] = 1.0  # Ensuring Cramér’s V(X, X) = 1
            if i > j:
                result_matrix.iloc[i, j] = cramers_v(df[col1], df[col2])
            else:
                result_matrix.iloc[i, j] = result_matrix.iloc[j, i]  # Symmetric matrix

    return result_matrix