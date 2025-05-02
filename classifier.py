import torch
import numpy as np
import tqdm as tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from url_extract import extract_url_features
from utils import identify_feature_types, scale_features
from txt_extract import extract_text_body_features as extract_text_features

def extractScaler(input:str, type:str, drop_columns:list):
    # Extract features from the input based on the type
    # Check if the input is a URL or text
    if type == 'text':
        target = 'body'
        extracted_features = extract_text_features(input)
    else:
        target = 'url'
        extracted_features = extract_url_features(input)
        
    # Convert the extracted features to a DataFrame-like structure
    # Assuming extracted_features is a DataFrame-like object (e.g., pandas DataFrame)
    data = extracted_features[target].values
    feature_columns = [col for col in extracted_features.columns if col not in drop_columns]
    eval_features = extracted_features[feature_columns]
    
    # Scale features
    feature_types = identify_feature_types(eval_features, feature_columns, target)
    scaled_features = scale_features(eval_features, feature_types)[0]
    eval_scaled_features = scaled_features.to_numpy(dtype=np.float32)
    
    return data, eval_scaled_features

def classifier(eval_dataset, model, output_type: str):
    
    # Create testing dataset loader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
    )
    
    # Make prediction
    with torch.no_grad():
        progress_bar = tqdm.tqdm(eval_dataloader, desc="Evaluating the inputs")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(torch.device("cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cpu"))
            features = batch['features'].to(torch.device("cpu"))
            
            # Get raw model outputs (logits)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        
            # Apply softmax to convert logits to probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get the predicted class and confidence score
            confidence, predicted = torch.max(probabilities, dim=1)
            
            # Convert confidence to percentage
            confidence_percentage = confidence.item() * 100
            
        # Format the output for single prediction
        if output_type == 'single':
            return f"Phishing (Confidence: {confidence_percentage:.2f}%)" if predicted.item() == 1 else f"Not Phishing (Confidence: {confidence_percentage:.2f}%)"
        
        # Format the output for batch prediction
        else:
            for pred in predicted:
                if pred.item() == 1:
                    predictions = "phishing"
                else:
                    predictions = "legitimate"
            return predictions, round(confidence_percentage, 2)