import torch
import warnings
from transformers import logging
from transformers import AutoTokenizer, AutoModel
from model_objects import HybridPhoBERTClassifier

logging.set_verbosity_error()

def load_models(feature_columns, label_encoder, checkpoint):
    # Suppress warnings about uninitialized weights
    warnings.filterwarnings("ignore", message="Some weights of the model were not initialized from the model checkpoint and are newly initialized")
    
    # Load PhoBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model initialized!\nUsing device: {device}")

    model = HybridPhoBERTClassifier(phobert, num_features=len(feature_columns), num_classes=len(label_encoder.classes_))
    model = model.to(device)

    # Load model for evaluation
    # Example: model_path = "phoBERT-base-v2-Text-16k-v0.1-hf-do0.5/phobert_text_phishing_model_best.pt"
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer