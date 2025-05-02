import torch
from torch.utils.data import Dataset

# Create a custom dataset that includes both URL text and additional features
class PhishingDataset(Dataset):
    def __init__(self, data, features, labels, tokenizer, max_length=128):
        self.data = data
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = str(self.data[idx])
        features = self.features[idx]
        label = self.labels[idx]  # Already numeric from LabelEncoder
        
        # PhoBERT uses RoBERTa architecture which doesn't use token_type_ids
        encoding = self.tokenizer(
            data,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'features': torch.tensor(features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create a hybrid classification model with PhoBERT and additional features
class HybridPhoBERTClassifier(torch.nn.Module):
    def __init__(self, phobert, num_features=54, num_classes=2):
        super(HybridPhoBERTClassifier, self).__init__()
        self.phobert = phobert
        
        # Freeze PhoBERT layers (optional - comment out for full fine-tuning)
        # for param in self.phobert.parameters():
        #     param.requires_grad = False
            
        self.dropout = torch.nn.Dropout(0.1)
        
        # Feature processing network
        self.feature_network = torch.nn.Sequential(
            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # Combined network
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(phobert.config.hidden_size + 64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, features):
        # Get PhoBERT embeddings
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        bert_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        
        # Process additional features
        feature_output = self.feature_network(features)
        
        # Concatenate BERT output and feature output
        combined = torch.cat((bert_output, feature_output), dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        
        return logits