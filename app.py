import os
import nltk

# You may need to download these first:
nltk.download('words', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt_tab', quiet=True)

import torch
import pandas as pd
import gradio as gr
from txt_extraction.text_body_features import has_url
from datetime import datetime
from load_model import load_models
from temp_cleanup import cleanup_temp
from model_objects import PhishingDataset
from sklearn.preprocessing import LabelEncoder
from history_loader import update_history_files
from classifier import extractScaler, classifier

# Set a valid temporary directory
temp_dir = os.path.join(os.getcwd(), "temp")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
# Set the environment variable for Gradio's temporary directory
os.environ['GRADIO_TEMP_DIR'] = temp_dir

# Cleanup function for temporary files
def clear_temp_files():
    """Check if there are any temporary files in the temp directory."""
    temp_path = os.environ.get('GRADIO_TEMP_DIR')
    if os.listdir(temp_path):
        """Clear temporary files."""
        cleanup_temp(temp_path)
        return gr.update(value="Temporary files cleared successfully.")
    else:
        return gr.update(value="Temp folder is empty. No files to clear.")
    

# Load models
load_text_model = torch.load("models/phoBERT-base-v2-Text-16k-v0.1-hf-do0.5/phobert_text_phishing_model_best.pt",
                        map_location=torch.device("cpu"), weights_only=False)
load_url_model = torch.load("models/phoBERT-base-v2-Urls-20k-v0.7-hf-do0.5-lrd-c-v1/phobert_phishing_model_best.pt",
                       map_location=torch.device("cpu"), weights_only=False)

# Define the label mapping
label_mapping = {'legitimate': 0, 'phishing': 1}
labels = list(label_mapping.keys())

# Define unnecessary columns filter
text_drop_columns = ['body', 'status']
url_drop_columns = ['url', 'status']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder with the labels
label_encoder.fit(labels)

# Read the CSV file to extract column names
text_csv_file_path = "training_sets/text_training.csv"
text_csv_columns = pd.read_csv(text_csv_file_path, nrows=0).columns.tolist()  # Read only the column names

url_csv_file_path = "training_sets/url_training.csv"
url_csv_columns = pd.read_csv(url_csv_file_path, nrows=0).columns.tolist()  # Read only the column names

text_feature_columns = [col for col in text_csv_columns if col not in text_drop_columns]
url_feature_columns = [col for col in url_csv_columns if col not in url_drop_columns]

# Load each model's checkpoint
text_model, text_tokenizer = load_models(text_feature_columns, label_encoder, load_text_model)
url_model, url_tokenizer = load_models(url_feature_columns, label_encoder, load_url_model)
    
def create_dataset(data, features, tokenizer):
    # Create the dataset for evaluation
    dataset = PhishingDataset(
        data=data,
        features=features,
        labels=label_encoder.transform(labels),
        tokenizer=tokenizer
    )
    
    return dataset

def evaluationBuilder(type, input):
    # Check if the input is a URL or text
    if type == 'text':
        drop_columns = text_drop_columns
        tokenizer = text_tokenizer
    else:
        drop_columns = url_drop_columns
        tokenizer = url_tokenizer
        
    # Extract features from the input
    data, eval_scaled_features = extractScaler(input, type, drop_columns)
    eval_dataset = create_dataset(data, eval_scaled_features, tokenizer)
    
    return eval_dataset

def format_row_count(row_count):
    """Format the row count into a readable string."""
    if row_count > 99999:
        return "1M"
    elif row_count > 999:
        return f"{row_count // 1000}k"
    elif row_count > 99:
        return f"{row_count / 1000:.1f}k"
    else:
        return str(row_count)
    
def check_input(text, input_type):
    if not text:
        return (
            "",
            gr.update(interactive=False)
        )
    elif input_type == "Text" and len(text.strip()) < 50:
        return (
            "<span style='color:orange;'>Caution: Inputs with low character count may lead to inaccurate predictions. Take the following prediction result with scrutiny.</span>",
            gr.update(interactive=True)
        )
    elif input_type == "URL" and not has_url(text):
        return (
            "<span style='color:red;'>Warning: URL input type selected, but no URL was detected in the input. Please Enter a valid URL format: \"www[dot]example[dot]com\" </span>",
            gr.update(interactive=False)  # Disable button if no URL
        )
    else:
        return ("", gr.update(interactive=True))  # Enable button

def process_input(model_type, input_text):
    model_type = model_type.lower()
    # Route the input to the appropriate model based on the selected type
    if model_type == 'text':
        model = text_model
    else:
        model = url_model
    return classifier(evaluationBuilder(model_type, input_text), model, 'single')

def process_csv(file, model_type):
    # Read the uploaded CSV file
    df = pd.read_csv(file.name)
    
    # Check if the required column exists
    if model_type == "Text" and "body" not in df.columns:
        return "Error: The uploaded CSV must contain a 'body' column for text classification."
    elif model_type == "URL" and "url" not in df.columns:
        return "Error: The uploaded CSV must contain a 'url' column for URL classification."
    
    # Extract the relevant column
    model_type = model_type.lower()
    if model_type == "text":
        inputs = df["body"]
        model = text_model
    else:
        inputs = df["url"]
        model = url_model

    # Process each input and collect probabilities
    results = []
    for input_text in inputs:
        results.append(classifier(evaluationBuilder(model_type, input_text), model , 'batch'))
        
    # Add results to the DataFrame and return it
    df["prediction"] = [result[0] for result in results]
    df["confidence"] = [result[1] for result in results]
    
    # Save the results to a relative path with a unique name
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    # Generate a unique filename using datetime and row count
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    row_count = format_row_count(len(df))
    output_file = os.path.join(output_dir, f"predictions-{row_count}-{model_type}-{timestamp}.csv")
    
    df.to_csv(output_file, index=False)
    # Return the file path and make the file_output visible
    return gr.update(value=output_file, visible=True)

with gr.Blocks(css=
    """
    .centered {
        text-align: center;
        margin: auto;
    }
    
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 0px;
        cursor: pointer;
        border-radius: 8px;
    }
    
    .button:hover {
        background-color: #45a049;
    }
    
    .status-clear-container {
        display: block;
        max-width: 98.25%;
        margin: auto;
    }
    
    #status_text {
        padding: 15px;
    }
    
    #clear_temp_btn {
        width: 100%;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 8px;
    }
    
    #user_input {
        padding-top: 15px;
        padding-right: 15px;
        padding-bottom: 15px;
        padding-left: 15px;
    }
    
    #single_output {
        padding-top: inherit;
        padding-right: 15px;
        padding-bottom: 15px;
        padding-left: 15px;
    }
    """
) as demo:
    gr.Markdown("<div class='title'>Email/SMS Phishing Detection Demo</div>")
    gr.themes.Monochrome().set()
    
    model_selector = gr.Radio(choices=["Text", "URL"], label="Select Model Type", value="Text")
    
    with gr.Tabs():
        with gr.Tab("Single Input"):
            gr.Markdown("Enter text or URL for prediction:")
            warning = gr.Markdown("")
            user_input = gr.Textbox(label="Input Text or URL", placeholder="Type here...", elem_id="user_input", lines=4)
            single_output = gr.Textbox(label="Prediction Result", elem_id="single_output")
            predict_btn = gr.Button("Predict", elem_classes="button", interactive=False)
            
            user_input.change(fn=check_input, inputs=[user_input, model_selector], outputs=[warning, predict_btn])
            model_selector.change(fn=check_input, inputs=[user_input, model_selector], outputs=[warning, predict_btn])
            
        with gr.Tab("Batch Input"):
            gr.Markdown("Upload a CSV file for batch processing:")
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            file_output = gr.File(label="Download Results")
            process_file_btn = gr.Button("Process CSV File", elem_classes="button")
            
        with gr.Tab("History"):
            gr.Markdown("Download previously generated files:")
            history_files = gr.File(label="Generated Files", file_types=[".csv"], interactive=False)
            
            # Automatically update the history tab when the app is launched
            demo.load(fn=update_history_files, inputs=[], outputs=history_files)
    
    with gr.Row(elem_classes="status-clear-container"):
        clear_confirm = gr.Markdown("")
        clear_temp_btn = gr.Button("Clear Temporary Files", elem_id="clear_temp_btn")
    
    # Predict single input
    predict_btn.click(fn=process_input, inputs=[model_selector, user_input], outputs=single_output)
    
    # Process CSV file
    process_file_btn.click(fn=process_csv, inputs=[file_input, model_selector], outputs=file_output)

    # Clear temporary files
    clear_temp_btn.click(fn=clear_temp_files, inputs=[], outputs=clear_confirm)
    
    gr.Markdown("<div style='text-align: center; margin-top: 20px;'>Demo được tạo bởi sinh viên Trần Minh Tâm</div>")
    
demo.launch()