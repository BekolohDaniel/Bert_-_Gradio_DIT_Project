import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import torch.nn.functional as F
import os

# 1. Redefine the exact matching Model architecture
class Model(nn.Module):
    def __init__(self, model_name="google-bert/bert-base-cased", num_classes=3):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name) 
        self.hidden_dim = self.model.config.hidden_size
        self.proj_lin = nn.Linear(self.hidden_dim, num_classes)
    
    def forward(self, input_ids):
        x = self.model(input_ids) 
        x = x.last_hidden_state[:, 0]  # Extract CLS token
        x = self.proj_lin(x)
        return x

# 2. Check hardware device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load the Saved Weights and Tokenizer from the training script output
checkpoint_dir = os.path.abspath("./model_checkpoint")
model_name = "google-bert/bert-base-cased"

print(f"Loading model checkpoint from {checkpoint_dir} onto {device}...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
model = Model(model_name=model_name, num_classes=3)
model.load_state_dict(torch.load(f"{checkpoint_dir}/sentiment_model.pt", map_location=device))
model.to(device)
model.eval()

# Mapping IDs back to human-readable strings
id_to_label = {0: "negative", 1: "neutral", 2: "positive"}

# 4. Text Prediction Function for Gradio
def prediction(text_input):
    if not text_input.strip():
        return "Veuillez entrer une phrase valide."
        
    # Tokenize input string
    inputs = tokenizer(
        text_input,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(device)
    
    # Calculate inference logits
    with torch.no_grad():
        outputs = model(input_ids)
        # Apply softmax to calculate probabilities for a cleaner interface presentation
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
        
    # Build a clean dictionary with confidence levels for Gradio's Label layout
    return {id_to_label[i]: float(probabilities[i]) for i in range(3)}

# 5. Build Gradio UI Dashboard Interface
demo = gr.Interface(
    fn=prediction, 
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Entrez votre texte ici (ex: COVID-19 news or public opinion)...", 
        label="Texte d'entrée"
    ), 
    outputs=gr.Label(num_top_classes=3, label="Sentiment Prédit (Confiance)"),
    title="Analyse de Sentiment BERT - Corona NLP",
    description="Saisissez une phrase pour analyser le sentiment. L'application affiche les scores de probabilité pour les classes : negative, neutral, et positive.",
    examples=[
        ["The global stock market is crashing due to quarantine lockdown measures."],
        ["A new vaccine trial has started today with positive preliminary results."],
        ["We are working from home today following the company guidelines."]
    ]
)

if __name__ == "__main__":
    # Launch local web service
    demo.launch(share=False)
