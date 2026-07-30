# 🦠 Corona Sentiment Classification using BERT

An end-to-end Natural Language Processing (NLP) pipeline that fine-tunes a **BERT (Bidirectional Encoder Representations from Transformers)** model on the Corona NLP dataset. It logs metrics via **Weights & Biases (W&B)** and serves a live production inference UI using **Gradio**.

---

## 🚀 Features
* **Custom PyTorch Pipeline**: Flexible training pipeline using `DataLoader` and custom tokenization mappings.
* **Sentiment Mapping**: Automatically normalizes 5 fine-grained dataset labels down to 3 macro classes (`negative`, `neutral`, `positive`).
* **Experiment Tracking**: Dynamic logging of training loss, verification accuracy, and model artifact tracking on the W&B dashboard.
* **Interactive UI**: Real-time deployment web interface using Gradio.

---

## 🛠️ Installation & Setup

1. **Clone or locate the workspace directory:**
   ```bash
   cd devoir_nlp
   ```

2. **Activate your Python Virtual Environment:**
   * **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   * **Mac/Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate Weights & Biases:**
   ```bash
   wandb login
   ```

---

## 📦 How to Run

Make sure your dataset is named `Corona_NLP_train.csv` and placed directly inside this folder before starting.

### Step 1: Train the Model
Run the core script to initialize your W&B dashboard session, build the dataset embeddings, and start training epochs:
```bash
python model.py
```
* **Output:** Once finished, this generates a local directory called `./model_checkpoint/` containing your fine-tuned model parameters and tokenization assets.

### Step 2: Launch the Gradio Interface
Once the checkpoint folder is completely populated, start up your lightweight browser inference engine:
```bash
python demo.py
```
* **Output:** Open your web browser and go to the local URL printed in your terminal (typically `http://127.0.0.1:7860`) to play with the dashboard app!

---

## 📂 Project Architecture
```text
devoir_nlp/
│
├── Corona_NLP_train.csv    # Source CSV data (Excluded in git)
├── model_checkpoint/       # Production weights output (Excluded in git)
│   ├── sentiment_model.pt
│   └── tokenizer configs...
│
├── model.py                # PyTorch dataset & training runtime loop
├── demo.py                 # Gradio visual inferencing environment
├── requirements.txt        # System library targets
└── .gitignore              # Tracking rules allocation
```
