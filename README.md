# 🦠 Classification des Sentiments sur les Tweets COVID-19 avec BERT

Ce projet consiste à entraîner un modèle **BERT** à l'aide de **PyTorch** afin de classifier automatiquement des textes selon leur sentiment. Les tweets du jeu de données **Corona NLP** sont classés en trois catégories :

- 🔴 Négatif
- ⚪ Neutre
- 🟢 Positif

Le projet comprend également une interface web interactive développée avec **Gradio**, permettant de tester le modèle en temps réel.

---

# 📌 Présentation du projet

L'objectif de ce projet est de mettre en œuvre un pipeline complet de traitement automatique du langage naturel (NLP) comprenant :

- Le chargement et le prétraitement du jeu de données Corona NLP
- La normalisation des étiquettes de sentiment
- L'entraînement d'un modèle BERT pré-entraîné
- La sauvegarde du modèle entraîné
- Le déploiement d'une interface de prédiction avec Gradio
- Le suivi de l'entraînement grâce à Weights & Biases (W&B)

---

# 🚀 Fonctionnalités

- Entraînement d'un modèle BERT avec PyTorch
- Tokenisation automatique avec Hugging Face Transformers
- Classification des sentiments en trois classes
- Interface utilisateur interactive avec Gradio
- Suivi des performances avec Weights & Biases (W&B)
- Architecture de projet claire et modulaire

---

# 📂 Structure du projet

```text
devoir_nlp/
│
├── Corona_NLP_train.csv      # Jeu de données d'entraînement
├── demo.py                   # Interface Gradio
├── model.py                  # Script d'entraînement
├── requirements.txt          # Dépendances Python
├── README.md
├── .gitignore
│
└── model_checkpoint/         # À télécharger depuis Google Drive
    ├── sentiment_model.pt
    ├── config.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ...
```

---

# ⚙️ Installation

## 1. Cloner le dépôt GitHub

```bash
https://github.com/BekolohDaniel/Bert_-_Gradio_DIT_Project.git
```

## 2. Créer un environnement virtuel

### Sous Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Sous Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 4. Se connecter à Weights & Biases (facultatif)

```bash
wandb login
```

---

# 📥 Téléchargement du modèle entraîné

Le dossier **model_checkpoint/** n'est pas inclus dans ce dépôt GitHub car sa taille dépasse la limite autorisée par GitHub.

Vous pouvez télécharger le modèle entraîné à partir du lien Google Drive suivant :

## 📁 Google Drive

**https://drive.google.com/drive/folders/1JZx_icqXcwSE_lSyyeUGj_CUyKmmb74o?usp=drive_link**

Après le téléchargement, placez le dossier **model_checkpoint** à la racine du projet.

La structure du projet doit être la suivante :

```text
devoir_nlp/
│
├── demo.py
├── model.py
├── model_checkpoint/
│   ├── sentiment_model.pt
│   ├── config.json
│   ├── tokenizer_config.json
│   └── ...
```

---

# 📊 Jeu de données

Le projet utilise le fichier :

```
Corona_NLP_train.csv
```

Le jeu de données d'origine contient cinq catégories de sentiments :

- Extremely Negative
- Negative
- Neutral
- Positive
- Extremely Positive

Ces catégories ont été regroupées en trois classes :

| Catégorie originale | Nouvelle catégorie |
|----------------------|--------------------|
| Extremely Negative | Négatif |
| Negative | Négatif |
| Neutral | Neutre |
| Positive | Positif |
| Extremely Positive | Positif |

---

# 🏋️ Entraînement du modèle

Placez le fichier **Corona_NLP_train.csv** dans le dossier principal du projet puis lancez :

```bash
python model.py
```

Le script :

- charge les données ;
- tokenize les textes ;
- entraîne le modèle BERT ;
- évalue les performances ;
- sauvegarde automatiquement le modèle dans :

```text
model_checkpoint/
```

---

# 🌐 Lancement de l'application Gradio

Après avoir téléchargé ou entraîné le modèle, lancez :

```bash
python demo.py
```

Une interface web Gradio sera disponible à l'adresse suivante :

```
http://127.0.0.1:7860
```

Saisissez une phrase et le modèle prédira automatiquement son sentiment.

### Exemple

**Entrée**

```
J'ai beaucoup aimé ce produit.
```

**Sortie**

```
Positif 😊
```

---

# 📈 Suivi de l'entraînement

Les performances du modèle sont enregistrées automatiquement avec **Weights & Biases (W&B)**.

Les informations suivies comprennent notamment :

- la perte (Training Loss) ;
- la précision (Accuracy) ;
- les performances à chaque époque (Epoch) ;
- les artefacts du modèle.

---

# 📦 Dépendances principales

Le projet utilise principalement les bibliothèques suivantes :

- Python
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Scikit-learn
- Gradio
- Weights & Biases

Toutes les dépendances peuvent être installées avec :

```bash
pip install -r requirements.txt
```

---

# 🚫 Fichiers exclus du dépôt GitHub

Les fichiers suivants ne sont pas inclus dans le dépôt GitHub :

- model_checkpoint/
- .venv/
- __pycache__/
- *.pyc

---

# ⚠️ Difficultés rencontrées

Au cours de ce projet, plusieurs difficultés ont été rencontrées :

- le temps d'entraînement relativement long du modèle BERT ;
- la gestion de la mémoire lors de l'entraînement ;
- le regroupement des cinq classes de sentiments en trois catégories ;
- la sauvegarde et le chargement des fichiers volumineux du modèle ;
- la limite de taille imposée par GitHub pour les fichiers supérieurs à 100 Mo.

---

# 👨‍💻 Auteur

**Daniel**

Étudiant en Master

---

# 📄 Remarque

Ce projet a été réalisé dans le cadre du devoir de **Classification des sentiments avec BERT**. Il présente une solution complète allant de l'entraînement du modèle jusqu'au déploiement d'une interface utilisateur interactive avec Gradio.
