import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
)
import umap

st.set_page_config(page_title="Windows Sentinel - UMAP Workflow", page_icon="🛡️", layout="wide")

st.title("🛡️ Windows Sentinel : Détection d'Anomalies avec UMAP")
st.markdown("Workflow complet de Machine Learning appliqué aux logs de sécurité Windows.")
st.markdown("---")

# ==========================================
# 1. DATA CLEAN, OUTLIERS & AUGMENTATION/EQUILIBRAGE
# ==========================================
st.header("1. Data Cleaning, Outliers & Équilibrage")

@st.cache_data
def load_data():
    # Chargement d'un Dataset réel en ligne sur GitHub
    url = "https://raw.githubusercontent.com/whit3rabbit/Windows-Event-Codes-CSV/main/updated_detailed_events.csv"
    df_raw = pd.read_csv(url)
    return df_raw

with st.spinner("Téléchargement du Dataset en ligne..."):
    df_raw = load_data()

# Nettoyage des valeurs manquantes (Outliers brut / Lignes vides)
df_clean = df_raw[['Event ID', 'Log', 'Criticality']].dropna().copy()

# Création de la cible (1 = Critique/Anomalie, 0 = Normal)
df_clean['IsSuspect'] = df_clean['Criticality'].apply(
    lambda x: 1 if str(x).strip().lower() in ['high', 'critical'] else 0
)

# Équilibrage / Data Augmentation simple (SMOTE simulation / Oversampling)
df_normals = df_clean[df_clean['IsSuspect'] == 0]
df_attacks = df_clean[df_clean['IsSuspect'] == 1]

# Duplication controlée des attaques pour équilibrer le Dataset (Augmentation)
if len(df_attacks) > 0:
    df_attacks_augmented = df_attacks.sample(len(df_normals), replace=True, random_state=42)
    df_balanced = pd.concat([df_normals, df_attacks_augmented], ignore_index=True)
else:
    df_balanced = df_clean.copy()

st.write(f"**Taille initiale :** {len(df_clean)} lignes | **Taille après augmentation/équilibrage :** {len(df_balanced)} lignes")
st.dataframe(df_balanced.head(5))

# ==========================================
# 2. FEATURE SELECTION & CORRELATION
# ==========================================
st.header("2. Feature Selection & Analyse des Corrélation")

X = df_balanced[['Event ID', 'Log']]
y = df_balanced['IsSuspect']

# Encodage One-Hot des variables catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Event ID', 'Log'])
    ]
)
X_processed = preprocessor.fit_transform(X)

# Affichage de la matrice de corrélation
st.subheader("Matrice de Corrélation des Features")
df_corr = pd.DataFrame(X_processed).corr().abs()
fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
sns.heatmap(df_corr.iloc[:10, :10], cmap='Blues', ax=ax_corr) # Aperçu 10x10
st.pyplot(fig_corr)

# ==========================================
# 3. FEATURE EXTRACTION (UMAP)
# ==========================================
st.header("3. Feature Extraction via UMAP")

with st.spinner("Calcul de la carte UMAP en cours..."):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_processed)

# Visualisation UMAP
fig_umap, ax_umap = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1], 
    hue=y, palette={0: '#2ecc71', 1: '#e74c3c'},
    style=y, markers={0: 'o', 1: 'X'}, s=70, ax=ax_umap
)
ax_umap.set_title("Espace Réduit UMAP 2D")
ax_umap.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig_umap)

# ==========================================
# 4. MODEL SELECTION & TRAINING (UMAP SEUL)
# ==========================================
st.header("4. Model Selection & Training (UMAP Seul par Distance)")

# Entraînement : Calcul du centroïde (Centre) des données normales sur UMAP
center = np.mean(X_umap[y == 0], axis=0)

# Distance de chaque point par rapport au centre normal
distances = np.linalg.norm(X_umap - center, axis=1)

# Réglage de l'hyperparamètre (Seuil de tolérance de distance)
seuil_percentile = st.slider("Hyperparamètre : Seuil d'anomalie (Percentile Distance)", 50, 99, 85)
seuil = np.percentile(distances, seuil_percentile)

# Prédiction finale du modèle UMAP
y_pred = (distances > seuil).astype(int)

# ==========================================
# 5. METRIQUES & EVALUATION COMPLETE
# ==========================================
st.header("5. Métriques & Généralisation")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy_score(y, y_pred)*100:.1f}%")
col2.metric("Precision", f"{precision_score(y, y_pred, zero_division=0)*100:.1f}%")
col3.metric("Recall", f"{recall_score(y, y_pred, zero_division=0)*100:.1f}%")
col4.metric("F1-Score", f"{f1_score(y, y_pred, zero_division=0)*100:.1f}%")
try:
    auc = roc_auc_score(y, distances)
    col5.metric("ROC AUC", f"{auc*100:.1f}%")
except:
    col5.metric("ROC AUC", "N/A")

st.subheader("Matrice de Confusion")
cm = confusion_matrix(y, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4, 2.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred Normal', 'Pred Attaque'], 
            yticklabels=['Vrai Normal', 'Vrai Attaque'], ax=ax_cm)
st.pyplot(fig_cm)
