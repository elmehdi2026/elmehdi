import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap

st.set_page_config(page_title="Master IAENG - EL MEHDI", layout="wide")

st.sidebar.title("Navigation")
choix = st.sidebar.selectbox("Choisir le modèle :", [
    "1. Decision Tree", "2. KNN", "3. LDA", "4. PCA", 
    "5. QDA", "6. Régression Logistique", "7. SVM (One-Class)", "8. t-SNE", "9. UMAP"
])

@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    return mnist.data[:2000] / 255.0, mnist.target[:2000].astype(int)

X, y = load_data()

if choix == "1. Decision Tree":
    st.title("🌳 Decision Tree")
    X_tr, X_te, y_tr, y_te = train_test_split(X, np.where(y==0, 1, 0), test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5).fit(X_tr, y_tr)
    st.success(f"Précision : {accuracy_score(y_te, clf.predict(X_te))*100:.2f}%")

elif choix == "2. KNN":
    st.title("👥 KNN Classifier")
    X_tr, X_te, y_tr, y_te = train_test_split(X, np.where(y==0, 1, 0), test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_tr, y_tr)
    st.success(f"Précision : {accuracy_score(y_te, clf.predict(X_te))*100:.2f}%")

elif choix == "3. LDA":
    st.title("📊 LDA Projection")
    mask = (y == 0) | (y == 1)
    clf = LinearDiscriminantAnalysis().fit(X[mask], y[mask])
    st.success(f"Modèle LDA entraîné avec succès sur 0 et 1. Score : {clf.score(X[mask], y[mask])*100:.2f}%")

elif choix == "4. PCA":
    st.title("📉 PCA Compression")
    pca = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(pca[:, 0], pca[:, 1], c=y, cmap="tab10", s=10)
    st.pyplot(fig)

elif choix == "5. QDA":
    st.title("📈 QDA Classification")
    mask = (y == 0) | (y == 1)
    qda = QuadraticDiscriminantAnalysis().fit(X[mask], y[mask])
    st.success(f"Score QDA : {qda.score(X[mask], y[mask])*100:.2f}%")

elif choix == "6. Régression Logistique":
    st.title("📉 Régression Logistique")
    X_tr, X_te, y_tr, y_te = train_test_split(X, np.where(y==0, 1, 0), test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
    st.success(f"Précision : {accuracy_score(y_te, clf.predict(X_te))*100:.2f}%")

elif choix == "7. SVM (One-Class)":
    st.title("🎯 One-Class SVM (Anomalies)")
    model = OneClassSVM(nu=0.1).fit(X[y == 0])
    st.success("Modèle One-Class SVM entraîné uniquement sur le chiffre 0 !")
    pred = model.predict([X[0]])
    st.write(f"Test sur un chiffre 0 (1 = Normal, -1 = Anomalie) : {pred[0]}")

elif choix == "8. t-SNE":
    st.title("🗺️ t-SNE Embedding")
    embedding = TSNE(n_components=2, random_state=42).fit_transform(X[:1000])
    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y[:1000], cmap="tab10", s=10)
    st.pyplot(fig)

elif choix == "9. UMAP":
    st.title("🌐 UMAP Embedding")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X[:1000])
    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y[:1000], cmap="tab10", s=10)
    st.pyplot(fig)
