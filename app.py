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

st.set_page_config(page_title="Rapports de TP - Master IAENG", page_icon="🎓", layout="wide")

st.sidebar.title("🧭 Navigation des TPs")
choix_tp = st.sidebar.selectbox(
    "Choisir un TP à afficher :",
    [
        "1. TP-Decision Tree",
        "2. TP-KNN",
        "3. TP-LDA",
        "4. TP-PCA",
        "5. TP-QDA",
        "6. TP-Régression Logistique",
        "7. TP-SVM (One-Class)",
        "8. TP-t-SNE",
        "9. TP-UMAP"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("🎓 **EL MEHDI - Master IAENG**\n\nPlateforme mutualisée de détection et d'analyse sur MNIST.")

# =========================================================================
# 1. TP-DECISION TREE
# =========================================================================
if choix_tp == "1. TP-Decision Tree":
    st.header("EL MEHDI - IAENG")
    st.title("🌳 TP-Decision Tree : Détection du Chiffre 0")
    st.write("Classification binaire avec le Dataset MNIST (0 vs Autres chiffres)")

    @st.cache_data
    def load_data_tree():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data[:10000] / 255.0, mnist.target[:10000]
        return X, y

    X, y = load_data_tree()
    y_binary = np.where(y == '0', 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_tree(X_t, y_t):
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_t, y_t)
        return model

    clf = train_tree(X_train, y_train)

    st.write("### 📊 Performance du modèle")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Précision (Accuracy) globale sur l'ensemble de test : {acc * 100:.2f}%")

    st.write("### 🔍 Tester le modèle de manière interactive")
    idx = st.number_input("Choisissez un index d'image de test (0 à 1999) :", min_value=0, max_value=1999, value=0, key="tree_idx")
    image_to_test = X_test[idx]
    true_label = y_test[idx]
    label_text = "C'est un 0" if true_label == 1 else "Autre chiffre"

    st.image(image_to_test.reshape(28, 28), caption=f"Classe réelle : {label_text}", width=150)
    prediction = clf.predict([image_to_test])[0]

    st.write("---")
    if prediction == 1:
        st.write("🔮 **Prédiction du modèle :** 🟢 **C'est un 0 !**")
    else:
        st.write("🔮 **Prédiction du modèle :** 🔴 **Anomalie (Ce n'est pas un 0)**")

# =========================================================================
# 2. TP-KNN
# =========================================================================
elif choix_tp == "2. TP-KNN":
    st.header("EL MEHDI - IAENG")
    st.title("👥 TP-KNN : Détection du Chiffre 0")
    st.write("Classification binaire avec le Dataset MNIST (0 vs Autres chiffres)")

    @st.cache_data
    def load_data_knn():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data[:5000] / 255.0, mnist.target[:5000]
        return X, y

    X, y = load_data_knn()
    y_binary = np.where(y == '0', 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    X_train, y_train = X_train[:2000], y_train[:2000]

    @st.cache_resource
    def train_knn(X_t, y_t):
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_t, y_t)
        return model

    clf = train_knn(X_train, y_train)

    st.write("### 📊 Performance du modèle")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Précision (Accuracy) globale sur l'ensemble de test : {acc * 100:.2f}%")

    st.write("### 🔍 Tester le modèle de manière interactive")
    max_idx = len(X_test) - 1
    idx = st.number_input(f"Choisissez un index d'image de test (0 à {max_idx}) :", min_value=0, max_value=max_idx, value=0, key="knn_idx")
    image_to_test = X_test[idx]
    true_label = y_test[idx]
    label_text = "C'est un 0" if true_label == 1 else "Autre chiffre"

    st.image(image_to_test.reshape(28, 28), caption=f"Classe réelle : {label_text}", width=150)
    prediction = clf.predict([image_to_test])[0]

    st.write("---")
    if prediction == 1:
        st.write("🔮 **Prédiction du modèle :** 🟢 **C'est un 0 !**")
    else:
        st.write("🔮 **Prédiction du modèle :** 🔴 **Anomalie (Ce n'est pas un 0)**")

# =========================================================================
# 3. TP-LDA
# =========================================================================
elif choix_tp == "3. TP-LDA":
    st.header("EL MEHDI - IAENG")
    st.title("TP LDA : Projection 1D (Version MNIST 28x28)")
    st.write("Chargement du vrai dataset MNIST et calcul de la projection 1D From Scratch.")

    @st.cache_data
    def load_mnist_784():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        return X, y

    X, y = load_mnist_784()
    masque = (y == 0) | (y == 1)
    X_filtre, y_filtre = X[masque] / 255.0, y[masque]

    X_0, X_1 = X_filtre[y_filtre == 0], X_filtre[y_filtre == 1]
    P_0, P_1 = len(X_0) / len(X_filtre), len(X_1) / len(X_filtre)

    mu_0, mu_1 = np.mean(X_0, axis=0), np.mean(X_1, axis=0)
    cov_0, cov_1 = np.cov(X_0, rowvar=False), np.cov(X_1, rowvar=False)
    Sigma = (cov_0 + cov_1) / 2 + 1e-4 * np.eye(784)

    w = np.linalg.solve(Sigma, mu_1 - mu_0)
    w_0 = 0.5 * np.dot(w.T, (mu_1 + mu_0)) - np.log(P_1 / P_0)
    st.success("Calculs LDA terminés avec succès !")

    projections = np.dot(X_filtre, w)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(projections[y_filtre == 0], bins=40, alpha=0.6, color='blue', label='Chiffre 0 (28x28)')
    ax.hist(projections[y_filtre == 1], bins=40, alpha=0.6, color='red', label='Chiffre 1 (28x28)')
    ax.axvline(x=w_0, color='black', linestyle='--', linewidth=2.5, label=f'Frontière (w_0 = {w_0:.2f})')
    ax.set_title("Projection LDA 1D From Scratch (MNIST 28x28)")
    ax.set_xlabel("Valeur de la projection 1D")
    ax.set_ylabel("Nombre d'images")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

# =========================================================================
# 4. TP-PCA
# =========================================================================
elif choix_tp == "4. TP-PCA":
    st.title("TP PCA : Compression, Reconstruction & Débruitage sur MNIST")
    st.subheader("EL MEHDI - Master IAENG")

    @st.cache_data
    def load_mnist_pca():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        mask = (y == 0) | (y == 1)
        return X[mask], y[mask]

    X_pca, y_pca = load_mnist_pca()
    col_ctrl, col_visu = st.columns([1, 2])

    with col_ctrl:
        st.header("⚙️ Paramètres de l'ACP")
        img_idx = st.slider("Sélectionner une image du dataset", 0, len(X_pca) - 1, 0, key="pca_slider")
        image_originale = X_pca[img_idx]
        
        st.markdown("---")
        mode = st.radio("Choisir le type de test :", 
                        ["Test 1 : PCA-1 (Retour en arrière extrême)", 
                         "Test 2 & 3 : Variance ciblée (Optimisation & QR Code)", 
                         "Test 4 : Noise Cancellation (Débruitage)"], key="pca_mode")

        if mode == "Test 1 : PCA-1 (Retour en arrière extrême)":
            n_components = 1
            st.info("PCA-1 : L'image va être écrasée sur un seul axe discriminant.")
        elif mode == "Test 2 & 3 : Variance ciblée (Optimisation & QR Code)":
            variance_target = st.slider("Variance expliquée ciblée", 0.10, 0.99, 0.40, step=0.05, key="pca_var_target")
            pca_temp = PCA().fit(X_pca)
            cum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = int(np.argmax(cum_variance >= variance_target) + 1)
            st.success(f"Nombre d'axes optimisés à retenir : **{n_components}**")
        else:
            n_components = st.slider("Nombre d'axes à conserver (Filtrage du bruit)", 5, 100, 30, key="pca_noise_axes")
            bruit = np.random.normal(0, 0.3, image_originale.shape)
            image_originale = np.clip(image_originale + bruit, 0, 1)

    pca = PCA(n_components=n_components)
    X_compressed = pca.fit_transform(X_pca)
    img_compressed = pca.transform(image_originale.reshape(1, -1))
    image_reconstruite = pca.inverse_transform(img_compressed).reshape(28, 28)

    with col_visu:
        st.header("📊 Visualisation des Résultats")
        fig_imgs, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        title_orig = "Image avec bruit blanc" if "Noise" in mode else "Image Originale (784 px)"
        ax1.imshow(image_originale.reshape(28, 28), cmap='gray')
        ax1.set_title(title_orig)
        ax1.axis('off')
        
        ax2.imshow(image_reconstruite, cmap='gray')
        ax2.set_title(f"Reconstruction (PCA-{n_components})")
        ax2.axis('off')
        st.pyplot(fig_imgs)
        
        st.markdown("---")
        st.write("### 🗜️ L'image sous sa forme COMPRESSÉE")
        fig_comp, ax_comp = plt.subplots(figsize=(8, 1.2))
        im = ax_comp.imshow(img_compressed, cmap='viridis', aspect='auto')
        ax_comp.set_yticks([])
        ax_comp.set_xlabel("Coordonnées dans le nouvel espace")
        ax_comp.set_title(f"Vecteur compressé : {img_compressed.shape[1]} valeur(s) au lieu de 784 !")
        fig_comp.colorbar(im, ax=ax_comp, orientation='horizontal', pad=0.5)
        st.pyplot(fig_comp)

# =========================================================================
# 5. TP-QDA
# =========================================================================
elif choix_tp == "5. TP-QDA":
    st.title("TP QDA : Frontières Quadratiques & Hétérogénéité des Classes")
    st.subheader("EL MEHDI - Master IAENG")

    @st.cache_data
    def load_mnist_qda():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        mask = (y == 0) | (y == 1)
        return X[mask], y[mask]

    X_qda, y_qda = load_mnist_qda()
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_qda)

    col_ctrl, col_visu = st.columns([1, 2])
    with col_ctrl:
        st.header("⚙️ Configuration du Modèle")
        mode_qda = st.radio("Sélectionner le test visuel :", 
                            ["Test 1 : Frontière Courbe (QDA vs LDA)", 
                             "Test 2 : Robustesse et Overfitting (Dimensions)", 
                             "Test 3 : Ellipses de Covariance (Hétérogénéité)"], key="qda_mode")

    lda = LinearDiscriminantAnalysis().fit(X_2d, y_qda)
    qda = QuadraticDiscriminantAnalysis().fit(X_2d, y_qda)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    with col_visu:
        st.header("📊 Graphiques de Décision")
        fig, ax = plt.subplots(figsize=(9, 6))
        idx_sample = np.random.choice(len(X_2d), 500, replace=False)
        ax.scatter(X_2d[idx_sample, 0], X_2d[idx_sample, 1], c=y_qda[idx_sample], cmap='coolwarm', alpha=0.6, edgecolors='k')
        ax.set_xlabel("Composante Principale 1")
        ax.set_ylabel("Composante Principale 2")
        
        if "Test 1" in mode_qda:
            Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contour(xx, yy, Z_lda, colors='red', linewidths=2.5, levels=[0.5])
            Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contour(xx, yy, Z_qda, colors='blue', linewidths=2.5, levels=[0.5])
            ax.set_title("Comparaison des frontières : LDA (Rouge) vs QDA (Bleu)")
            st.pyplot(fig)
        elif "Test 2" in mode_qda:
            fig_dims, ax_dims = plt.subplots(figsize=(9, 4))
            dimensions_list = [2, 5, 10, 15, 20, 30, 40]
            scores = []
            for d in dimensions_list:
                p = PCA(n_components=d)
                X_p = p.fit_transform(X_qda)
                q = QuadraticDiscriminantAnalysis().fit(X_p, y_qda)
                scores.append(q.score(X_p, y_qda))
            ax_dims.plot(dimensions_list, scores, marker='s', color='purple', linestyle='--')
            ax_dims.set_xlabel("Nombre de dimensions")
            ax_dims.set_ylabel("Précision")
            ax_dims.grid(True, alpha=0.3)
            st.pyplot(fig_dims)
        else:
            import matplotlib.patches as patches
            ax.set_title("Hétérogénéité spatiale des classes")
            for class_val, color, name in zip([0, 1], ['red', 'blue'], ['Chiffre 0', 'Chiffre 1']):
                X_class = X_2d[y_qda == class_val]
                mean, cov = np.mean(X_class, axis=0), np.cov(X_class, rowvar=False)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w_el, h_el = 2 * 2 * np.sqrt(vals)
                ellipse = patches.Ellipse(xy=mean, width=w_el, height=h_el, angle=theta, edgecolor=color, facecolor='none', linewidth=3, label=name)
                ax.add_patch(ellipse)
            ax.legend()
            st.pyplot(fig)

# =========================================================================
# 6. TP-RÉGRESSION LOGISTIQUE
# =========================================================================
elif choix_tp == "6. TP-Régression Logistique":
    st.header("EL MEHDI - IAENG")
    st.title("📈 TP-Régression Logistique : Détection du Chiffre 0")
    st.write("Classification binaire avec le Dataset MNIST (0 vs Autres chiffres)")

    @st.cache_data
    def load_data_log():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data[:10000] / 255.0, mnist.target[:10000]
        return X, y

    X, y = load_data_log()
    y_binary = np.where(y == '0', 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_logistic():
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model

    clf = train_logistic()

    st.write("### 📊 Performance du modèle")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Précision (Accuracy) globale sur l'ensemble de test : {acc * 100:.2f}%")

    st.write("### 🔍 Tester le modèle de manière interactive")
    idx = st.number_input("Choisissez un index d'image de test (0 à 1999) :", min_value=0, max_value=1999, value=0, key="log_idx")
    image_to_test = X_test[idx]
    true_label = y_test[idx]
    label_text = "C'est un 0" if true_label == 1 else "Autre chiffre"

    st.image(image_to_test.reshape(28, 28), caption=f"Classe réelle : {label_text}", width=150)
    prediction = clf.predict([image_to_test])[0]
    probabilities = clf.predict_proba([image_to_test])[0]

    st.write("---")
    if prediction == 1:
        st.write(f"🔮 **Prédiction du modèle :** 🟢 **C'est un 0 !** (Confiance : {probabilities[1]*100:.2f}%)")
    else:
        st.write(f"🔮 **Prédiction du modèle :** 🔴 **Anomalie (Ce n'est pas un 0)** (Confiance : {probabilities[0]*100:.2f}%)")

# =========================================================================
# 7. TP-SVM
# =========================================================================
elif choix_tp == "7. TP-SVM (One-Class)":
    st.header("EL MEHDI - IAENG")
    st.title("Détecteur de '0' (One-Class SVM)")

    @st.cache_data
    def load_data_svm():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        return mnist.data / 255.0, mnist.target

    X, y = load_data_svm()

    @st.cache_resource
    def train_model_svm(X, y):
        X_zeros = X[y == '0']
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        model.fit(X_zeros)
        return model

    model_svm = train_model_svm(X, y)
    st.success("Modèle One-Class SVM entraîné sur les '0' !")

    chiffre = st.selectbox("Chiffre à tester :", list(range(10)))
    if st.button("Tester une image"):
        indices = np.where(y == str(chiffre))[0]
        idx = np.random.choice(indices)
        img = X.iloc[idx].values if hasattr(X, "iloc") else X[idx]
        
        st.image(img.reshape(28, 28), width=150)
        pred = model_svm.predict([img])
        if pred[0] == 1:
            st.success("🎯 Résultat : C'est un 0 !")
        else:
            st.error("❌ Résultat : Ce n'est PAS un 0 !")

# =========================================================================
# 8. TP-T-SNE
# =========================================================================
elif choix_tp == "8. TP-t-SNE":
    st.markdown("""
        <div style="background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%); color: white; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 25px;">
            <div style="font-size: 11pt; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85;">EL MEHDI - Master IAENG</div>
            <h1 style="margin: 10px 0 0 0; font-size: 24pt;">TP-t-SNE</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Réduction Non Linéaire & Visualisation des Clusters MNIST</p>
        </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_mnist_tsne():
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X = mnist.data.astype("float32") / 255.0
        y = mnist.target.astype("int")
        return X, y

    X_tsne, y_tsne = load_mnist_tsne()

    st.sidebar.header("Paramètres t-SNE")
    sample_size = st.sidebar.slider("Taille de l'échantillon", 1000, 5000, 2000, 500, key="tsne_size")
    perplexity = st.sidebar.slider("Perplexité", 5, 50, 30, key="tsne_perp")
    learning_rate = st.sidebar.slider("Taux d'apprentissage", 10, 500, 200, key="tsne_lr")

    indices = np.random.choice(len(X_tsne), sample_size, replace=False)
    X_subset = X_tsne.iloc[indices].values if hasattr(X_tsne, "iloc") else X_tsne[indices]
    y_subset = y_tsne.iloc[indices].values if hasattr(y_tsne, "iloc") else y_tsne[indices]

    st.subheader("1. Projection t-SNE en 2D")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    embedding = tsne.fit_transform(X_subset)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y_subset, cmap="tab10", s=10, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Chiffres", loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.add_artist(legend)
    ax.set_title("Projection t-SNE des chiffres MNIST (Espace 2D)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig, bbox_inches="tight")

# =========================================================================
# 9. TP-UMAP
# =========================================================================
elif choix_tp == "9. TP-UMAP":
    st.markdown("""
        <div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 25px;">
            <div style="font-size: 11pt; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85;">EL MEHDI - Master IAENG</div>
            <h1 style="margin: 10px 0 0 0; font-size: 24pt;">TP-UMAP</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Réduction de Dimension non linéaire & Visualisation des clusters MNIST</p>
        </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_mnist_umap():
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X = mnist.data.astype("float32") / 255.0
        y = mnist.target.astype("int")
        return X, y

    X_umap, y_umap = load_mnist_umap()

    st.sidebar.header("Paramètres UMAP")
    sample_size = st.sidebar.slider("Taille de l'échantillon", 1000, 10000, 2500, 500, key="umap_size")
    n_neighbors = st.sidebar.slider("Nombre de voisins", 2, 50, 15, key="umap_neigh")
    min_dist = st.sidebar.slider("Distance minimale", 0.0, 0.9, 0.1, key="umap_dist")

    indices = np.random.choice(len(X_umap), sample_size, replace=False)
    X_subset = X_umap.iloc[indices].values if hasattr(X_umap, "iloc") else X_umap[indices]
    y_subset = y_umap.iloc[indices].values if hasattr(y_umap, "iloc") else y_umap[indices]

    st.subheader("1. Projection UMAP en 2D")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X_subset)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y_subset, cmap="tab10", s=10, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Chiffres", loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.add_artist(legend)
    ax.set_title("Projection UMAP des chiffres MNIST (Espace 2D)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig, bbox_inches="tight")
