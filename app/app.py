import streamlit as st
st.set_page_config(page_title="Analyse Nessus + IA", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from kneed import KneeLocator

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def charger_base_vect():
    vect_path = r"C:\Users\LOQ\Desktop\base_vect_index_chroma"  # adapte si besoin
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=vect_path, embedding_function=embedding)


vecteur_db = charger_base_vect()


def get_contexte_vecteur(question: str, k: int = 3) -> str:
    docs = vecteur_db.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in docs])


def ollama_est_actif() -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        return r.status_code in (200, 404)
    except:
        return False


def obtenir_analyse_ollama(prompt: str, model: str = "mistral") -> str:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180
        )
        return r.json()["response"] if r.ok else "❌ Erreur Ollama"
    except Exception as e:
        return f"⚠️ Impossible de contacter Ollama : {e}"


def severity_from_cvss(score: float) -> str:
    if score == 0:
        return "Informational"
    if score < 4.0:
        return "Low"
    if score < 7.0:
        return "Medium"
    if score < 9.0:
        return "High"
    return "Critical"


def prompt_cluster(c):
    return (
        f"Cluster {c['Cluster']} – Gravité : {c['Gravité dominante']} – "
        f"Score CVSS moyen : {c['CVSS moyen']} – "
        f"{c['Nb vulnérabilités']} vulnérabilités détectées – "
        f"Port dominant : {c['Port dominant']} – Protocole : {c['Protocole dominant']}\n\n"
        "Analyse les vulnérabilités de ce cluster à l’aide du contexte technique fourni, incluant des CVE connues, "
        "des descriptions de protocoles, des services sensibles et des mesures correctives.\n\n"
        "Commence par une **analyse technique** précise : types d'exploits, services concernés, CVE similaires, vecteurs d'exploitation, etc.\n"
        "Ensuite, propose des **recommandations concrètes** : correctifs, protocoles à désactiver, outils de mitigation, bonnes pratiques.\n\n"
        "Tu répondras exclusivement en français, avec un ton professionnel et structuré."
    )


st.title("Analyse intelligente des vulnérabilités  avec IA")
uploaded_file = st.file_uploader("📁 Charger un rapport Nessus (CSV)", type="csv")

if "cluster_reports_done" not in st.session_state:
    st.session_state["cluster_reports_done"] = False
if "cluster_reports" not in st.session_state:
    st.session_state["cluster_reports"] = []
if "anomaly_report" not in st.session_state:
    st.session_state["anomaly_report"] = ""

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    # --- colonne CVSS / port / protocole -----------------
    col_cvss = next((c for c in df.columns if "cvss" in c.lower()), None)
    col_port = next((c for c in df.columns if c.lower() == "port"), None)
    col_proto = next((c for c in df.columns if "protocol" in c.lower()), None)

    df["CVSS"] = pd.to_numeric(df[col_cvss], errors="coerce").fillna(0)
    df["Risk_orig"] = df.get("Risk", df.get("Severity", "None"))
    df["Risk"] = df["CVSS"].apply(severity_from_cvss)

    risk_map = {"Informational": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    df["Risk_num"] = df["Risk"].map(risk_map)
    df["Port"] = pd.to_numeric(df[col_port], errors="coerce").fillna(0)
    df["Protocol"] = df[col_proto].fillna("unknown")
    df["Protocol_num"] = LabelEncoder().fit_transform(df["Protocol"])

    # =====================================================
    # 1) NEW: handle NaNs BEFORE PCA / clustering
    # =====================================================
    feature_cols = ["CVSS", "Risk_num", "Port", "Protocol_num"]
    df_features = df[feature_cols].copy()

    # drop rows with any NaN in features
    mask_valid = df_features.notna().all(axis=1)
    df_clust = df.loc[mask_valid].copy()
    X_features = df_features.loc[mask_valid]

    if df_clust.shape[0] < 2:
        st.error(
            "Pas assez de lignes sans valeurs manquantes pour lancer PCA / clustering "
            f"({df_clust.shape[0]} ligne valide)."
        )
        st.stop()

    # =====================================================
    # 2) PCA + KMeans / DBSCAN on cleaned data only
    # =====================================================
    X_scaled = StandardScaler().fit_transform(X_features)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    wcss = [KMeans(n_clusters=k, random_state=42).fit(X_pca).inertia_ for k in range(1, 8)]
    k_opt = KneeLocator(range(1, 8), wcss, curve="convex", direction="decreasing").elbow or 3

    st.sidebar.subheader("🔧 Paramètres clustering")
    k_kmeans = st.sidebar.slider("Clusters KMeans", 2, 6, int(k_opt))
    eps = st.sidebar.slider("ε (DBSCAN)", 0.1, 2.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("min_samples DBSCAN", 2, 10, 5)

    kmeans_labels = KMeans(n_clusters=k_kmeans, random_state=42).fit_predict(X_pca)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X_pca)
    dbscan_labels = dbscan.labels_

    df_clust["Cluster_KMeans"] = kmeans_labels
    df_clust["Cluster_DBSCAN"] = dbscan_labels
    df_clust["Anomalie"] = df_clust["Cluster_DBSCAN"] == -1

    # =====================================================
    # 3) Visualisation + résumé toujours basés sur df_clust
    # =====================================================
    st.subheader(" Projection PCA (couleur=KMeans, losange=anomalie DBSCAN)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df_clust["Cluster_KMeans"],
        style=df_clust["Anomalie"],
        palette=sns.color_palette("Set2", k_kmeans),
        s=80,
        ax=ax
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

    st.subheader("📄 Synthèse clusters")
    summary = []
    for cid in sorted(df_clust["Cluster_KMeans"].unique()):
        sub = df_clust[df_clust["Cluster_KMeans"] == cid]
        cvss_moy = sub["CVSS"].mean()
        sev_dom = severity_from_cvss(cvss_moy)

        port_mode = sub["Port"].mode()
        port_dominant = int(port_mode[0]) if not port_mode.empty else -1

        proto_mode = sub["Protocol"].mode()
        proto_dominant = proto_mode[0] if not proto_mode.empty else "Inconnu"

        summary.append({
            "Cluster": cid,
            "Gravité dominante": sev_dom,
            "CVSS moyen": round(cvss_moy, 2),
            "Port dominant": port_dominant,
            "Protocole dominant": proto_dominant,
            "Nb vulnérabilités": len(sub)
        })
    st.dataframe(pd.DataFrame(summary))

    st.subheader("📋 Détail des anomalies détectées par DBSCAN")
    anomalies = df_clust[df_clust["Anomalie"]]
    if anomalies.empty:
        st.info("Aucune anomalie détectée.")
    else:
        st.dataframe(anomalies[["CVSS", "Risk", "Port", "Protocol"]])

    # =====================================================
    # 4) IA Ollama (inchangé)
    # =====================================================
    if ollama_est_actif():
        st.subheader(" Analyse IA des clusters")

        if not st.session_state["cluster_reports_done"]:
            if st.button("Analyser les clusters avec Ollama"):
                for c in summary:
                    with st.spinner(f"Analyse du cluster {c['Cluster']}."):
                        prompt = prompt_cluster(c)
                        contexte = get_contexte_vecteur(prompt)
                        prompt_final = contexte + "\n\n" + prompt
                        reply = obtenir_analyse_ollama(prompt_final)
                        html = f"### Cluster {c['Cluster']} – {c['Gravité dominante']}<br>{reply.replace(chr(10), '<br>')}"
                        st.session_state["cluster_reports"].append(html)
                        st.markdown(html, unsafe_allow_html=True)
                st.session_state["cluster_reports_done"] = True
        else:
            for html in st.session_state["cluster_reports"]:
                st.markdown(html, unsafe_allow_html=True)
