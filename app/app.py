import os
import streamlit as st

st.set_page_config(page_title="Analyse Nessus + IA", layout="wide")

# =========================
# AUTHENTIFICATION SIMPLE
# =========================
USERS = {
    "admin": "admin123",
    "analyst": "analyst123"
}

def login():
    st.sidebar.subheader("🔐 Authentification")
    user = st.sidebar.text_input("Utilisateur", key="login_user")
    pwd = st.sidebar.text_input("Mot de passe", type="password", key="login_pwd")

    if st.sidebar.button("Se connecter"):
        if USERS.get(user) == pwd:
            st.session_state["auth"] = True
            st.session_state["user"] = user
            st.sidebar.success("✅ Connecté")
            st.rerun()
        else:
            st.sidebar.error("❌ Identifiants invalides")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login()
    st.stop()

st.sidebar.success(f"Connecté : {st.session_state.get('user','')}")

# =========================
# IMPORTS
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.preprocessing import LabelEncoder, StandardScaler
import umap.umap_ as umap
import hdbscan

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# Vector DB (RAG)
# =========================
# Charge la base vectorielle Chroma utilisée par le RAG : vérifie si l’index
# vectoriel existe localement, puis initialise la connexion à cette base avec
# le modèle d’embeddings. Si l’index n’existe pas, retourne None pour permettre
# à l’application de fonctionner même sans le système RAG.
@st.cache_resource
def charger_base_vect():
    """
    Cherche un index Chroma local (portable) : ../data/chroma_index
    Si introuvable, retourne None (l'app marche quand même, sans RAG).
    """
    base_dir = os.path.dirname(__file__)
    vect_path = os.path.normpath(os.path.join(base_dir, "../data/chroma_index"))

    if not os.path.exists(vect_path):
        return None

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #ouvre la base vectorielle existante et la connecte avec le modèle d'embeddings pour les recherches sémantiques (RAG)  
    return Chroma(persist_directory=vect_path, embedding_function=embedding)

vecteur_db = charger_base_vect()

def get_contexte_vecteur(question: str, k: int = 3) -> str:
    if vecteur_db is None:
        return ""
    docs = vecteur_db.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

# =========================
# Ollama
# =========================
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
        return r.json().get("response", "") if r.ok else "❌ Erreur Ollama"
    except Exception as e:
        return f"⚠️ Impossible de contacter Ollama : {e}"

# =========================
# Helpers
# =========================
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

def topk_dist_str(series: pd.Series, k: int = 3) -> str:
    """
    Retourne un résumé: '25 (41%), 5432 (20%), 22 (9%), autres (30%)'
    """
    if series.empty:
        return "N/A"

    vc = series.value_counts(dropna=False)
    total = int(vc.sum())
    top = vc.head(k)

    parts = []
    top_sum = 0
    for val, cnt in top.items():
        pct = (cnt / total) * 100 if total else 0
        parts.append(f"{val} ({pct:.0f}%)")
        top_sum += int(cnt)

    rest = total - top_sum
    if rest > 0:
        rest_pct = (rest / total) * 100
        parts.append(f"autres ({rest_pct:.0f}%)")

    return ", ".join(parts)

def sample_cluster_rows(df_clust: pd.DataFrame, cluster_id: int, n: int = 12) -> str:
    """
    Donne un mini-échantillon réel du cluster (preuve) pour éviter
    une analyse Ollama trop générique.
    """
    cols = [c for c in ["CVSS", "Risk", "Port", "Protocol"] if c in df_clust.columns]
    sample = df_clust[df_clust["Cluster"] == cluster_id][cols].head(n)
    if sample.empty:
        return ""
    return sample.to_csv(index=False)

def build_cluster_prompt(c: dict, df_clust: pd.DataFrame) -> str:
    """
    Prompt adapté au nouveau résumé (Top ports + stats CVSS) + échantillon réel.
    """
    sample_csv = sample_cluster_rows(df_clust, c["Cluster"], n=12)

    prompt = f"""
Tu es un expert cybersécurité (pentest + hardening + remédiation).
On analyse un cluster de vulnérabilités issues d’un rapport Nessus, regroupées automatiquement.

## Résumé du cluster
- ID cluster: {c['Cluster']}
- Nombre de vulnérabilités: {c['Nb vulnérabilités']}
- Gravité dominante: {c['Gravité dominante']}
- CVSS: moyenne={c['CVSS moyen']}, médian={c['CVSS médian']}, max={c['CVSS max']}
- Ports les plus fréquents: {c['Top ports']}
- Protocoles les plus fréquents: {c['Top protocoles']}

## Travail demandé
1) **Interprétation technique**: à quels services correspondent probablement les ports principaux (ex: 25=SMTP, 5432=PostgreSQL, 445=SMB, 53=DNS, 80/443=HTTP(S), 22=SSH, 3306=MySQL, 3389=RDP...).
2) **Risques typiques** associés à ces services (mauvaise configuration, auth faible, version obsolète, TLS faible, brute force, RCE, injection, etc.).
3) **Priorisation**: quoi corriger en premier et pourquoi (CVSS max + exposition + impact).
4) **Recommandations concrètes**: patching/upgrade, durcissement, segmentation, firewall, monitoring/logs, validation.
5) Termine par une **checklist actionnable** (6–10 points).

Contraintes:
- Réponds en **français**
- Sois **structuré** (titres + puces)
- Évite le blabla générique : appuie-toi sur les ports/protocoles et le CVSS.
""".strip()

    if sample_csv:
        prompt += "\n\n## Échantillon réel du cluster (CSV)\n" + sample_csv

    return prompt

# =========================
# UI
# =========================
st.title("Analyse intelligente des vulnérabilités avec IA (UMAP + HDBSCAN)")
uploaded_file = st.file_uploader("📁 Charger un rapport Nessus (CSV)", type="csv")

# état UI IA
if "cluster_reports_done" not in st.session_state:
    st.session_state["cluster_reports_done"] = False
if "cluster_reports" not in st.session_state:
    st.session_state["cluster_reports"] = []

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    # --- colonnes Nessus ---
    col_cvss = next((c for c in df.columns if "cvss" in c.lower()), None)
    col_port = next((c for c in df.columns if c.lower() == "port"), None)
    col_proto = next((c for c in df.columns if "protocol" in c.lower()), None)

    if not col_cvss or not col_port or not col_proto:
        st.error("Colonnes attendues introuvables (CVSS / Port / Protocol). Vérifie ton CSV Nessus.")
        st.stop()

    # Nettoyage / normalisation
    df["CVSS"] = pd.to_numeric(df[col_cvss], errors="coerce").fillna(0)
    df["Port"] = pd.to_numeric(df[col_port], errors="coerce").fillna(0).astype(int)
    df["Protocol"] = df[col_proto].fillna("unknown").astype(str).str.lower().str.strip()

    # Risk (affichage / résumé)
    df["Risk"] = df["CVSS"].apply(severity_from_cvss)

    # Encodage protocole
    df["Protocol_num"] = LabelEncoder().fit_transform(df["Protocol"])

    # =====================================================
    # Features (ML)
    # =====================================================
    feature_cols = ["CVSS", "Port", "Protocol_num"]
    df_features = df[feature_cols].copy()

    # on garde les lignes valides ET on conserve leur index
    mask_valid = df_features.notna().all(axis=1)
    df_clust = df.loc[mask_valid].copy()
    X_features = df_features.loc[mask_valid]

    if df_clust.shape[0] < 5:
        st.error(f"Pas assez de lignes valides pour clustering ({df_clust.shape[0]} lignes).")
        st.stop()

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # =====================================================
    # Paramètres UMAP + HDBSCAN
    # =====================================================
    st.sidebar.subheader("🔧 Paramètres UMAP + HDBSCAN")

    n_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 50, 15, 1)
    min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 1.0, 0.1, 0.05)

    min_cluster_size = st.sidebar.slider("HDBSCAN min_cluster_size", 5, 80, 15, 1)
    min_samples = st.sidebar.slider("HDBSCAN min_samples", 1, 40, 5, 1)

    # UMAP = visualisation
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    X_umap = umap_model.fit_transform(X_scaled)

    # HDBSCAN = clustering (recommandé sur X_scaled)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(X_scaled)  # -1 = anomalies

    # IMPORTANT: labels et df_clust sont alignés car on a conservé mask_valid
    df_clust["Cluster"] = labels
    df_clust["Anomalie"] = df_clust["Cluster"] == -1

    # =====================================================
    # Visualisation UMAP
    # =====================================================
    st.subheader("📌 Projection UMAP (couleur=HDBSCAN, losange=anomalie)")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=df_clust["Cluster"].astype(str),
        style=df_clust["Anomalie"],
        s=80,
        ax=ax
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    st.pyplot(fig)

    # =====================================================
    # Synthèse clusters (HORS -1) — VERSION FIABLE
    # =====================================================
    st.subheader("📄 Synthèse clusters (HDBSCAN)")

    summary = []
    clusters = sorted([c for c in df_clust["Cluster"].unique() if c != -1])

    nb_total = int(len(df_clust))
    nb_noise = int((df_clust["Cluster"] == -1).sum())
    noise_rate = (nb_noise / nb_total) * 100 if nb_total else 0
    st.caption(f"Total lignes analysées: {nb_total} | Bruit / anomalies (-1): {nb_noise} ({noise_rate:.1f}%)")

    if not clusters:
        st.warning(
            "HDBSCAN n'a détecté aucun cluster (tout est considéré comme bruit). "
            "Essaie d'augmenter min_cluster_size ou de diminuer min_samples."
        )
    else:
        for cid in clusters:
            sub = df_clust[df_clust["Cluster"] == cid]
            n = int(len(sub))

            cvss_mean = float(sub["CVSS"].mean()) if n else 0.0
            cvss_median = float(sub["CVSS"].median()) if n else 0.0
            cvss_max = float(sub["CVSS"].max()) if n else 0.0

            risk_mode = sub["Risk"].mode()
            sev_dom = risk_mode.iloc[0] if not risk_mode.empty else "Inconnu"

            summary.append({
                "Cluster": int(cid),
                "Nb vulnérabilités": n,
                "Gravité dominante": sev_dom,
                "CVSS moyen": round(cvss_mean, 2),
                "CVSS médian": round(cvss_median, 2),
                "CVSS max": round(cvss_max, 2),
                "Top ports": topk_dist_str(sub["Port"], k=3),
                "Top protocoles": topk_dist_str(sub["Protocol"], k=2)
            })

        summary_df = pd.DataFrame(summary).sort_values(
            ["CVSS moyen", "Nb vulnérabilités"],
            ascending=False
        )
        st.dataframe(summary_df, use_container_width=True)

        # =====================================================
        # Détails par cluster (distributions) — BUG DUPLICATE FIXED
        # =====================================================
        with st.expander("📊 Détails par cluster (distribution CVSS / ports)"):
            chosen = st.selectbox("Choisir un cluster", clusters, key="cluster_detail_select")
            sub = df_clust[df_clust["Cluster"] == chosen]

            c1, c2 = st.columns(2)
            with c1:
                st.write("Distribution CVSS (Top 10 valeurs) :")
                st.dataframe(
                    sub["CVSS"]
                    .value_counts()
                    .head(10)
                    .rename_axis("CVSS")
                    .reset_index(name="count"),
                    use_container_width=True
                )

            with c2:
                st.write("Top ports (Top 10) :")
                st.dataframe(
                    sub["Port"]
                    .value_counts()
                    .head(10)
                    .rename_axis("Port")
                    .reset_index(name="count"),
                    use_container_width=True
                )

    # =====================================================
    # Anomalies
    # =====================================================
    st.subheader("📋 Détail des anomalies détectées (HDBSCAN = -1)")
    anomalies = df_clust[df_clust["Anomalie"]]
    if anomalies.empty:
        st.info("Aucune anomalie détectée.")
    else:
        st.dataframe(
            anomalies[["CVSS", "Risk", "Port", "Protocol"]].sort_values(["CVSS", "Port"], ascending=False),
            use_container_width=True
        )

    # =====================================================
    # IA Ollama (RAG + LLM) — PROMPT ADAPTÉ
    # =====================================================
    if vecteur_db is None:
        st.warning("⚠️ Index Chroma introuvable (../data/chroma_index). Lance build_index.py pour activer le RAG.")

    if ollama_est_actif():
        st.subheader("🤖 Analyse IA des clusters (RAG + Ollama)")

        if not st.session_state["cluster_reports_done"]:
            if st.button("Analyser les clusters avec Ollama"):
                st.session_state["cluster_reports"] = []

                if not summary:
                    st.warning("Aucun cluster exploitable (tout est bruit).")
                else:
                    for c in summary:
                        with st.spinner(f"Analyse du cluster {c['Cluster']}..."):
                            prompt = build_cluster_prompt(c, df_clust)

                            # RAG (si dispo)
                            contexte = get_contexte_vecteur(prompt) if vecteur_db is not None else ""
                            prompt_final = (contexte + "\n\n" + prompt).strip() if contexte else prompt

                            reply = obtenir_analyse_ollama(prompt_final)

                            html = (
                                f"### Cluster {c['Cluster']} – {c['Gravité dominante']}<br>"
                                f"{reply.replace(chr(10), '<br>')}"
                            )
                            st.session_state["cluster_reports"].append(html)
                            st.markdown(html, unsafe_allow_html=True)

                    st.session_state["cluster_reports_done"] = True
        else:
            for html in st.session_state["cluster_reports"]:
                st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("Ollama n'est pas détecté (http://localhost:11434). Lance Ollama pour activer l'analyse IA.")
else:
    st.info("Charge un fichier CSV Nessus pour démarrer l'analyse.")