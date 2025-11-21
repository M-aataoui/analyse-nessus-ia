# Analyse intelligente des vulnérabilités Nessus

Application Streamlit utilisant :
- PCA + clustering (KMeans / DBSCAN)
- Base de connaissances vectorielle (Chroma)
- IA locale via Ollama (mistral)

## Installation
pip install -r requirements.txt

## Construire l'index vectoriel
python app/build_index.py

## Lancer l'application
streamlit run app/app.py
