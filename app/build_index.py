import os

from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------------------------------------
# 1. Clean and portable paths (GitHub-friendly)
# --------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)  # /app directory
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "../data/knowledge_base")
CHROMA_INDEX_DIR = os.path.join(BASE_DIR, "../data/chroma_index")

# Create folder if it doesn't exist
os.makedirs(CHROMA_INDEX_DIR, exist_ok=True)

# --------------------------------------------------------
# 2. Embeddings and text splitter
# --------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# --------------------------------------------------------
# 3. Load and split all .txt knowledge files
# --------------------------------------------------------

# Liste qui va contenir tous les documents (chunks) qui seront indexés dans la base vectorielle
docs = []

# Parcourt tous les fichiers .txt de la base de connaissances, lit leur contenu,
# découpe le texte en petits segments (chunks), puis transforme chaque segment
# en objet Document afin de les stocker dans la liste docs pour la création
# ultérieure de la base vectorielle (RAG).
for filename in os.listdir(KNOWLEDGE_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = text_splitter.split_text(content)

        docs.extend([
            Document(page_content=chunk, metadata={"source": filename})
            for chunk in chunks
        ])

# --------------------------------------------------------
# 4. Build Chroma vector index
# --------------------------------------------------------

# Crée l’index vectoriel Chroma à partir des documents (chunks) en générant leurs
# embeddings, puis sauvegarde cet index sur le disque afin qu’il puisse être
# réutilisé par l’application pour effectuer des recherches sémantiques (RAG)
vect_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_INDEX_DIR
)

vect_store.persist()
print(f" Vector index created in: {CHROMA_INDEX_DIR}")
