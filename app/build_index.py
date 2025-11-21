import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


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
docs = []

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
vect_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_INDEX_DIR
)

vect_store.persist()
print(f" Vector index created in: {CHROMA_INDEX_DIR}")
