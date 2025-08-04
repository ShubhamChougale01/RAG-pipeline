import os
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

qdrant_client = QdrantClient(host="localhost", port=6333)

try:
    qdrant_client.get_collection(collection_name="resume_rag")
    print("Using existing collection: resume_rag")
except Exception as e:
    print(f"Creating new collection: resume_rag (Error: {str(e)})")
    qdrant_client.create_collection(
        collection_name="resume_rag",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE), 
    )

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="resume_rag",
    embedding=embeddings
)

pdf_path = "/Users/shubhamchougale/RAG-pipeline/docs/Shubham_re.pdf"
try:
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text.strip():
        raise ValueError("No text extracted from PDF")
except Exception as e:
    print(f"Error processing PDF: {str(e)}")
    exit(1)

documents = [
    Document(page_content=pdf_text, metadata={"source": "Shubham_re.pdf"})
]

vectorstore.add_documents(documents)
print("Successfully added PDF content to resume_rag collection")