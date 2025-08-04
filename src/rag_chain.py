import os
from langchain_groq import ChatGroq
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langsmith import Client
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from PyPDF2 import PdfReader

pdf_path = "/RAG-pipeline/docs/Shubham_re.pdf"
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

langsmith_client = Client()
COLLECTION_NAME = "resume_rag"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise ValueError("No text found in PDF.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

def get_vectorstore():
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        try:
            qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            print("Loaded existing collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Creating new collection: {COLLECTION_NAME} (Error: {str(e)})")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
    except Exception as e:
        raise Exception(f"Failed to initialize vector db: {str(e)}")

def upsert_pdf_to_vectorstore(pdf_path):
    try:
        text = extract_text_from_pdf(pdf_path)
        documents = [Document(page_content=text, metadata={"source": os.path.basename(pdf_path)})]
        vectorstore = get_or_create_vectorstore()
        vectorstore.add_documents(documents)
        print(f"Successfully upserted PDF content in '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Failed to process and upsert PDF: {str(e)}")

def query_resume(query: str) -> str:
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            callbacks=[LangChainTracer(project_name="ai-pipeline", client=langsmith_client)]
        )
        response = qa_chain.invoke({"query": query})["result"]

        return response
    except Exception as e:
        return f"RAG Pipeline failed: {str(e)}"