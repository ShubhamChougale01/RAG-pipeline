import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_huggingface")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langsmith import Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set TOKENIZERS_PARALLELISM to avoid Hugging Face warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RAGPipeline:
    def __init__(self, pdf_path: str = "/Users/shubhamchougale/RAG-pipeline/docs/Multi-Task Reinforcement Learning for Generalizable Spatial Intelligence.pdf", collection_name: str = "pdf_collection"):
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})
        self.llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.langsmith_client = Client()  # Initialize LangSmith client
        self._initialize()

    def _initialize(self):
        # Load and split PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Initialize in-memory Qdrant client
        client = QdrantClient(":memory:")
        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        # Initialize Qdrant vector store with in-memory client
        self.vector_store = Qdrant(
            client=client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

        # Add documents to vector store
        self.vector_store.add_documents(splits)
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Create RAG chain with LangSmith tracing
        prompt_template = """
        You are an intelligent assistant. Answer the query based on the provided context.
        Context: {context}
        Query: {query}
        Answer: """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = (
            {"context": self.retriever, "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        ).with_config({"run_name": "RAGPipeline"})  # Add LangSmith run name for tracing

    def query(self, question: str) -> str:
        try:
            # Run query with LangSmith tracing
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"Error processing query: {str(e)}"