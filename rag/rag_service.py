import os
import sys
import glob
import re
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

class NeuroFinanceRAGService:
    def __init__(self, persist_dir="vector_store", data_dir="data/knowledge"):
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        self.collection_name = "neurofinance_kb"
        
        # Load embedding model
        self.embed_model = None
        if os.environ.get("RENDER") is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("RAG Service: Loading sentence-transformer model (all-MiniLM-L6-v2)...")
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("RAG Service: Embedding model loaded successfully!")
            except Exception as e:
                print(f"RAG Service Warning: Failed to load embedding model: {e}. Fallback to keyword-matching search is active.")
        else:
            print("RAG Service: Running on Render (memory-constrained). Bypassing heavy SentenceTransformer loading to avoid OOM.")
            
        # Setup ChromaDB client
        self.chroma_client = None
        self.collection = None
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("RAG Service: ChromaDB client initialized successfully!")
        except Exception as e:
            print(f"RAG Service Warning: Failed to setup ChromaDB: {e}. Fallback to local file search is active.")
            
        # Setup Gemini
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key and self.api_key != "your_key_here":
            genai.configure(api_key=self.api_key)
            self.model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            print("RAG Service: Gemini API configured successfully!")
        else:
            self.model_gemini = None
            print("RAG Service Warning: GEMINI_API_KEY is missing or default. RAG generation will run in fallback/mock mode.")

    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def index_documents(self):
        """Loads txt/md files from data_dir, chunks them, computes embeddings, and stores in ChromaDB."""
        if self.embed_model is None or self.collection is None:
            print("RAG Service Indexing Warning: Embeddings or ChromaDB is not available. Skipping persistent vector indexing.")
            return

        path = Path(self.data_dir)
        if not path.exists():
            print(f"RAG Service: Data directory {self.data_dir} does not exist. Skipping indexing.")
            return
            
        files = glob.glob(str(path / "*.txt")) + glob.glob(str(path / "*.md"))
        if not files:
            print("RAG Service: No document files found to index.")
            return
            
        print(f"RAG Service: Indexing {len(files)} files...")
        
        # Reset collection to avoid duplicate indexings
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
        
        doc_id = 0
        for file_path in files:
            file_name = Path(file_path).name
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = self.chunk_text(content)
            print(f"  - {file_name}: split into {len(chunks)} chunks.")
            
            for idx, chunk in enumerate(chunks):
                # Compute embedding
                embedding = self.embed_model.encode(chunk).tolist()
                
                # Metadata
                metadata = {
                    "source": file_name,
                    "chunk_id": idx
                }
                
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[f"doc_{doc_id}"]
                )
                doc_id += 1
                
        print("RAG Service: Indexing complete!")

    def query_local_fallback(self, query_text, top_k=3):
        """Standard word-overlap keyword matching search across documents in data_dir (completely offline)."""
        chunks = []
        path = Path(self.data_dir)
        if path.exists():
            files = glob.glob(str(path / "*.txt")) + glob.glob(str(path / "*.md"))
            for file_path in files:
                file_name = Path(file_path).name
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    for ch in self.chunk_text(content):
                        chunks.append({"text": ch, "source": file_name})
                except Exception:
                    pass
                    
        if not chunks:
            return [], []
            
        # Basic word match scoring
        query_words = set(re.findall(r"\w+", query_text.lower()))
        scored_chunks = []
        for ch in chunks:
            chunk_words = re.findall(r"\w+", ch["text"].lower())
            intersection = query_words.intersection(set(chunk_words))
            score = len(intersection)
            scored_chunks.append((score, ch))
            
        # Sort by match score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Select best matches
        best_matches = [item[1] for item in scored_chunks[:top_k] if item[0] > 0]
        
        # If no word matches, just return the first few chunks of our index
        if not best_matches:
            best_matches = chunks[:top_k]
            
        retrieved_chunks = [ch["text"] for ch in best_matches]
        sources = list(set([ch["source"] for ch in best_matches]))
        return retrieved_chunks, sources

    def query(self, query_text, top_k=3):
        """Retrieves relevant context chunks and passes to Gemini to synthesize an answer."""
        retrieved_chunks = []
        sources = []
        
        # 1. Try retrieving matching chunks from ChromaDB if active
        if self.embed_model is not None and self.collection is not None:
            try:
                query_embedding = self.embed_model.encode(query_text).tolist()
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                retrieved_chunks = results["documents"][0] if results["documents"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []
                sources = list(set([m["source"] for m in metadatas]))
            except Exception as e:
                print(f"RAG Service Warning: Vector query failed: {e}. Falling back to offline local search.")
                retrieved_chunks, sources = [], []
                
        # 2. Local fallback search if vector db is offline or empty
        if not retrieved_chunks:
            retrieved_chunks, sources = self.query_local_fallback(query_text, top_k=top_k)
            
        if not retrieved_chunks:
            return {
                "answer": "No relevant financial documents found in the local knowledge base.",
                "sources": []
            }
            
        # Combine context
        context = "\n\n".join([f"[Source: {src}]\n{doc}" for doc, src in zip(retrieved_chunks, sources)])
        
        # 3. Synthesize answer using Gemini
        if self.model_gemini is not None:
            prompt = (
                f"You are NeuroBot, an AI financial decision assistant. Answer the user's question "
                f"based ONLY on the provided financial documents context below.\n"
                f"If the answer cannot be determined from the context, respond by stating that the "
                f"information is not found in the approved guidelines.\n\n"
                f"--- CONTEXT ---\n{context}\n\n"
                f"--- QUESTION ---\n{query_text}\n\n"
                f"--- ANSWER ---"
            )
            try:
                response = self.model_gemini.generate_content(prompt)
                answer = response.text.strip()
            except Exception as e:
                answer = f"Error generating answer via Gemini: {e}\n\n[Retrieved Context]:\n{context}"
        else:
            # Fallback/mock mode when API key is missing
            answer = (
                f"[Local Offline Mode - GEMINI_API_KEY missing]\n"
                f"Based on the knowledge base guidelines, here is the retrieved information:\n\n"
                + "\n* ".join(retrieved_chunks)
            )
            
        return {
            "answer": answer,
            "sources": sources,
            "context_chunks": retrieved_chunks
        }

if __name__ == "__main__":
    service = NeuroFinanceRAGService()
    service.index_documents()
    print("\nTesting Query:")
    res = service.query("What documents are required for salaried individuals?")
    print(f"Answer:\n{res['answer']}")
    print(f"Sources: {res['sources']}")
