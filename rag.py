import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

# Load embedding model (lightweight and fast for local RAG)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(uploaded_file):
    """
    Extracts text from PDF and repairs specific broken words 
    (ligatures) without deleting spaces between normal words.
    """
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            # --- TARGETED WORD REPAIR ---
            # We fix ONLY these common PDF-to-text errors found in your document
            content = content.replace("abstr action", "abstraction")
            content = content.replace("algorit hm", "algorithm")
            content = content.replace("dif ference", "difference")
            content = content.replace("per formance", "performance")
            content = content.replace("T uple", "Tuple")
            content = content.replace("L ist", "List")
            content = content.replace("efficien t","efficient")
            content = content.replace("W e","We")
            
            # Remove any stray "  " (double spaces) but keep single spaces
            content = re.sub(r'\s+', ' ', content)
            
            text += content + "\n"

    return text

def chunk_text(text, chunk_size=150, overlap=40):
    """
    Chunks text with an overlap. 
    Overlap ensures that if a definition is at the end of a chunk, 
    the full context is preserved in the next one.
    """
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        # Create a chunk of 'chunk_size' words
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        # Move forward by (chunk_size - overlap) to create the overlap
        i += (chunk_size - overlap)

    return chunks

def create_faiss_index(chunks):
    """
    Converts text chunks into vectors and stores them in a FAISS index.
    """
    # Generate embeddings
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    
    # Get vector dimensions (usually 384 for MiniLM)
    dimension = embeddings.shape[1]
    
    # Create the FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings (convert to float32 for FAISS compatibility)
    index.add(np.array(embeddings).astype('float32'))

    return index, embeddings

def retrieve(query, index, chunks, top_k=4):
    # BOOSTER: We add the query words multiple times to make FAISS 
    # strictly look for those specific keywords (e.g. "benefits")
    boosted_query = f"{query} {query} {query}"
    
    query_embedding = embed_model.encode([boosted_query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    return [chunks[i] for i in indices[0]]

