import streamlit as st
import re
from rag import extract_text, chunk_text, create_faiss_index, retrieve
from llm import get_response

# 1. Page Configuration
st.set_page_config(page_title="AI Python Analyst", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

st.title("🤖 AI Data Analyst (Advanced RAG + FAISS)")

# 2. Sidebar for PDF Upload
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

    if uploaded_file:
        with st.spinner("Reading Document..."):
            # Uses updated rag.py logic (repaired words + overlapping chunks)
            text = extract_text(uploaded_file)
            chunks = chunk_text(text)
            index, _ = create_faiss_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
        st.success("✅ PDF Ready!")

# 3. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Chat Input & Logic
if prompt := st.chat_input("Ask a question..."):
    if st.session_state.faiss_index is None:
        st.warning("Please upload the PDF first.")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # A. Retrieve context (Uses top_k=4 for better variety)
    relevant_chunks = retrieve(prompt, st.session_state.faiss_index, st.session_state.chunks)
    context = "\n".join(relevant_chunks)[:1500] 

    # B. Completion Prompt (Forces model to start answering immediately)
    final_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer with technical bullet points starting with a dot (.):\n."

    # C. Generate and Clean Response
        # 3. Generate and Clean Response
    with st.chat_message("assistant"):
        with st.spinner("Searching PDF..."):
            # A. Get the raw chunks from FAISS
                # Retrieve context
            relevant_chunks = retrieve(prompt, st.session_state.faiss_index, st.session_state.chunks)
            
            # NEW: Filter context to match the topic
            topic_keywords = [w.lower() for w in prompt.split() if len(w) > 3]
            filtered_context = []
            
            for chunk in relevant_chunks:
                # If the user asks for "benefits", prioritize chunks containing "benefits"
                if any(key in chunk.lower() for key in topic_keywords):
                    filtered_context.insert(0, chunk) # Move to top
                else:
                    filtered_context.append(chunk)

            context_text = "\n".join(filtered_context[:3]) # Take the best 3

            
            # B. Ask the model for a clean version (Optional fallback)
            raw_answer = get_response(final_prompt)
            
            # --- THE "STRICT CLEANER" PIPELINE ---
            # We combine the model output and context to find the BEST sentences
            all_text = raw_answer + "\n" + context_text
            
            # Split into individual sentences or lines
            segments = re.split(r'\n|(?<=[.!?])\s+', all_text)
            clean_bullets = []
            
            # Words that indicate "Intro junk" we want to delete
            junk_phrases = ["return only", "do not", "context:", "question:", "instruction", "follows:", "uses it", "google uses"]

            for line in segments:
                line = line.strip()
                
                # 1. KILL REPEATED QUESTIONS & INTRO JUNK
                if line.endswith('?') or any(j in line.lower() for j in junk_phrases):
                    continue
                
                # 2. FRAGMENT FILTER: Ensure the line is a real, informative sentence
                if len(line) < 25 or not line.endswith(('.', '!', '?')):
                    continue

                # 3. PDF MARKER STRIPPER: Remove (i., ii., 1., IV., •)
                # This specifically targets the "i." and "ii." on Page 7 of your PDF
                line = re.sub(r'^([ivxIVX]{1,3}\.|[A-Z0-9]{1,3}\.|[\d\.\-\*\•\s]+)', '', line).strip()
                
                # 4. DEDUPLICATION: Don't add the same point twice
                if line not in clean_bullets:
                    clean_bullets.append(line)

            # --- FINAL OUTPUT FORMATTING ---
            if not clean_bullets:
                answer = "I found the section on Page 7, but I couldn't format it into clean points. Please try: 'List 4 benefits of Python'."
            else:
                # Format with requested "." and double spacing for readability
                # We take the top 5 unique technical points found
                formatted_list = [f". {p}" for p in clean_bullets[:5]]
                answer = "\n\n".join(formatted_list)

            st.markdown(answer)


    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
