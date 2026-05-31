#Install dependencies from requirements.txt before running the app:
import os
import re
import numpy as np
import streamlit as st
import faiss
import time 

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from groq import APIError, RateLimitError, APITimeoutError

#Deciding on the model's and output variations based on the number of pages in the PDF
load_dotenv()

try:
    GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")

NOTES_MODEL = "llama-3.1-8b-instant"
QA_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CONTEXT_CHARS = 12000
MAX_ANSWER_TOKENS = 800
MAX_FINAL_NOTES_INPUT_CHARS = 12000

MAX_PDF_PAGES = 15
MAX_FILE_SIZE_MB = 10
MAX_NOTE_SECTIONS = 8
MAX_CHARS_PER_NOTE_SECTION = 4000
MAX_NOTES_GENERATIONS_PER_SESSION = 1
MAX_QA_QUESTIONS_PER_SESSION = 10

def choose_rag_settings(page_count):
    if page_count <= 5:
        chunk_size = 800
        k = 6
        fetch_k = 20
        pdf_type = "Small PDF"

    elif page_count <= 10:
        chunk_size = 900
        k = 7
        fetch_k = 25
        pdf_type = "Medium PDF"

    else:
        chunk_size = 1000
        k = 8
        fetch_k = 30
        pdf_type = "Large PDF"

    chunk_overlap = int(chunk_size * 0.30)

    return {
        "pdf_type": pdf_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k,
        "fetch_k": fetch_k
    }

#Reading the PDF and extracting text from it, while also keeping track of the number of pages for later use in deciding the RAG settings
def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    page_count = len(reader.pages)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    return full_text, page_count

#Cleaning the extracted text by removing excessive newlines, tabs, and extra spaces
def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text

#Create chunks of the cleaned text based on the chunk size and overlap determined by the RAG settings
def split_text_into_chunks(text, chunk_size, chunk_overlap):
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    overlap_text = current_chunk[-chunk_overlap:] if current_chunk else ""
                    current_chunk = overlap_text + " " + sentence

        else:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk += "\n\n" + paragraph
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                overlap_text = current_chunk[-chunk_overlap:] if current_chunk else ""
                current_chunk = overlap_text + "\n\n" + paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# This function loads the embedding model once and reuses it.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# This function creates embeddings for chunks and stores them inside a FAISS index. 
#We use cosine similarity for retrieval, so we normalize the embeddings and use an inner product index. 
def create_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings

# This function calculates cosine similarity between one vector and many vectors.
def cosine_similarity(query_vector, candidate_vectors):
    return np.dot(candidate_vectors, query_vector.T).flatten()

# This function selects relevant but diverse chunks using MMR.
def mmr_select(question_embedding, candidate_embeddings, candidate_indices, k, lambda_mult=0.7):
    if len(candidate_indices) == 0:
        return []
    
    selected_indices = []
    remaining_indices = list(range(len(candidate_indices)))

    question_scores = cosine_similarity(question_embedding[0], candidate_embeddings)

    first_choice = int(np.argmax(question_scores))
    selected_indices.append(first_choice)
    remaining_indices.remove(first_choice)

    while len(selected_indices) < min(k, len(candidate_indices)) and remaining_indices:
        best_score = -float("inf")
        best_candidate = None

        for candidate in remaining_indices:
            relevance_score = question_scores[candidate]

            selected_vectors = candidate_embeddings[selected_indices]
            candidate_vector = candidate_embeddings[candidate].reshape(1, -1)

            diversity_scores = cosine_similarity(candidate_vector[0], selected_vectors)
            max_similarity_to_selected = np.max(diversity_scores)

            mmr_score = (
                lambda_mult * relevance_score
                - (1 - lambda_mult) * max_similarity_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)

    original_chunk_indices = [candidate_indices[i] for i in selected_indices]

    return original_chunk_indices

# This function retrieves the best PDF chunks for a user question using FAISS + MMR.
def retrieve_relevant_chunks(question, chunks, embedding_model, index, embeddings, k, fetch_k):
    question_embedding = embedding_model.encode([question], convert_to_numpy=True)
    question_embedding = question_embedding.astype("float32")

    faiss.normalize_L2(question_embedding)

    fetch_k = min(fetch_k, len(chunks))

    distances, candidate_indices = index.search(question_embedding, fetch_k)

    candidate_indices = candidate_indices[0]
    candidate_indices = [int(i) for i in candidate_indices if i != -1]

    if not candidate_indices:
        return [], [], ""

    candidate_embeddings = embeddings[candidate_indices]

    selected_indices = mmr_select(
        question_embedding=question_embedding,
        candidate_embeddings=candidate_embeddings,
        candidate_indices=candidate_indices,
        k=k,
        lambda_mult=0.9
    )

    selected_chunks = []
    final_indices = []
    context = ""

    for idx in selected_indices:
        chunk = chunks[idx]

        if len(context) + len(chunk) + 2 > MAX_CONTEXT_CHARS:
            break

        selected_chunks.append(chunk)
        final_indices.append(idx)
        context += f"\n\n--- Chunk {idx + 1} ---\n{chunk}"

    return selected_chunks, final_indices, context

# This function creates and returns the Groq API client.
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to your .env file.")
        st.stop()

    return Groq(
        api_key=api_key,
        timeout=90.0
    )


def call_llm(prompt, model, max_tokens=1200, temperature=0.2):
    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Use only the provided text. Be accurate, structured, and exam-oriented. Do not hallucinate."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except RateLimitError:
        st.error(
            "Groq rate limit reached. Please wait and try again later, "
            "or reduce the PDF size."
        )
        return None

    except APITimeoutError:
        st.error(
            "The AI request timed out. Try again with a smaller PDF or fewer pages."
        )
        return None

    except APIError as e:
        st.error(f"Groq API error: {str(e)}")
        return None

    except Exception as e:
        st.error(f"Unexpected AI error: {str(e)}")
        return None

# This function generates a high-quality answer using only the retrieved PDF context.
def generate_answer(question, retrieved_context):
    if not retrieved_context.strip():
        return "I could not find this in the PDF."

    prompt = f"""
You are a highly accurate PDF question-answering assistant.

You must answer the user's question using ONLY the PDF context provided below.

STRICT RULES:
1. Use only the PDF context.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not invent facts, names, dates, examples, definitions, formulas, or statistics.
5. If the answer is not present, say exactly:
   "I could not find this in the PDF."
6. If the answer is partially present, state the available part and clearly mention what is missing.
7. If the question asks for "all", "list", "features", "points", "requirements", "advantages", "disadvantages", or "types", extract as complete a list as possible from the provided PDF context.
8. Preserve the PDF's terminology.
9. Combine related points across the retrieved context.
10. Do not mention chunks or retrieval in the final answer.

ANSWER FORMAT:
- Start with a direct answer.
- Use bullet points for lists.
- Use numbered steps for processes.
- Use a table for comparisons.
- For exam-style answers, include a short explanation after the main points.

PDF CONTEXT:
{retrieved_context}

USER QUESTION:
{question}

Now answer accurately using only the PDF context.
"""

    answer = call_llm(
        prompt=prompt,
        model=QA_MODEL,
        max_tokens=MAX_ANSWER_TOKENS,
        temperature=0.1
    )

    if answer is None:
        return "The answer could not be generated because the AI API request failed."

    return answer

# This function generates clear and detailed study notes from one PDF chunk without making them unnecessarily long.
def generate_mini_notes(section_text, section_number, total_sections):
    prompt = f"""
Convert the PDF section into detailed, accurate, exam-oriented study notes.

Section {section_number} of {total_sections}.

Rules:
- Use only the PDF section text.
- Do not add outside knowledge.
- Do not hallucinate.
- Do not skip important points.
- Rewrite clearly instead of copying blindly.
- Preserve important terminology, definitions, examples, formulas, rules, steps, cases, dates, classifications, and distinctions.
- If something is incomplete because it continues elsewhere, summarize only what is available.
- Remove obvious repetition, headers, footers, and page noise.

Output requirements:
- Start with: ## Section {section_number}: Detailed Study Notes
- Then create only the headings that are actually useful for this section.
- Do not create empty headings.
- Do not write filler like "Not applicable" unless absolutely necessary.
- Use clear headings and subheadings based on the actual PDF content.
- Use bullet points for lists.
- Use numbered steps only for processes.
- Include examples, definitions, formulas, comparisons, advantages/disadvantages, or questions only if they are relevant to this section.
- End with a short "Exam Revision Points" section.

Suggested structure, but use only what fits:
- Overview
- Detailed Notes
- Key Terms / Definitions
- Important Points
- Steps / Process / Framework
- Examples / Applications
- Comparisons / Advantages / Disadvantages
- Exam Revision Points
- Possible Questions

PDF SECTION:
{section_text}

Generate the study notes now.
"""

    notes = call_llm(
        prompt=prompt,
        model=NOTES_MODEL,
        max_tokens=650,
        temperature=0.1
    )

    if notes is None:
        return None

    return notes

def group_chunks_for_notes(chunks):
    sections = []
    current_section = ""

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if len(current_section) + len(chunk) + 2 <= MAX_CHARS_PER_NOTE_SECTION:
            current_section += "\n\n" + chunk
        else:
            if current_section.strip():
                sections.append(current_section.strip())
            current_section = chunk

    if current_section.strip():
        sections.append(current_section.strip())

    return sections

def generate_notes_from_chunks(chunks):
    note_sections = group_chunks_for_notes(chunks)

    st.info(
        f"Your PDF has been grouped into {len(note_sections)} note sections. "
        f"This reduces API usage while still covering the full PDF."
    )

    mini_notes_list = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_sections = len(note_sections)

    for i, section_text in enumerate(note_sections, start=1):
        status_text.write(
            f"Generating notes for section {i} of {total_sections}..."
        )

        section_notes = generate_mini_notes(
            section_text=section_text,
            section_number=i,
            total_sections=total_sections
        )

        if section_notes is None:
            st.warning(
                "Notes generation stopped because one AI request failed. "
                "Try again later or use a smaller PDF."
            )
            return None, None

        mini_notes_list.append(
            f"## Section {i} Notes\n\n{section_notes}"
        )

        progress_bar.progress(i / total_sections)

        if i < total_sections:
            status_text.write(
                f"Section {i} completed. Waiting briefly to avoid API token limits..."
                f"Please be patient, there is a gap of 40 seconds between processing each section."
                f"It's still faster than GPT and Claude lol."
            )
            time.sleep(40)

    status_text.write("Preparing final notes document...")

    all_mini_notes = "\n\n---\n\n".join(mini_notes_list)

    final_notes = f"""
# Study Notes

These notes were generated section-wise from the uploaded PDF.

## Document Processing Summary

- Total note sections generated: {total_sections}
- The PDF was divided into larger sections to reduce API usage and avoid token-rate-limit errors.
- The notes below preserve the section-wise structure of the PDF.

---

{all_mini_notes}
"""

    progress_bar.progress(1.0)
    status_text.write("Notes generated successfully.")

    return final_notes, all_mini_notes

# This function estimates a safe max_tokens value for final notes based on mini-notes length.
def calculate_final_notes_max_tokens(all_mini_notes):
    estimated_input_tokens = len(all_mini_notes) // 4
    calculated_tokens = int(estimated_input_tokens * 0.6)
    final_max_tokens = min(calculated_tokens, 1600)
    return final_max_tokens

# This function combines all mini-notes into one final polished notes document.
def smart_trim_text(text, max_chars):
    if len(text) <= max_chars:
        return text

    first_part = text[:max_chars // 2]
    last_part = text[-max_chars // 2:]

    return (
        first_part
        + "\n\n[Some middle content was shortened due to model input limits.]\n\n"
        + last_part
    )

def generate_final_notes(all_mini_notes):
    #all_mini_notes = smart_trim_text(
        #all_mini_notes,
        #MAX_FINAL_NOTES_INPUT_CHARS
    #)

    final_max_tokens = calculate_final_notes_max_tokens(all_mini_notes)

    if final_max_tokens < 1500:
        final_max_tokens = 1500

    prompt = f"""
Combine the section-wise notes into one polished study-notes document.

Rules:
- Use only the given notes.
- Do not add outside knowledge.
- Preserve important definitions, rules, formulas, examples, cases, dates, steps, classifications, and distinctions.
- Remove repetition.
- Merge overlapping points.
- Improve structure and flow.
- Keep the final notes detailed, clear, and exam-oriented.
- Do not create empty sections.
- Use headings only where useful.

Required:
# Study Notes

After that, organize the content naturally based on the actual material.
Use headings such as Overview, Detailed Notes, Key Terms, Rules/Formulas/Steps, Examples, Comparisons, Revision Points, or Possible Questions only when relevant.

SECTION-WISE NOTES:
{all_mini_notes}

Create the final polished notes.
"""

    final_notes = call_llm(
        prompt=prompt,
        model=NOTES_MODEL,
        max_tokens=final_max_tokens,
        temperature=0.1
    )

    if final_notes is None:
        return None

    return final_notes

#For downloading the notes as a TXT file, we need to remove any markdown formatting that may be present in the generated notes.
def convert_notes_to_txt(notes):
    return notes.replace("#", "").replace("*", "")


def get_download_file_name(uploaded_file, extension):
    base_name = uploaded_file.name.rsplit(".", 1)[0]
    safe_name = base_name.replace(" ", "_")
    return f"{safe_name}_study_notes.{extension}"

#=========================================================================================================================#
#Streamlit app code starts here. The above functions are imported and used in the app to create the user interface and handle user interactions.

# This is the main Streamlit user interface for the PDF Study Assistant.
st.set_page_config(
    page_title="GenAI PDF Study Assistant",
    page_icon="📘",
    layout="wide"
)

st.title("📘 GenAI PDF Study Assistant")

if "notes_generation_count" not in st.session_state:
    st.session_state.notes_generation_count = 0

if "qa_question_count" not in st.session_state:
    st.session_state.qa_question_count = 0

if "generated_notes" not in st.session_state:
    st.session_state.generated_notes = None

if "all_mini_notes" not in st.session_state:
    st.session_state.all_mini_notes = None

st.warning(
    "Privacy note: PDF text may be sent to an external AI API for notes and answers. "
    "Do not upload confidential or sensitive documents unless you are comfortable with that."
)
st.write("Upload a PDF to generate structured study notes and ask questions from the document.")

uploaded_file = st.file_uploader(
    "Upload your PDF (maximum 10 MB)",
    type=["pdf"],
    max_upload_size=10
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large. Please upload a PDF under {MAX_FILE_SIZE_MB} MB.")
        st.stop()
    
    with st.spinner("Reading PDF..."):
        raw_text, page_count = read_pdf(uploaded_file)

        if page_count > MAX_PDF_PAGES:
            st.error(
                f"This PDF has {page_count} pages. "
                f"For the deployed version, please upload PDFs up to {MAX_PDF_PAGES} pages."
            )
            st.stop()

        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            st.error("Could not extract readable text from this PDF.")
            st.stop()

    rag_settings = choose_rag_settings(page_count)

    chunks = split_text_into_chunks(
        text=cleaned_text,
        chunk_size=rag_settings["chunk_size"],
        chunk_overlap=rag_settings["chunk_overlap"]
    )

    note_sections_preview = group_chunks_for_notes(chunks)

    st.info(
        f"For study notes, this PDF will be grouped into {len(note_sections_preview)} larger sections. "
        f"That means notes generation will use about {len(note_sections_preview)} AI calls, "
        f"not {len(chunks)} calls."
    )

    st.success("PDF processed successfully.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pages", page_count)

    with col2:
        st.metric("PDF Type", rag_settings["pdf_type"])

    with col3:
        st.metric("Chunks", len(chunks))

    with col4:
        st.metric("Chunk Size", rag_settings["chunk_size"])

    embedding_model = load_embedding_model()

    with st.spinner("Creating FAISS index..."):
        index, embeddings = create_faiss_index(chunks, embedding_model)

        notes_tab, qa_tab = st.tabs(["📝 Study Notes", "❓ PDF Q&A"])

    with notes_tab:
        st.subheader("Generate Study Notes")
        st.write("Create structured, exam-oriented notes from the full PDF.")

        if st.button("Generate Study Notes"):
            if st.session_state.notes_generation_count >= MAX_NOTES_GENERATIONS_PER_SESSION:
                st.error(
                    "Notes generation limit reached for this session. "
                    "Please refresh the app or try again later."
                )
            else:
                with st.spinner("Generating detailed section-wise notes..."):
                    raw_notes, all_mini_notes = generate_notes_from_chunks(chunks)

                if raw_notes:
                        with st.spinner("Polishing final study notes..."):
                            polished_notes = generate_final_notes(all_mini_notes)

                        if polished_notes:
                            st.session_state.generated_notes = polished_notes
                        else:
                            st.session_state.generated_notes = raw_notes

                        st.session_state.all_mini_notes = all_mini_notes
                        st.session_state.notes_generation_count += 1
                        st.success("Study notes generated successfully.")

        if st.session_state.generated_notes:
            st.markdown(st.session_state.generated_notes)

            st.subheader("Download Notes")

            markdown_file_name = get_download_file_name(uploaded_file, "md")
            txt_file_name = get_download_file_name(uploaded_file, "txt")

            st.download_button(
                label="Download as Markdown",
                data=st.session_state.generated_notes,
                file_name=markdown_file_name,
                mime="text/markdown"
            )

            st.download_button(
                label="Download as TXT",
                data=convert_notes_to_txt(st.session_state.generated_notes),
                file_name=txt_file_name,
                mime="text/plain"
            )

            if "all_mini_notes" in st.session_state and st.session_state.all_mini_notes:
                with st.expander("View raw section notes"):
                    st.markdown(st.session_state.all_mini_notes)

    with qa_tab:
        st.subheader("Ask Questions from the PDF")

        question = st.text_input("Ask a question from the PDF:")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")

            elif st.session_state.qa_question_count >= MAX_QA_QUESTIONS_PER_SESSION:
                st.error(
                    "Q&A limit reached for this session. "
                    "Please refresh the app or try again later."
                )

            else:
                with st.spinner("Retrieving relevant PDF chunks and generating answer..."):
                    selected_chunks, selected_indices, context = retrieve_relevant_chunks(
                        question=question,
                        chunks=chunks,
                        embedding_model=embedding_model,
                        index=index,
                        embeddings=embeddings,
                        k=rag_settings["k"],
                        fetch_k=rag_settings["fetch_k"]
                    )

                    answer = generate_answer(question, context)

                st.session_state.qa_question_count += 1

                st.subheader("Answer")
                st.write(answer)

                with st.expander("View retrieved PDF chunks"):
                    for chunk_number, chunk_text in zip(selected_indices, selected_chunks):
                        st.markdown(f"### Chunk {chunk_number + 1}")
                        st.write(chunk_text)
                        st.divider()

else:
    st.info("Upload a PDF to begin.")