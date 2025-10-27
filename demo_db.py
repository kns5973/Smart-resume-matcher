"""
This is the main Python file for the Streamlit app.
The project description is now part of the UI below.
"""

import streamlit as st
import chromadb
import pypdf  # PDF library
from io import BytesIO  # Needed to read file bytes
from sentence_transformers import SentenceTransformer

# --- CORE FUNCTIONS ---

@st.cache_resource  # This caches the model so it doesn't reload
def get_embedding_model():
    """Loads the Sentence-Transformer model."""
    # Using st.progress to show a user-friendly loading bar
    with st.spinner("Loading AI model... (This runs only once on startup)"):
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource # Caches the database connection
def initialize_database():
    """
    Initializes the ChromaDB client. 
    This is the main connection to our vector DBMS.
    """
    client = chromadb.Client() 
    
    # --- DB OPERATION ---
    # Get or create our "tables" (collections)
    resume_collection = client.get_or_create_collection(name="resumes")
    job_collection = client.get_or_create_collection(name="jobs")
    
    return client, resume_collection, job_collection

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file using pypdf."""
    try:
        pdf_bytes = BytesIO(pdf_file.read())
        pdf_reader = pypdf.PdfReader(pdf_bytes)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# --- STREAMLIT UI ---

# 1. Initialize Model and DB Connection
model = get_embedding_model()
client, resume_collection, job_collection = initialize_database()

st.title("Smart Resume/Internship Matcher 💼")
st.write("A DBMS project using a Vector Database (ChromaDB) to perform semantic search.")

# --- PROJECT DESCRIPTION (MOVED TO UI) ---
with st.expander("Show Project Description & Workflow", expanded=False):
    st.markdown("""
    ### DESCRIPTION
    This application is an intelligent matching system. It finds the best student resumes for a given job description based on **semantic meaning**, not just keyword matching.

    ---
    
    ### WORKFLOW
    1.  **INGESTION (Adding Data):**
        * A user uploads a PDF resume or enters a text job description.
        * The system extracts the raw text.
        * An AI model (SentenceTransformer) converts this text into a high-dimensional vector (an "embedding"). This vector numerically represents the *meaning* of the text.
        * This vector and its metadata are then stored in the database.

    2.  **QUERYING (Finding Matches):**
        * The user selects a job description to find matches for.
        * The system retrieves the vector for that job from the database.
        * This job vector is used as a *query*.
        * The system asks the database: "Find the top 3 resume vectors that are most similar to this job vector."
        * The database performs a high-speed similarity search and returns the top 3 matches.
        * The application displays these matches to the user.

    ---

    ### ⭐️ HIGHLIGHT: THE DBMS CORE
    The AI model only *creates* the vectors, but the **database** is the core engine that makes this project work.

    * **DATABASE TYPE:** We use **ChromaDB**, a specialized **Vector Database**. This is a modern NoSQL database designed for storing and querying high-dimensional vector data.

    * **"SCHEMA" DESIGN:**
        * `resume_collection`: Stores all resume vectors and their text.
        * `job_collection`: Stores all job vectors and their text.

    * **INDEXING:** When we add a vector, ChromaDB builds a special index (like HNSW). This index allows for *extremely fast* and efficient similarity search, which a traditional SQL database cannot do.

    * **CORE OPERATIONS:** The project is built around two key DBMS operations:
        * `collection.add()`: This is our **"INSERT"** statement. We write the document (text), metadata (title), and embedding (vector) into the database.
        * `collection.query()`: This is our **"SELECT"** statement. We give it a query vector, and the database itself handles the complex math of finding the closest matches.
    """)
# --- END OF DESCRIPTION ---


# 2. Sidebar for Data Ingestion (Adding to the DB)
with st.sidebar:
    st.header("Add New Documents to Database")

    # --- Upload Resumes ---
    st.subheader("Upload Resumes (INSERT)")
    uploaded_resumes = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Resumes") and uploaded_resumes:
        with st.spinner("Processing resumes..."):
            for resume_file in uploaded_resumes:
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    # 1. Create embedding (AI part)
                    resume_embedding = model.encode(resume_text).tolist()
                    
                    # 2. --- DB OPERATION ---
                    resume_collection.add(
                        embeddings=[resume_embedding],
                        documents=[resume_text],
                        metadatas=[{"filename": resume_file.name}],
                        ids=[resume_file.name] 
                    )
            st.sidebar.success(f"Processed and added {len(uploaded_resumes)} resumes to the database!")

    # --- Add Job Descriptions ---
    st.subheader("Add Job Description (INSERT)")
    job_title = st.text_input("Job Title (e.g., 'Backend Engineer')")
    job_desc = st.text_area("Job Description Text")

    if st.button("Process Job Description") and job_title and job_desc:
        with st.spinner("Processing job description..."):
            # 1. Create embedding (AI part)
            job_embedding = model.encode(job_desc).tolist()
            
            # 2. --- DB OPERATION ---
            job_collection.add(
                embeddings=[job_embedding],
                documents=[job_desc],
                metadatas=[{"title": job_title}],
                ids=[job_title] 
            )
        st.sidebar.success(f"Added job '{job_title}' to the database!")


# 3. Main Area for Querying the Database
st.header("Find Best Candidate Matches (QUERY)")

# --- DB OPERATION ---
all_jobs = job_collection.get()
job_titles = []
if all_jobs['metadatas']:
    job_titles = [meta['title'] for meta in all_jobs['metadatas']]

if not job_titles:
    st.warning("Database is empty. Please add a job description via the sidebar to start matching.")
else:
    # Dropdown to select a job
    selected_job_title = st.selectbox("Select Job Description to Query With:", options=job_titles)
    
    if st.button("Find Top 3 Matches"):
        with st.spinner(f"Querying database for matches to '{selected_job_title}'..."):
            
            # 1. --- DB OPERATION ---
            job_data = job_collection.get(ids=[selected_job_title], include=["embeddings"])
            
            if job_data['embeddings']:
                query_embedding = job_data['embeddings'][0]
                
                # 2. --- THE CORE DB QUERY ---
                results = resume_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3  
                )
                
                st.subheader(f"Top 3 Matches for '{selected_job_title}':")
                
                # 3. Display results
                if not results['ids'][0]:
                    st.info("No resumes found in the database. Please upload some.")
                else:
                    for i, (resume_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                        st.markdown(f"**Match #{i+1}**")
                        st.write(f"**Resume:** `{resume_id}`")
                        st.write(f"**Similarity Score:** {1 - distance:.2f} (Closer to 1 is better)")
                        
                        # Get the resume text to show it
                        resume_text = resume_collection.get(ids=[resume_id])['documents'][0]
                        with st.expander("Show Resume Text Snippet"):
                            st.write(resume_text[:500] + "...")
            else:
                st.error("Could not find job data. Please try again.")
