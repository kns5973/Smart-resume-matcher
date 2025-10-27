import streamlit as st
import chromadb
import pypdf  # Updated PDF library
from io import BytesIO  # Needed to read file bytes
from sentence_transformers import SentenceTransformer

# --- CORE FUNCTIONS ---

@st.cache_resource  # This caches the model so it doesn't reload every time
def get_embedding_model():
    """Loads the Sentence-Transformer model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource # Caches the database connection
def initialize_database():
    """Initializes the ChromaDB client and collections."""
    client = chromadb.Client() 
    
    resume_collection = client.get_or_create_collection(name="resumes")
    job_collection = client.get_or_create_collection(name="jobs")
    
    return client, resume_collection, job_collection

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file using pypdf."""
    try:
        # Read the file into memory
        pdf_bytes = BytesIO(pdf_file.read())
        
        # Create a PDF reader object
        pdf_reader = pypdf.PdfReader(pdf_bytes)
        
        text = ""
        # Loop through all pages and extract text
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# --- STREAMLIT UI ---
# This is the "frontend" of your app.

# 1. Initialize Model and DB
model = get_embedding_model()
client, resume_collection, job_collection = initialize_database()

st.title("Smart Resume/Internship Matcher 💼")
st.write("Find the best candidate by matching resumes to job descriptions using AI.")

# 2. Sidebar for Uploading Documents
with st.sidebar:
    st.header("Add New Documents")

    # --- Upload Resumes ---
    st.subheader("Upload Resumes")
    uploaded_resumes = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Resumes") and uploaded_resumes:
        for resume_file in uploaded_resumes:
            resume_text = extract_text_from_pdf(resume_file)
            if resume_text:
                # Create embedding
                resume_embedding = model.encode(resume_text).tolist()
                
                # Add to ChromaDB
                resume_collection.add(
                    embeddings=[resume_embedding],
                    documents=[resume_text],
                    metadatas=[{"filename": resume_file.name}],
                    ids=[resume_file.name] # Use filename as a unique ID
                )
        st.sidebar.success(f"Processed {len(uploaded_resumes)} resumes!")

    # --- Add Job Descriptions ---
    st.subheader("Add Job Description")
    job_title = st.text_input("Job Title (e.g., 'Backend Engineer')")
    job_desc = st.text_area("Job Description Text")

    if st.button("Process Job Description") and job_title and job_desc:
        # Create embedding for the job description
        job_embedding = model.encode(job_desc).tolist()
        
        # Add to ChromaDB
        job_collection.add(
            embeddings=[job_embedding],
            documents=[job_desc],
            metadatas=[{"title": job_title}],
            ids=[job_title] # Use job title as a unique ID
        )
        st.sidebar.success(f"Added job: {job_title}")


# 3. Main Area for Matching
st.header("Find Best Candidate Matches")

# Get list of all jobs added to the DB
all_jobs = job_collection.get()
job_titles = []
if all_jobs['metadatas']:
    job_titles = [meta['title'] for meta in all_jobs['metadatas']]

if not job_titles:
    st.warning("Please add a job description via the sidebar to start matching.")
else:
    # Dropdown to select a job
    selected_job_title = st.selectbox("Select Job Description", options=job_titles)
    
    if st.button("Find Top 3 Matches"):
        # 1. Get the selected job's details from the DB
        job_data = job_collection.get(ids=[selected_job_title])
        
        if job_data['documents']:
            query_job_desc = job_data['documents'][0]
            
            # 2. Create the query embedding
            query_embedding = model.encode(query_job_desc).tolist()
            
            # 3. Query the resume collection
            results = resume_collection.query(
                query_embeddings=[query_embedding],
                n_results=3  # Ask for the top 3 matches
            )
            
            st.subheader(f"Top 3 Matches for '{selected_job_title}':")
            
            # 4. Display results
            if not results['ids'][0]:
                st.info("No resumes found in the database. Please upload some.")
            else:
                for i, (resume_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    st.markdown(f"**Match #{i+1}**")
                    st.write(f"**Resume:** `{resume_id}`")
                    st.write(f"**Similarity Score:** {1 - distance:.2f}") # Convert distance to similarity
                    
                    # Expander to show the resume text
                    with st.expander("Show Resume Text"):
                        resume_text = resume_collection.get(ids=[resume_id])['documents'][0]
                        st.write(resume_text[:500] + "...") # Show a snippet
        else:
            st.error("Could not find job data. Please try again.")