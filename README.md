Smart Resume/Internship Matcher 💼
This project is a web application that revolutionizes the hiring process by using AI to find the best candidates for a job.

Instead of relying on simple keyword matching, this tool understands the semantic meaning behind the text in both resumes and job descriptions. It uses sentence-transformer models to convert documents into vector embeddings and stores them in a ChromaDB vector database.

When you query with a job description, it performs a similarity search to find and rank the most contextually relevant resumes, even if they don't share the exact same keywords.

Core Features:
Semantic Matching: Understands that "built a scalable backend service" is similar to "experience with microservice architecture."

Easy UI: Built with Streamlit for a simple interface to upload resumes (PDFs) and add new jobs.

Vector Database: Uses ChromaDB to efficiently store and query high-dimensional text embeddings.

Instant Results: Quickly ranks the top candidates for any given job description.
