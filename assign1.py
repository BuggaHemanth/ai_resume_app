# %%
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import logging
import re

# %%
# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Access the Google API key
google_api_key = os.environ.get("GOOGLE_API_KEY")

# %%
# Set page config
st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="wide")

# Add this line to set file upload limit to 6MB
st.config.set_option('server.maxUploadSize', 6)

# %%
# Setup embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS vectorstore as None (will be created when needed)
vectorstore = None

# %%
def scrape_job_posting(url):
    """Scrape job requirements from a job posting URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Use LLM to extract relevant job information
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.2
        )
        
        extraction_prompt = f"""Extract and summarize the key job requirements from the following job posting text. Focus on:

1. Job Title and Company
2. Required Skills and Technologies
3. Experience Requirements
4. Education/Qualifications
5. Responsibilities
6. Nice to Have Skills

Job Posting Text:
{text[:4000]}

Provide a clean, structured summary focusing only on the most relevant hiring criteria."""
        
        extracted_requirements = llm.invoke(extraction_prompt).content
        return extracted_requirements
        
    except requests.RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"Error processing the webpage: {str(e)}"

# %%
def extract_text_from_resume(file):
    temp_file_path = f"temp_{file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# %%
# Text splitting
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# %%
# Store resume analysis in FAISS vector store
def store_resume_analysis(analysis, doc_id):
    global vectorstore
    documents = split_text(analysis)
    
    # Create new FAISS vectorstore or add to existing one
    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents, embedding_model)
    else:
        # Add documents to existing vectorstore
        new_vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.merge_from(new_vectorstore)
    
    # Store in session state for persistence during session
    st.session_state.vectorstore = vectorstore

# %%
# Extract percentage score from analysis text
def extract_suitability_score(text):
    match = re.search(r"Suitability Score: (\d{1,3})%", text)
    if match:
        return int(match.group(1))
    return None

# %%
# Main App
def main():
    global vectorstore
    
    # Load vectorstore from session state if available
    if 'vectorstore' in st.session_state:
        vectorstore = st.session_state.vectorstore
    
    st.markdown("<h2 style='text-align: center; color: #1f4e79;'>🎯 AI-Powered Resume Screening System</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # Create 3 columns layout
    col1, col2, col3 = st.columns([1, 1, 0.8], gap='medium')
    
    # Column 1: Job Requirements
    with col1:
        st.markdown("### 📋 Job Requirements")
        
        # Job input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Job Link", "Manual Input"],
            horizontal=True
        )
        
        job_requirements = ""
        
        if input_method == "Job Link":
            job_url = st.text_input(
                "Enter job posting URL:",
                placeholder="https://example.com/job-posting"
            )
            
            if job_url and st.button("🔍 Scrape Job Requirements", use_container_width=True):
                with st.spinner("Scraping job requirements..."):
                    job_requirements = scrape_job_posting(job_url)
                    st.session_state.job_requirements = job_requirements
            
            # Display scraped requirements if available
            if "job_requirements" in st.session_state:
                st.text_area(
                    "Extracted Requirements:", 
                    st.session_state.job_requirements, 
                    height=300,
                    disabled=True
                )
                job_requirements = st.session_state.job_requirements
        
        else:  # Manual Input
            job_requirements = st.text_area(
                "Enter job requirements manually:", 
                height=300, 
                placeholder="Describe the role, required skills, experience..."
            )
    
    # Column 2: Resume Upload and Analysis
    with col2:
        st.markdown("### 📄 Resume Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload Resume", 
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        # Show resume preview if uploaded
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
        # Analyze button - always visible
        analyze_disabled = not (uploaded_file and job_requirements.strip())
        
        if st.button(
            "🚀 Analyze Resume", 
            type="primary", 
            use_container_width=True,
            disabled=analyze_disabled
        ):
            if not job_requirements.strip():
                st.error("Please provide job requirements first!")
            elif not uploaded_file:
                st.error("Please upload a resume first!")
            else:
                with st.spinner("🔍 Analyzing resume..."):
                    # Extract resume text
                    resume_text = extract_text_from_resume(uploaded_file)
                    
                    # Show extracted text in expander
                    with st.expander("📖 View Resume Text"):
                        st.text_area("Extracted Text", resume_text, height=150)

                    # Initialize LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=google_api_key,
                        temperature=0.2
                    )

                    # Create analysis prompt
                    prompt = f"""Analyze this resume against the job requirements and provide:

1. Suitability Score: X% (where X is 0-100)
2. Brief Summary (2-3 sentences)
3. Areas for Improvement (bullet points only)
4. Missing Skills/Experience (bullet points only)

Job Requirements:
{job_requirements}

Resume:
{resume_text}

Keep the response concise and focused on actionable feedback."""
                    
                    # Get AI analysis
                    analysis = llm.invoke(prompt).content
                    
                    # Store analysis in session state
                    st.session_state.analysis = analysis
                    st.session_state.resume_text = resume_text
                    
                    # Extract and display suitability score
                    score = extract_suitability_score(analysis)
                    if score:
                        st.markdown(
                            f"<div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                            f"padding: 20px; border-radius: 15px; margin: 20px 0;'>"
                            f"<h1 style='color: white; font-size: 3rem; margin: 0;'>{score}%</h1>"
                            f"<h3 style='color: #f0f0f0; margin: 5px 0 0 0;'>Suitability Score</h3>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )

                    st.markdown("**📊 Analysis Results**")
                    st.markdown(analysis)

                    # Store in FAISS
                    store_resume_analysis(analysis, uploaded_file.name)
                    st.success("✅ Analysis stored successfully!")

    # Column 3: Q&A Section
    with col3:
        st.markdown("### 💬 Ask Questions")
        
        question = st.text_area(
            "Ask about analyzed resumes:",
            height=100,
            placeholder="e.g., What skills are missing?"
        )
        
        if st.button("Get Answer", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question!")
            elif vectorstore is None:
                st.warning("No resume data found. Please analyze a resume first.")
            else:
                try:
                    # Use FAISS similarity search
                    docs = vectorstore.similarity_search(question, k=3)
                    context = " ".join([doc.page_content for doc in docs])
                    
                    if context.strip():
                        # Create Q&A prompt
                        qa_prompt = f"""Based on the resume analysis context below, answer this question concisely:

Question: {question}

Context: {context}

Provide a brief, direct answer in 2-3 sentences."""
                        
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.0-flash",
                            google_api_key=google_api_key,
                            temperature=0.2
                        )
                        
                        answer = llm.invoke(qa_prompt).content
                        st.markdown(f"**Answer:** {answer}")
                    else:
                        st.warning("No relevant information found. Please analyze a resume first.")
                        
                except Exception as e:
                    st.error(f"Error retrieving data: {str(e)}")
        
        # Show chat history if available
        if "analysis" in st.session_state:
            st.markdown("---")
            st.markdown("**Recent Analysis Available** ✅")

if __name__ == "__main__":
    main()

# %%
