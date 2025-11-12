import streamlit as st
import os
import io
from datetime import datetime, timezone
import PyPDF2
import uuid
import json
import numpy as np
import requests
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
import hashlib
import re
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication Configuration
# You can store these in environment variables for better security
VALID_USERS = {
    "Youri": "Youri@#neuro",
    "Sofina": "Sofina@#1984",
    "taaha": "Taaha@#1888",
    "Furqan": "Furqan@#1998"
    # Add more users as needed
}

# You can also load from environment variables:
# VALID_USERS = {
#     os.getenv('AUTH_USER1', 'admin'): os.getenv('AUTH_PASS1', 'admin123'),
#     os.getenv('AUTH_USER2', 'user1'): os.getenv('AUTH_PASS2', 'password123'),
# }

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def authenticate_user(username, password):
    """Authenticate user credentials"""
    return VALID_USERS.get(username) == password

def login_form():
    """Display login form"""
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 60vh;">
        <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); width: 400px;">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="color: #667eea; margin-bottom: 0.5rem;">🔐 Login Required</h1>
                <h2 style="color: #666; font-weight: normal; font-size: 1.2rem;">RAG Document Assistant</h2>
                <p style="color: #888; margin-top: 1rem;">Please enter your credentials to access the application</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### 👤 Authentication")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.error("⚠️ Please enter both username and password")
        
        # Optional: Show demo credentials (remove in production)
        # with st.expander("🔧 Demo Credentials"):
        #     st.info("""
        #     **Demo Accounts:**
        #     - Username: `admin` | Password: `admin123`
        #     - Username: `user1` | Password: `password123`
        #     - Username: `client` | Password: `client2024`
            
        #     *Note: These are demo credentials. Change them in production!*
        #     """)

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

# Check authentication first
if not check_authentication():
    login_form()
    st.stop()

# Custom CSS (only applied after authentication)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .folder-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metadata-info {
        font-size: 0.8em;
        color: #666;
        font-style: italic;
    }
    .logout-btn {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
OPENAI_EMBEDDING_ENDPOINT = os.getenv('OPENAI_EMBEDDING_ENDPOINT')
OPENAI_EMBEDDING_KEY = os.getenv('OPENAI_EMBEDDING_KEY')
OPENAI_CHAT_ENDPOINT = os.getenv('OPENAI_CHAT_ENDPOINT')
OPENAI_CHAT_KEY = os.getenv('OPENAI_CHAT_KEY')
MONGODB_CONNECTION_STRING = os.getenv('MONGODB_CONNECTION_STRING')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'RAGDatabase')
BLOB_CONNECTION_STRING = os.getenv('BlobStorageConnection')

# Configuration constants
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4.1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CONTAINER_NAME = "documents"

class DirectOpenAIClient:
    """Direct API client for Azure OpenAI without using the OpenAI library"""
    
    def __init__(self, api_key, endpoint, api_version):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')
        self.api_version = api_version
        
    def create_embedding(self, text, model):
        """Create embedding using direct API call"""
        url = f"{self.endpoint}/openai/deployments/{model}/embeddings"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        data = {
            "input": text
        }
        
        params = {"api-version": self.api_version}
        
        try:
            response = requests.post(url, headers=headers, json=data, params=params, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            st.error(f"Embedding API error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error in embedding: {str(e)}")
            return None
    
    def create_chat_completion(self, messages, model, temperature=0.7, max_tokens=1000):
        """Create chat completion using direct API call"""
        url = f"{self.endpoint}/openai/deployments/{model}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        params = {"api-version": self.api_version}
        
        try:
            response = requests.post(url, headers=headers, json=data, params=params, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"Chat API error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error in chat: {str(e)}")
            return None

# Initialize Azure clients
@st.cache_resource
def init_clients():
    # Direct API clients
    embedding_client = DirectOpenAIClient(
        api_key=OPENAI_EMBEDDING_KEY,
        endpoint=OPENAI_EMBEDDING_ENDPOINT,
        api_version="2023-05-15"
    )
    
    chat_client = DirectOpenAIClient(
        api_key=OPENAI_CHAT_KEY,
        endpoint=OPENAI_CHAT_ENDPOINT,
        api_version="2024-02-01"
    )
    
    # MongoDB client
    mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
    database = mongo_client[DATABASE_NAME]
    
    # Collections
    documents_collection = database.documents
    chat_collection = database.chat_sessions
    
    # Create indexes for better performance
    try:
        documents_collection.create_index("folder")
        documents_collection.create_index([("folder", 1), ("documentName", 1)])
        chat_collection.create_index("sessionId")
    except Exception as e:
        pass  # Indexes may already exist
    
    # Blob storage client
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    
    return embedding_client, chat_client, documents_collection, chat_collection, blob_service_client

embedding_client, chat_client, documents_collection, chat_collection, blob_service_client = init_clients()

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_file):
        """Extract text from PDF with metadata"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            metadata = {
                "total_pages": len(pdf_reader.pages),
                "author": "",
                "title": ""
            }
            
            # Extract metadata
            if pdf_reader.metadata:
                metadata.update({
                    "author": str(pdf_reader.metadata.get('/Author', '')),
                    "title": str(pdf_reader.metadata.get('/Title', ''))
                })
            
            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            return text.strip(), metadata
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None, None

    @staticmethod
    def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_id = 0
        
        # Simple sentence detection for better chunking
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": chunk_id
                })
                
                # Handle overlap
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else []
                current_chunk = " ".join(overlap_words) + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": chunk_id
            })
        
        return chunks

class MongoRAGService:
    @staticmethod
    def generate_embeddings(text):
        """Generate embeddings using direct API client"""
        try:
            return embedding_client.create_embedding(text, EMBEDDING_MODEL)
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None

    @staticmethod
    def store_document_chunks(folder_name, document_name, chunks, pdf_metadata):
        """Store document chunks in MongoDB"""
        try:
            stored_count = 0
            
            for chunk_data in chunks:
                chunk_text = chunk_data["text"]
                chunk_id = chunk_data["chunk_id"]
                
                # Generate embedding for chunk
                embedding = MongoRAGService.generate_embeddings(chunk_text)
                if embedding is None:
                    continue
                
                # Create document for MongoDB
                doc_id = f"{folder_name}_{document_name}_{chunk_id}"
                
                document = {
                    "_id": doc_id,
                    "folder": folder_name,
                    "documentName": document_name,
                    "chunkIndex": chunk_id,
                    "text": chunk_text,
                    "embedding": embedding,
                    "totalPages": pdf_metadata.get('total_pages', 0),
                    "author": pdf_metadata.get('author', ''),
                    "title": pdf_metadata.get('title', ''),
                    "createdAt": datetime.now(timezone.utc),
                    "uploadedBy": st.session_state.get("username", "unknown")
                }
                
                try:
                    documents_collection.insert_one(document)
                    stored_count += 1
                except Exception as e:
                    # Document might already exist, try to update it
                    try:
                        documents_collection.replace_one({"_id": doc_id}, document, upsert=True)
                        stored_count += 1
                    except Exception as e2:
                        st.error(f"Error storing chunk {chunk_id}: {str(e2)}")
            
            return stored_count
        except Exception as e:
            st.error(f"Error storing document chunks: {str(e)}")
            return 0

    @staticmethod
    def get_folders():
        """Get list of unique folders from MongoDB"""
        try:
            folders = documents_collection.distinct("folder")
            return sorted(folders)
        except Exception as e:
            st.error(f"Error getting folders: {str(e)}")
            return []

    @staticmethod
    def process_existing_folder(folder_name):
        """Process all PDFs in an existing blob folder"""
        try:
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
            
            # Get all PDFs in the folder
            blobs = container_client.list_blobs(name_starts_with=f"{folder_name}/")
            pdf_blobs = [blob for blob in blobs if blob.name.endswith('.pdf')]
            
            if not pdf_blobs:
                return {"success": False, "message": "No PDF files found in folder"}
            
            processed_files = []
            total_chunks = 0
            
            for blob in pdf_blobs:
                try:
                    # Download blob
                    blob_client = container_client.get_blob_client(blob.name)
                    pdf_data = blob_client.download_blob().readall()
                    
                    # Extract filename
                    filename = blob.name.split('/')[-1]
                    
                    # Check if already processed
                    existing_count = documents_collection.count_documents({
                        "folder": folder_name,
                        "documentName": filename
                    })
                    
                    if existing_count > 0:
                        st.info(f"📄 {filename} already processed, skipping...")
                        continue
                    
                    # Process PDF
                    pdf_file = io.BytesIO(pdf_data)
                    text, pdf_metadata = DocumentProcessor.extract_text_from_pdf(pdf_file)
                    
                    if text:
                        # Create chunks
                        chunks = DocumentProcessor.chunk_text(text)
                        
                        # Store in MongoDB
                        stored_count = MongoRAGService.store_document_chunks(
                            folder_name, filename, chunks, pdf_metadata
                        )
                        
                        processed_files.append({
                            "filename": filename,
                            "chunks": len(chunks),
                            "stored": stored_count
                        })
                        total_chunks += stored_count
                        
                        st.success(f"✅ Processed {filename}: {stored_count} chunks")
                
                except Exception as e:
                    st.error(f"❌ Error processing {blob.name}: {str(e)}")
                    continue
            
            return {
                "success": True,
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "message": f"Successfully processed {len(processed_files)} files with {total_chunks} total chunks"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error processing folder: {str(e)}"}

    @staticmethod
    def search_relevant_chunks(query, folder_filter=None, top_k=5):
        """Search for relevant chunks using vector similarity"""
        try:
            # Generate embedding for query
            query_embedding = MongoRAGService.generate_embeddings(query)
            if query_embedding is None:
                return []
            
            # Build MongoDB query
            mongo_query = {}
            if folder_filter and folder_filter != "All Folders":
                mongo_query["folder"] = folder_filter
            
            # Get all documents (or filtered by folder)
            all_docs = list(documents_collection.find(mongo_query))
            
            if not all_docs:
                return []
            
            # Calculate cosine similarity for each document
            similarities = []
            for doc in all_docs:
                if 'embedding' in doc:
                    doc_embedding = np.array(doc['embedding']).reshape(1, -1)
                    query_emb = np.array(query_embedding).reshape(1, -1)
                    
                    similarity = cosine_similarity(query_emb, doc_embedding)[0][0]
                    similarities.append((similarity, doc))
            
            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_results = similarities[:top_k]
            
            # Format results
            chunks = []
            for score, doc in top_results:
                chunks.append({
                    "text": doc["text"],
                    "document": doc["documentName"],
                    "folder": doc["folder"],
                    "chunk_index": doc["chunkIndex"],
                    "score": float(score),
                    "metadata": {
                        "author": doc.get("author", ""),
                        "title": doc.get("title", ""),
                        "total_pages": doc.get("totalPages", 0)
                    }
                })
            
            return chunks
        except Exception as e:
            st.error(f"Error searching chunks: {str(e)}")
            return []

    @staticmethod
    def generate_chat_response(query, context_chunks, session_id, chat_history=None):
        """Generate response using retrieved context and chat history"""
        try:
            # Prepare context from chunks
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                metadata_info = f"[Document: {chunk['document']}, Folder: {chunk['folder']}"
                if chunk['metadata']['title']:
                    metadata_info += f", Title: {chunk['metadata']['title']}"
                if chunk['metadata']['author']:
                    metadata_info += f", Author: {chunk['metadata']['author']}"
                metadata_info += f", Chunk: {chunk['chunk_index']}]"
                
                context_parts.append(f"Context {i}: {metadata_info}\n{chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Prepare system prompt
            system_prompt = f"""You are a helpful AI assistant that answers questions based on provided document context. 
            
Use the following context to answer the user's question. If the context doesn't contain enough information, say so politely.
When referencing information, mention the source document and folder when relevant.

Context from documents:
{context}
"""
            
            # Prepare messages with chat history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent chat history for context (last 6 messages)
            if chat_history:
                messages.extend(chat_history[-6:])
            
            messages.append({"role": "user", "content": query})
            
            # Generate response using direct API client
            assistant_response = chat_client.create_chat_completion(
                messages=messages,
                model=CHAT_MODEL,
                temperature=0.7,
                max_tokens=1000
            )
            
            if assistant_response:
                # Store conversation in MongoDB
                MongoRAGService.store_chat_message(session_id, query, assistant_response, context_chunks)
                return assistant_response
            else:
                return "I apologize, but I encountered an error while generating a response."
                
        except Exception as e:
            st.error(f"Error generating chat response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."

    @staticmethod
    def store_chat_message(session_id, user_message, assistant_response, context_chunks):
        """Store chat message in MongoDB"""
        try:
            chat_doc = {
                "_id": str(uuid.uuid4()),
                "sessionId": session_id,
                "username": st.session_state.get("username", "unknown"),
                "timestamp": datetime.now(timezone.utc),
                "userMessage": user_message,
                "assistantResponse": assistant_response,
                "sources": [
                    {
                        "document": chunk["document"],
                        "folder": chunk["folder"],
                        "chunk_index": chunk["chunk_index"],
                        "score": chunk["score"]
                    }
                    for chunk in context_chunks
                ]
            }
            
            chat_collection.insert_one(chat_doc)
        except Exception as e:
            st.error(f"Error storing chat message: {str(e)}")

    @staticmethod
    def get_chat_history(session_id, limit=10):
        """Get chat history for a session"""
        try:
            chat_docs = list(chat_collection.find(
                {"sessionId": session_id}
            ).sort("timestamp", -1).limit(limit))
            
            messages = []
            for doc in reversed(chat_docs):  # Reverse to get chronological order
                messages.append({
                    "role": "user",
                    "content": doc["userMessage"]
                })
                messages.append({
                    "role": "assistant", 
                    "content": doc["assistantResponse"]
                })
            
            return messages
        except Exception as e:
            st.error(f"Error getting chat history: {str(e)}")
            return []

def get_blob_folders_with_status():
    """Get list of folders from blob storage with processing status"""
    try:
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Get all folders from blob storage
        blob_folders = {}
        blobs = container_client.list_blobs()
        
        for blob in blobs:
            if '/' in blob.name and blob.name.endswith('.pdf'):
                folder = blob.name.split('/')[0]
                if folder not in blob_folders:
                    blob_folders[folder] = []
                blob_folders[folder].append(blob.name.split('/')[-1])
        
        # Check which folders have embeddings in MongoDB
        processed_folders = set(MongoRAGService.get_folders())
        
        # Return folder status
        folder_status = {}
        for folder, files in blob_folders.items():
            folder_status[folder] = {
                'files': files,
                'processed': folder in processed_folders,
                'file_count': len(files)
            }
        
        return folder_status
    except Exception as e:
        st.error(f"Error getting folder status: {str(e)}")
        return {}

def upload_pdf_to_folder(file, folder_name, custom_filename=None):
    """Upload PDF to specific folder in blob storage"""
    try:
        filename = custom_filename if custom_filename else file.name
        blob_name = f"{folder_name}/{filename}"
        
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        
        file.seek(0)
        blob_client.upload_blob(file.read(), overwrite=True)
        
        return blob_name
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Main UI Header with logout option
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<div class="main-header"><h1>📚 RAG Document Assistant</h1><p>SOF DR Assistant Web App</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Logout", key="logout_btn"):
        logout()

# Welcome message with current user
st.info(f"👋 Welcome, **{st.session_state.username}**! You are successfully logged in.")

# Sidebar for folder management and upload
with st.sidebar:
    st.header("📁 Document Management")
    
    # User info in sidebar
    st.markdown(f"**👤 Logged in as:** {st.session_state.username}")
    st.markdown("---")
    
    # Get available folders with status
    folder_status = get_blob_folders_with_status()
    
    if folder_status:
        st.subheader("📂 Available Folders")
        
        # Display folder status
        for folder, info in folder_status.items():
            with st.expander(f"📁 {folder} ({info['file_count']} files)"):
                if info['processed']:
                    st.success("✅ Embeddings ready - Can chat!")
                else:
                    st.warning("⚠️ Not processed - Need to create embeddings")
                    if st.button(f"🔄 Process {folder}", key=f"process_{folder}"):
                        with st.spinner(f"Processing {folder}..."):
                            result = MongoRAGService.process_existing_folder(folder)
                            if result["success"]:
                                st.success(result["message"])
                                st.rerun()
                            else:
                                st.error(result["message"])
                
                # Show files in folder
                st.write("**Files:**")
                for file in info['files']:
                    st.write(f"• {file}")
    
    # Folder selection for chat
    available_folders = ["All Folders"] + [f for f, info in folder_status.items() if info['processed']]
    
    if len(available_folders) > 1:
        selected_folder = st.radio(
            "Select Folder for Chat:",
            available_folders,
            help="Choose which folder to search in (only processed folders shown)"
        )
    else:
        st.info("📝 Upload documents or process existing folders to start chatting")
        selected_folder = "All Folders"
    
    st.markdown("---")
    
    # Upload section
    st.subheader("📤 Upload New Document")
    
    # Folder selection for upload
    upload_option = st.radio(
        "Upload to:",
        ["Existing Folder", "New Folder"]
    )
    
    if upload_option == "Existing Folder":
        existing_folders = [f for f in folder_status.keys()]
        if existing_folders:
            upload_folder = st.selectbox("Select folder:", existing_folders)
        else:
            st.warning("No existing folders found. Please create a new folder.")
            upload_folder = st.text_input("New folder name:")
    else:
        upload_folder = st.text_input("New folder name:", placeholder="e.g., contracts, reports, manuals")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=['pdf'],
        help="Upload a PDF document to process"
    )
    
    # Custom filename option
    custom_filename = st.text_input(
        "Custom filename (optional):",
        placeholder="Leave empty to use original filename"
    )
    
    if uploaded_file and upload_folder:
        if st.button("🚀 Process & Upload Document", type="primary"):
            with st.spinner("Processing document..."):
                # Extract text and metadata
                text, pdf_metadata = DocumentProcessor.extract_text_from_pdf(uploaded_file)
                
                if text:
                    # Create chunks
                    chunks = DocumentProcessor.chunk_text(text)
                    
                    # Upload to blob storage
                    filename = custom_filename if custom_filename else uploaded_file.name
                    blob_name = upload_pdf_to_folder(uploaded_file, upload_folder, filename)
                    
                    if blob_name:
                        # Store in MongoDB
                        uploaded_chunks = MongoRAGService.store_document_chunks(
                            upload_folder, filename, chunks, pdf_metadata
                        )
                        
                        st.success(f"✅ Successfully processed '{filename}'!")
                        st.info(f"📊 Created {len(chunks)} chunks, stored {uploaded_chunks} in MongoDB")
                        st.info(f"📁 Saved to folder: {upload_folder}")
                        
                        # Show document metadata
                        with st.expander("📋 Document Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Pages", pdf_metadata.get('total_pages', 'N/A'))
                                st.metric("Chunks Created", len(chunks))
                            with col2:
                                st.metric("Total Characters", len(text))
                                st.metric("Chunks Stored", uploaded_chunks)
                            
                            if pdf_metadata.get('author'):
                                st.write(f"**Author:** {pdf_metadata['author']}")
                            if pdf_metadata.get('title'):
                                st.write(f"**Title:** {pdf_metadata['title']}")

# Main chat interface
st.header("💬 Chat with Your Documents")

# Display current folder selection
if selected_folder != "All Folders":
    st.info(f"🔍 Searching in folder: **{selected_folder}**")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.write(f"**{source['document']}** (Folder: {source['folder']}, Chunk: {source['chunk_index']}, Score: {source['score']:.3f})")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Search for relevant chunks
            folder_filter = selected_folder if selected_folder != "All Folders" else None
            relevant_chunks = MongoRAGService.search_relevant_chunks(prompt, folder_filter)
            
            if relevant_chunks:
                # Get chat history for context
                chat_history = MongoRAGService.get_chat_history(st.session_state.session_id)
                
                # Generate response with context
                response = MongoRAGService.generate_chat_response(
                    prompt, relevant_chunks, st.session_state.session_id, chat_history
                )
                
                st.write(response)
                
                # Show sources
                with st.expander("📎 Sources Used"):
                    for chunk in relevant_chunks:
                        st.write(f"**{chunk['document']}** (Folder: {chunk['folder']})")
                        st.write(f"Chunk {chunk['chunk_index']} | Score: {chunk['score']:.3f}")
                        if chunk['metadata']['title']:
                            st.write(f"Title: {chunk['metadata']['title']}")
                        if chunk['metadata']['author']:
                            st.write(f"Author: {chunk['metadata']['author']}")
                        if chunk['metadata']['total_pages']:
                            st.write(f"Pages: {chunk['metadata']['total_pages']}")
                        st.write("---")
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": relevant_chunks
                })
                
                # Update chat history for context
                st.session_state.chat_history.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ])
                
                # Keep only recent history
                if len(st.session_state.chat_history) > 12:
                    st.session_state.chat_history = st.session_state.chat_history[-12:]
            else:
                response = "I couldn't find relevant information in the selected documents to answer your question."
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with information
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    processed_folders = len([f for f, info in folder_status.items() if info['processed']])
    st.metric("Processed Folders", processed_folders)
with col2:
    st.metric("Chat Messages", len(st.session_state.messages))
with col3:
    st.metric("Session ID", st.session_state.session_id[:8] + "...")
with col4:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()