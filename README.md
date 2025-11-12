# RAG Document Assistant

[![Azure](https://img.shields.io/badge/Azure-Ready-blue)](https://azure.microsoft.com) [![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org) [![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)](https://streamlit.io)

> Secure AI-powered document search and chat system with Azure OpenAI and MongoDB vCore.

## 🚀 Features

- **🔐 Authentication** - Username/password login with session management
- **📄 PDF Processing** - Upload, extract text, and create embeddings
- **📁 Folder Organization** - Organize documents by categories/projects
- **🔍 Semantic Search** - Vector similarity search with GPT-4.1 chat
- **💬 Contextual Chat** - Multi-turn conversations with source citations
- **💾 Persistent Storage** - Chat history and documents in MongoDB

## 🏗️ Architecture

```
Authentication → Streamlit UI → Azure OpenAI (GPT-4.1 + Embeddings)
                      ↓
                MongoDB vCore ← Azure Blob Storage
```

## ⚙️ Setup

### Prerequisites
- Azure AI Foundry (embeddings)
- Azure AI Foundry (GPT-4.1)
- MongoDB vCore (Cosmos DB)
- Azure Blob Storage
- Azure App Service

### Environment Variables
```bash
# OpenAI Services
OPENAI_EMBEDDING_ENDPOINT=https://your-service.openai.azure.com/
OPENAI_EMBEDDING_KEY=your-embedding-key
OPENAI_CHAT_ENDPOINT=https://your-foundry.cognitiveservices.azure.com/
OPENAI_CHAT_KEY=your-chat-key

# Database & Storage
MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.com/?tls=true&authMechanism=SCRAM-SHA-256
DATABASE_NAME=RAGDatabase
BlobStorageConnection=DefaultEndpointsProtocol=https;AccountName=...
```

### Deployment
```bash
# Local Development
git clone https://github.com/nuuro-ai/sol-ai-dra-rag-system.git
cd sol-ai-dra-rag-system
pip install -r requirements.txt
streamlit run app.py

# Azure Deployment
az webapp up --name your-app --resource-group your-rg
```

## 📱 Usage

1. **Login** - Enter username/password to access system
2. **Upload PDFs** - Add documents to folders (new or existing)
3. **Process Folders** - Click "🔄 Process" to create embeddings
4. **Chat** - Select folder and ask questions about documents
5. **View Sources** - See which documents answered your questions

## 🔧 Configuration Options

### Core Settings (app.py)
```python
# Document Processing
CHUNK_SIZE = 1000              # Characters per chunk
CHUNK_OVERLAP = 200            # Overlap between chunks

# AI Models
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4.1"
```

### Search Configuration
```python
# Vector Search Parameters (customize in search_relevant_chunks)
top_k = 5                      # Number of results to retrieve
similarity_threshold = 0.7     # Minimum similarity score

# Cosine Similarity Calculation
similarity = cosine_similarity(query_emb, doc_embedding)[0][0]
```

### Authentication Settings
```python
# Session Management (customize in authentication logic)
SESSION_TIMEOUT = 3600         # 1 hour session timeout
MAX_LOGIN_ATTEMPTS = 3         # Failed login limit
```

## 🎛️ Customization Guide

### **1. Search Algorithm**
Modify `MongoRAGService.search_relevant_chunks()`:
```python
# Change similarity threshold
if similarity > 0.8:  # Increase for stricter matching

# Adjust result count
top_results = similarities[:10]  # Get more results

# Add metadata filtering
if doc.get("author") == "specific_author":
    weight_boost = 1.2
```

### **2. Document Processing**
Modify `DocumentProcessor.chunk_text()`:
```python
# Adjust chunk size for different document types
CHUNK_SIZE = 1500  # Larger chunks for technical documents
CHUNK_SIZE = 500   # Smaller chunks for dialogue/chat

# Change overlap strategy
CHUNK_OVERLAP = 300  # More overlap for better context
```

### **3. Chat Response Generation**
Modify `MongoRAGService.generate_chat_response()`:
```python
# Adjust context window
messages.extend(chat_history[-10:])  # More conversation history

# Modify temperature for creativity
temperature=0.3  # More focused responses
temperature=0.9  # More creative responses

# Change max tokens
max_tokens=1500  # Longer responses
```

### **4. Authentication**
Customize login logic:
```python
# Add role-based access
if user_role == "admin":
    allow_folder_creation = True

# Implement password complexity
min_password_length = 12
require_special_chars = True
```

## 🔀 Git Workflow

```bash
# Feature Development
git checkout dev
git checkout -b feature/your-feature
# Make changes
git commit -m "feat: description"
git push -u origin feature/your-feature

# Merge to Production
git checkout dev && git merge feature/your-feature
git checkout main && git merge dev
git push origin main
```

## 📊 Performance Notes

- **Document Processing**: ~2-3 seconds per PDF page
- **Search Speed**: <1 second for most queries
- **Concurrent Users**: Scales with Azure App Service plan
- **Storage**: MongoDB handles millions of chunks efficiently

## 🐛 Troubleshooting

### Authentication Issues
```bash
# Check credentials and session timeout
# Clear browser cache if login fails
# Verify environment variables are set
```

### PDF Processing Fails
```bash
# Ensure PDFs are text-based (not scanned)
# Check file size limits (default: 16MB)
# Verify Azure Blob Storage connection
```

### No Search Results
```bash
# Confirm documents are processed (embeddings created)
# Check folder selection and processing status
# Try different query phrasing
```

### API Errors
```bash
# Verify all environment variables
# Check Azure OpenAI service status
# Confirm model deployment names match config
```

## 🎯 Key Files

- **`app.py`** - Main Streamlit application
- **`requirements.txt`** - Python dependencies
- **`startup.sh`** - Azure App Service startup script
- **`.gitignore`** - Security and cleanup rules

## 📈 Extensibility

**Easy to add:**
- Multi-user management with roles
- Document type filters (by author, date, type)
- Export functionality (conversations, summaries)
- Advanced analytics dashboard
- REST API endpoints for integration
- Different embedding models or chat models

**Configuration points:**
- Vector database (swap MongoDB for Pinecone/Weaviate)
- Authentication provider (Azure AD, OAuth)
- File storage (different cloud providers)
- AI models (different OpenAI deployments)

---

**Built for SOL-AI-DRA Project** | **Enterprise-Ready** | **Secure & Scalable**
