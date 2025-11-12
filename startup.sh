#!/bin/bash

# startup.sh - Azure App Service startup script for Streamlit with MongoDB
echo "Starting Streamlit RAG Application with MongoDB vCore..."

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start Streamlit app
echo "Starting Streamlit server..."
streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false