
# Use a lightweight Python image
FROM python:3.13-slim

# Set application directory inside container
WORKDIR /app

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python logs immediately
ENV PYTHONUNBUFFERED=1

# Copy Python dependencies first
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY src/ ./src/
COPY vector_store/ ./vector_store/
COPY knowledge_graph/ ./knowledge_graph/

# Streamlit application port
EXPOSE 8501

# Start Streamlit
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
