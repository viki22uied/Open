FROM python:3.11-slim

LABEL maintainer="OpenEnv Contributors"
LABEL description="Invoice Review OpenEnv - AI Agent Benchmark Environment"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (7860 is the HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
