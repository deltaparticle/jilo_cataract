FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (build tools for scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only specifically to save 2GB+ of image size
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models, configurations, static site, and API logic
COPY . .

# Expose port (Railway overrides this with its PORT environment variable)
EXPOSE 80

# Command to launch the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
