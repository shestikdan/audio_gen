FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p lections_text_base lections_text_mistral lections_audio samples

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV COQUI_TOS_AGREED=1

# Skip model download during build
ENV SKIP_XTTS_DOWNLOAD=1

# Modify setup_xtts.py to bypass memory check and skip model download
RUN sed -i 's/response = input("Continue anyway? (y\/n): ")/response = "y"/' scripts/setup_xtts.py

# Run setup script with modified behavior
RUN python scripts/setup_xtts.py

# Default command
CMD ["python", "scripts/lection_to_audio.py"] 