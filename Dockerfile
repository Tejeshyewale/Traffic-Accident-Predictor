# Use a lightweight Python base image (use Google mirror to avoid Docker Hub CDN DNS issues)
FROM mirror.gcr.io/library/python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files (app.py, model, dataset, requirements if any)
COPY . /app

# Install system dependencies (for ML libraries compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# If you have requirements.txt, replace the line below with:
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask gunicorn xgboost pandas scikit-learn

# Expose the port the app will run on
EXPOSE 8080

# Start the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
