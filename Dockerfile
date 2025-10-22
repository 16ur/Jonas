FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/

# Copy data and models directories
COPY data/ ./data/
COPY models/ ./models/

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]