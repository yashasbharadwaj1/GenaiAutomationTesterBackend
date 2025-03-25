FROM python:3.11-slim

# Install system dependencies for FastAPI and Playwright
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xvfb \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and download browsers
RUN pip install playwright && python -m playwright install-deps

# Copy your application code
COPY . /app

# Unset DISPLAY to avoid X server issues
ENV DISPLAY=

# Expose the port (adjust if needed)
EXPOSE 8000

# Start your FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
