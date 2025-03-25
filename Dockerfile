FROM python:3.11-slim

# Install system dependencies required by Playwright
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

# Install Playwright and then download the browser binaries
RUN pip install playwright && python -m playwright install-deps
RUN playwright install

# Copy your application code
COPY . /app

# Unset DISPLAY to avoid X server issues (if necessary)
ENV DISPLAY=

# Expose the port your FastAPI app uses (e.g., 8000)
EXPOSE 8000

# Start your FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
