FROM python:3.11-slim

# Install system dependencies for Playwright, X server, DBus, and git
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xvfb \
    xorg \
    dbus-x11 \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and download browser binaries and dependencies
RUN pip install playwright && python -m playwright install-deps && playwright install

# Copy your application code
COPY . /app

# Optionally, if you run into issues with GitPython not locating git, set the executable path explicitly:
# ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

# Expose the port your FastAPI app uses
EXPOSE 8000

# Use an entrypoint that starts Xvfb and sets required environment variables
CMD ["bash", "-c", "Xvfb :99 -screen 0 1920x1080x24 & export DISPLAY=:99; export DBUS_SESSION_BUS_ADDRESS=/dev/null; uvicorn main:app --host 0.0.0.0 --port 8000"]
