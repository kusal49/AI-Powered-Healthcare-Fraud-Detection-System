# 1. Use Python 3.11 base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy files
COPY . /app

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose the port
EXPOSE 8501

# 6. Simple command to run the app (settings are now in config.toml)
CMD ["streamlit", "run", "dashboard.py"]