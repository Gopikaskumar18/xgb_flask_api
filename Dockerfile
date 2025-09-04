# Use official Python image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Command to run Flask
CMD ["python", "app.py"]
