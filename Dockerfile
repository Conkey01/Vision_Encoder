# Use an official Python runtime
FROM python:3.9-slim

# HF Spaces requires a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy and install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY --chown=user . .

# Hugging Face routes traffic to port 7860
EXPOSE 7860

# Run the app on port 7860
CMD["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]