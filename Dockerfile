# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the project files into the container
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
# --no-root: Do not install the project itself, only dependencies
# --no-dev: Do not install development dependencies
RUN poetry config virtualenvs.create true && \
    poetry install --no-root --no-dev

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["poetry", "run", "python", "app.py"]