# Use an official Python runtime as a parent image
FROM python:3.10-slim-bookworm
RUN apt-get update
RUN apt-get update && \
    apt-get install -y build-essential git && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libopencv-dev && \
    apt-get install -y libboost-dev && \
    apt-get install -y libopenblas-dev && \
    apt-get clean
RUN apt-get -y install gcc
RUN apt-get update && \
    apt-get install -y curl && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH=/root/.cargo/bin:$PATH
RUN apt-get -y install ninja-build
RUN mkdir -p /app
RUN mkdir -p /app/nltk_data
ENV NLTK_DATA=/app/nltk_data

RUN pip install --upgrade pip

# Install PIP Requirements
# Set the working directory to /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app
WORKDIR /app

# Expose port 8000 for the Flask app
EXPOSE 8000

# Set the environment variable for Flask
ENV FLASK_APP=run.py

#CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
