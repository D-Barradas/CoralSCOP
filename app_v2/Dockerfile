# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
# Install necessary packages including OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends wget git vim curl libglib2.0-0 libglu1-mesa-dev && rm -rf /var/lib/apt/lists/*

# Install Miniforge 
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
        && bash Mambaforge-Linux-x86_64.sh -b -p /opt/conda \
        && rm Mambaforge-Linux-x86_64.sh

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH

# Copy the application files
COPY checkpoints /app/checkpoints
COPY segment_anything /app/segment_anything

# RUN conda env create -n coral -f environment.yml -y
RUN mamba create -n coral python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "coral", "/bin/bash", "-c"]

# # Install necessary packages
RUN pip install --upgrade pip --no-cache-dir && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir && \
    pip install jupyterlab opencv-python pycocotools matplotlib onnxruntime onnx plotly streamlit scikit-learn scikit-image --no-cache-dir

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Use the environment created by environment.yml
SHELL ["conda", "run", "-n", "coral", "/bin/bash", "-c"]

# The command to run the app
ENTRYPOINT ["conda", "run", "-n", "coral"]
CMD ["streamlit", "run", "app.py"]
