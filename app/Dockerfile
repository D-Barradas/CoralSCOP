# Start from the latest Ubuntu image
FROM ubuntu:latest

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages including OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends wget git vim curl libglib2.0-0 libglu1-mesa-dev && rm -rf /var/lib/apt/lists/*

# Install Miniforge 
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
        && bash Mambaforge-Linux-x86_64.sh -b -p /opt/conda \
        && rm Mambaforge-Linux-x86_64.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
# COPY environment.yml environment.yml
# Create the Conda environment
# RUN conda env create -n coral -f environment.yml -y
RUN mamba create -n coral python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "coral", "/bin/bash", "-c"]


# # Install necessary packages
RUN pip install --upgrade pip --no-cache-dir && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir && \
    pip install jupyterlab opencv-python pycocotools matplotlib onnxruntime onnx plotly streamlit scikit-learn scikit-image streamlit-extras easyocr seaborn --no-cache-dir

# RUN  python -m pip install git+https://github.com/facebookresearch/segment-anything.git

# Test if PyTorch can access CUDA
# RUN python -c "import torch; print(torch.cuda.is_available())"

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Use the environment created by environment.yml
SHELL ["conda", "run", "-n", "coral", "/bin/bash", "-c"]

# The command to run the app
ENTRYPOINT ["conda", "run", "-n", "coral"]
CMD ["streamlit", "run", "streamlit_starting_page.py"]
