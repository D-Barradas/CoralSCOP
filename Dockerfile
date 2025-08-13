FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget vim gcc bzip2 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
#     rm /tmp/miniconda.sh

# Install Miniforge
#RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh && \
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# # Accept Anaconda terms if you are using Miniconda
# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY environment.yml /tmp/environment.yml

# Create a virtual environment named 'coralscop'
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Initialize conda and activate environment in bashrc
RUN conda init bash && \
    echo "source activate coralscop" >> /root/.bashrc

# copy the folders src and segment_anything in the container
# create a data folder and a model folder
RUN mkdir -p /app/data 
COPY src /app/src
COPY segment_anything /app/segment_anything
COPY checkpoints /app/checkpoints

WORKDIR /app

