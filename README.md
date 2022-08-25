# Adaptive Coder

Transform digital data to ATCG sequences for DNA storage in high logical density,
while output sequences comply with arbitrary user-defined constraints.


## First time setup

The following steps are required in order to run Adaptive Coder:

1.  Install [Docker](https://www.docker.com/).
    *   Install
        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
        for GPU support.
    *   Setup running
        [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).
1.  Check GPUs are avaliable by running:

    ```bash
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
    ```
    
    The output of this command should show a list of your GPUs.

## Running Adaptive Coder

**The simplest way to run Adaptive Coder is using the provided Docker script.** This
was tested with 20 vCPUs, 64 GB of RAM, and a 3090 GPU.

1.  Launch the nvidia maintained container by running:

    ```bash
    docker run --gpus all -it --rm nvcr.io/nvidia/tensorflow:xx.xx-tf1-py3
    ```
   
   Where xx.xx is the container version. For example, 21.12.

1.  Install the `bert4keras` dependencies in running container, then commit it as images for later use.

    ```bash
    pip install bert4keras
    docker commit <CONTAINER ID> adaptive-coder:1.0
    ```

1.  Clone this repository to your machine and `cd` into it.

    ```bash
    git clone https://github.com/chill868686/adaptive-coder.git
    ```
    
1.  Install the `run_docker.py` dependencies. Note: You can use `Conda` or `Virtualenv` to
    create a new environment to prevent conflicts with your system's Python environment.

    ```bash
    pip3 install -r docker/requirements.txt
    ```

1.  Run `run_docker.py` pointing to a file containing digital data which you wish to transform to DNA sequences. 
    You optionally provide the path to the output directory and parameters of the coder. For example, for the
    `Francis Crick.jpg`:

    ```bash
    python3 docker/run_docker.py \
      --data_path=&&&&&T1050.fasta \
      --output_path=&&&&&T1050.fasta \
      --model_preset=&&&&&$DOWNLOAD_DIR
    ```
   
   We provide the following models:

    * **&&monomer**: This is the original model used at CASP14 with no ensembling.

    * **&&monomer\_casp14**: This is the original model used at CASP14 with
      `num_ensemble=8`, matching our CASP14 configuration. This is largely
      provided for reproducibility as it is 8x more computationally
      expensive for limited accuracy gain (+0.1 average GDT gain on CASP14
      domains).

    * **&&monomer\_ptm**: This is the original CASP14 model fine tuned with the
      pTM head, providing a pairwise confidence measure. It is slightly less
      accurate than the normal monomer model.

    * **&&multimer**: This is the [AlphaFold-Multimer](#citing-this-work) model.
      To use this model, provide a multi-sequence FASTA file. In addition, the
      UniProt database should have been downloaded.
