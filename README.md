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

1. Install the `bert4keras` dependencies in running container, then commit it as a new image for later use.

    ```bash
    pip install bert4keras
    docker commit <CONTAINER ID> adaptive-coder:1.0
    ```

2. Clone this repository to your machine and `cd` into it.

    ```bash
    git clone https://github.com/chill868686/adaptive-coder.git
    ```
    
3. Install the `run_docker.py` dependencies. Note: You can 
    create a new environment by `Conda` or `Virtualenv` to prevent conflicts with your system's Python environment.

    ```bash
    pip3 install -r docker/requirements.txt
    ```

4. Run `run_docker.py` pointing to a file containing digital data or DNA sequences which you wish to transform. 
    You optionally provide parameters to command coding:
    ```bash
       python docker/run_docker.py --file_path=(file_path) [OPTIONS]
       OPTIONS(defaluts):
         --log=running.log \
         --model=best_model.weights \
         --docker_image_name=adaptive-coder:1.0 \
         --coding_type=en_decoding|encoding|decoding|training
    ```
   
      We provide the following pattern:
   1. DNA encoding&decoding:
   ```bash
    python docker/run_docker.py --file_path=mutimedias/poetry.txt
    ```
   2. DNA encoding:
   ```bash
    python docker/run_docker.py --file_path=mutimedias/poetry.txt --coding_type=encoding
    ```
   3. DNA decoding:
   ```bash
    python docker/run_docker.py --file_path=results/encodes/poetry.txt.dna --coding_type=decoding
    ```
   4. model training:
   ```bash
    python docker/run_docker.py --file_path=datasets/seq_good_256_m.txt --coding_type=training
    ```