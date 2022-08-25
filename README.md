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

1.  Clone this repository and `cd` into it.

    ```bash
    git clone https://github.com/chill868686/adaptive-coder.git
    ```

1.  Build the Docker image:

    ```bash
    docker build -f docker/Dockerfile -t adacoder .
    ```
    
1.  Install the `run_docker.py` dependencies. Note: You may optionally wish to
    create a
    [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
    to prevent conflicts with your system's Python environment.

    ```bash
    pip3 install -r docker/requirements.txt
    ```

1.  Make sure that the output directory exists (the default is `/tmp/alphafold`)
    and that you have sufficient permissions to write into it. You can make sure
    that is the case by manually running `mkdir /tmp/alphafold` and
    `chmod 770 /tmp/alphafold`.

1.  Run `run_docker.py` pointing to a FASTA file containing the protein
    sequence(s) for which you wish to predict the structure. If you are
    predicting the structure of a protein that is already in PDB and you wish to
    avoid using it as a template, then `max_template_date` must be set to be
    before the release date of the structure. You must also provide the path to
    the directory containing the downloaded databases. For example, for the
    T1050 CASP14 target:

    ```bash
    python3 docker/run_docker.py \
      --fasta_paths=T1050.fasta \
      --max_template_date=2020-05-14 \
      --data_dir=$DOWNLOAD_DIR
    ```

    By default, Alphafold will attempt to use all visible GPU devices. To use a
    subset, specify a comma-separated list of GPU UUID(s) or index(es) using the
    `--gpu_devices` flag. See
    [GPU enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration)
    for more details.

1.  You can control which AlphaFold model to run by adding the
    `--model_preset=` flag. We provide the following models:

    * **monomer**: This is the original model used at CASP14 with no ensembling.

    * **monomer\_casp14**: This is the original model used at CASP14 with
      `num_ensemble=8`, matching our CASP14 configuration. This is largely
      provided for reproducibility as it is 8x more computationally
      expensive for limited accuracy gain (+0.1 average GDT gain on CASP14
      domains).

    * **monomer\_ptm**: This is the original CASP14 model fine tuned with the
      pTM head, providing a pairwise confidence measure. It is slightly less
      accurate than the normal monomer model.

    * **multimer**: This is the [AlphaFold-Multimer](#citing-this-work) model.
      To use this model, provide a multi-sequence FASTA file. In addition, the
      UniProt database should have been downloaded.

1.  You can control MSA speed/quality tradeoff by adding
    `--db_preset=reduced_dbs` or `--db_preset=full_dbs` to the run command. We
    provide the following presets:

    *   **reduced\_dbs**: This preset is optimized for speed and lower hardware
        requirements. It runs with a reduced version of the BFD database.
        It requires 8 CPU cores (vCPUs), 8 GB of RAM, and 600 GB of disk space.

    *   **full\_dbs**: This runs with all genetic databases used at CASP14.

    Running the command above with the `monomer` model preset and the
    `reduced_dbs` data preset would look like this:

    ```bash
    python3 docker/run_docker.py \
      --fasta_paths=T1050.fasta \
      --max_template_date=2020-05-14 \
      --model_preset=monomer \
      --db_preset=reduced_dbs \
      --data_dir=$DOWNLOAD_DIR
    ```
