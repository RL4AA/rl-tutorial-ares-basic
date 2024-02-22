# Tutorial on Applying Reinforcement Learning to the Particle Accelerator ARES

## Getting started

- First, download the material to your local disk by cloning the repository using
  - https: `git clone https://github.com/RL4AA/rl-tutorial-ares-basic.git`
  - ssh: `git clone git@github.com:RL4AA/rl-tutorial-ares-basic.git`
- If you don't have git installed, you can click on the green button that says "Code", and choose to download it as a `.zip` file.

### Install `ffmpeg`

- OS X: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Ubuntu 14.04: `sudo apt-get install libav-tools`
- With pip: `pip install imageio-ffmpeg`

### Setup the environment locally

- Open terminal app
- (Suggested) Create a virtual environment using `conda` or `venv`.

#### Using venv

```bash
python3 -m venv rl-tutorial
source rl-tutorial/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

- Open the tutorial notebook `tutorial.ipynb` in the jupyter server in browser
- When you are done type `deactivate`

#### Using conda only

Instructions to install conda [here](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/install/index.html)

```bash
conda env create -f environment.yml
conda activate rl-tutorial
jupyter notebook
```

- Open the tutorial notebook `tutorial.ipynb` in the jupyter server in browser
- When you are done type `conda deactivate` to deactivate the virtual environment

#### Using conda + pip

```bash
cd path_to_your_folder/rl-tutorial-ares-basic
```

```bash
conda create -n rl-tutorial python=3.10
conda activate rl-tutorial
pip3 install -r requirements.txt
jupyter notebook
```

- Open the tutorial notebook `tutorial.ipynb` in the jupyter server in browser
- When you are done type `conda deactivate` to deactivate the virtual environment

---

## Acknowledgement

This tutorial is developed by [Jan Kaiser](https://github.com/jank324), [Andrea Santamaria Garcia](https://github.com/ansantam), and [Chenran Xu](https://github.com/cr-xu).

The content is based on the tutorial given at the RL4AA'23 workshop: [GitHub repository](https://github.com/RL4AA/RL4AA23)
