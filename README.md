[![DOI](https://zenodo.org/badge/761710489.svg)](https://zenodo.org/doi/10.5281/zenodo.10777476)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Tutorial on Applying Reinforcement Learning to the Particle Accelerator ARES

You can view the tutorial notebook as [HTML slides here](https://RL4AA.github.io/rl-tutorial-ares-basic/slides.html#/).

## Download the repository

### Get the repository with Git

You will need to have Git previously installed in your computer.
To check if you have it installed, open your terminal and type:

```bash
git --version
```

#### Git installation in macOS

```bash
brew update
brew install git
```

#### Git installation in Linux

In Ubuntu/Debian

```bash
sudo apt install git
```

In CentOS

```bash
sudo yum install git
```

## Downloading the repository

Once you have Git installed open your terminal, go to your desired directory, and type:

```bash
git clone https://github.com/RL4AA/rl-tutorial-ares-basic.git
```

Then enter the downloaded repository:

```bash
cd rl-tutorial-ares-basic
```

### Get the repository with direct download

Open your terminal, go to your desired directory, and type:

```bash
wget https://github.com/RL4AA/rl-tutorial-ares-basic/archive/refs/heads/main.zip
unzip main.zip
cd rl-tutorial-ares-basic
```

## Getting started

You need to install the dependencies before running the notebooks.

### Install `ffmpeg`

Please also run these commands to install `ffmpeg`:

- OS X: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`

### Using conda

If you don't have conda installed already and want to use conda for environment management, you can install the miniconda as [described here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

- Create a conda env from the provided env file `conda env create -f environment.yml`
- Activate the environment with `conda activate rl-icfa`
- Additional installation steps:

```bash
python -m jupyter contrib nbextension install --user
python -m jupyter nbextension enable varInspector/main
```

- **After the tutorial** you can remove your environment with `conda remove -n rl-icfa --all`

### Using venv only

If you do not have conda installed:

Alternatively, you can create the virtual env with `venv` in the standard library

```bash
python -m venv rl-icfa
```

and activate the env with $ source <venv>/bin/activate (bash) or C:> <venv>/Scripts/activate.bat (Windows)

Then, install the packages with pip within the activated environment

```bash
python -m pip install -r requirements.txt
```

Finally, install the notebook extensions if you want to see them in slide mode:

```bash
python -m jupyter contrib nbextension install --user
python -m jupyter nbextension enable varInspector/main
```

Now you should be able to run the provided notebook.

## Running the tutorial

After installing the package

You can start the jupyter notebook in the terminal, and it will start a browser automatically

```bash
python -m jupyter notebook
```

Alternatively, you can use supported Editor to run the jupyter notebooks, e.g. with VS Code.

---

## Citing the tutorial

This tutorial is uploaded to [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10777476).
Please use the following DOI when citing this code:

```bibtex
@software{xu_2024_10777477,
  author       = {Xu, Chenran and
                  Santamaria Garcia, Andrea and
                  Kaiser, Jan},
  title        = {Tutorial on Applying Reinforcement Learning to the Particle Accelerator {ARES}},
  month        = {03},
  year         = {2024},
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.10777477},
  url          = {https://doi.org/10.5281/zenodo.10777477}
}
```

---

## Acknowledgement

This tutorial is developed by [Jan Kaiser](https://github.com/jank324), [Andrea Santamaria Garcia](https://github.com/ansantam), and [Chenran Xu](https://github.com/cr-xu).

The content is based on the tutorial given at the RL4AA'23 workshop: [GitHub repository](https://github.com/RL4AA/RL4AA23)
