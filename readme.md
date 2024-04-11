# 🩸Hemoflow

![GitHub repo size](https://img.shields.io/github/repo-size/lewinkoon/hemoflow)

> Visualize **velocity** image series from a phase contrast **magnetic resonance** imaging study as a three-dimensional vector field.

## Setup

Clone the repository

```shell
git clone https://github.com/lewinkoon/hemoflow
```

Change into the project directory

```shell
cd hemoflow
```

Create a virtual environment inside the directory

```shell
python -m venv .venv
```

Activate the previously created virtual environment

```shell
.venv\Scripts\activate # on Windows
source .venv/bin/activate # on Unix or MacOS
```

Install `poetry` dependency manager

```shell
python -m pip install poetry
```

Install the required dependencies for the project

```shell
poetry install
```

## Import the required dicom files

Create the required file directories

```shell
mkdir -p files/{M,FH,AP,RL}
```
The four directories correspond with the following image series:

- **FH** - Feet to head flow images.
- **AP** - Anterior to posterior flow images.
- **RL** - Right to left flow images.
- **M (optional)** - Segmentation images to apply volume masking.

Copy your `dicom` image series to `files/` directory.

```
files/
├───FH
│   ├───IM1.DCM
│   ├───IM2.DCM
│   ├───...
│   └───IMX.DCM
├───AP
│   ├───IM1.DCM
│   ├───IM2.DCM
│   ├───...
│   └───IMX.DCM
├───RL
│   ├───IM1.DCM
│   ├───IM2.DCM
│   ├───...
│   └───IMX.DCM
└───M (optional)
    ├───IM1.DCM
    ├───IM2.DCM
    ├───...
    └───IMX.DCM
```

## Run the package

Finally, run the script

```shell
poetry run hemoflow
```

Data files in `.csv` format will be created for each timestep in `output/` folder.

```
output/
├───data.csv.0
├───data.csv.26
├───...
└───data.csv.603
```

The example table below illustrates how data should look like. Velocities are supposed to be in *cm/s*.

| x   | y   | z   | vx    | vy    | vz    |
| --- | --- | --- | ----- | ----- | ----- |
| 0   | 0   | 0   | 26.54 | -1.54 | 62.14 |
| ... | ... | ... | ...   | ...   | ...   |


# License

This project is licensed under the **MIT License**. See `license.txt` file for details.