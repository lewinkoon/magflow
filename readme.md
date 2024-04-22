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

## Import dicom files

Create the required file directories

```shell
hemoflow init
```
The four directories correspond with the following image series:

- **FH** - Feet to head flow images.
- **AP** - Anterior to posterior flow images.
- **RL** - Right to left flow images.

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
└───RL
    ├───IM1.DCM
    ├───IM2.DCM
    ├───...
    └───IMX.DCM
```

## Run the script

Finally, build volumetric the velocity field from dicom files.

```shell
hemoflow build
```

Data files in `.csv` format will be created for each timestep in `output/` folder.

```
output/
├───data.csv.0
├───data.csv.26
├───...
└───data.csv.603
```

Each dataframe will have cordinates and velocity components for each axis.

> [!IMPORTANT]
> Velocity components are meant to be in **cm/s**.

# License

This project is licensed under the **MIT License**. See `license.txt` file for details.