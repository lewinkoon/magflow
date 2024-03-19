# 🩸Hemoflow

![GitHub repo size](https://img.shields.io/github/repo-size/lewinkoon/hemoflow)

> This script aims to visualize velocity image series from a phase contrast mri study as a three-dimensional vector field.

## Quickstart

Clone the repository

```bash
git clone https://github.com/lewinkoon/hemoflow
```

Change into the project directory

```bash
cd hemoflow
```

Create a virtual environment inside the directory

```bash
python -m venv .venv
```

Activate the previously created virtual environment

```bash
.venv\Scripts\activate # on Windows
source .venv/bin/activate # on Unix or MacOS
```

Install the required dependencies from the `requirements.txt` file.

```bash
python -m pip install -r requirements.txt
```

Create the required file directories

```bash
mkdir -p files/{MK,FH,AP,RL}
```
The four directories correspond with the following image series:

- **MK** - Segmentation images to applu volume masking.
- **FH** - Feet to head flow images.
- **AP** - Anterior to posterior flow images.
- **RL** - Right to left flow images.

Copy your `dicom` image series to `files/` directory.

```
files/
├───MK
│   ├───IM1.DCM
│   ├───IM2.DCM
│   ├───...
│   └───IMX.DCM
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

Finally, run the script

```bash
python main.py
```

Data files in `.csv` format will be created for each timestep in `output/` folder.

```
output/
├───data.csv.0
├───data.csv.26
├───...
└───data.csv.603
```

The example table below illustrates how data should look like. Velocities are supposed to be in cm/s.

| x   | y   | z   | vx    | vy    | vz    |
| --- | --- | --- | ----- | ----- | ----- |
| 0   | 0   | 0   | 26.54 | -1.54 | 62.14 |
| ... | ... | ... | ...   | ...   | ...   |


# License

This project is licensed under the **MIT License**. See `license.txt` file for details.