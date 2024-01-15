# Paraflow

> This script aims to visualize velocity image series from a phase contrast mri study as a three-dimensional vector field.

## Usage

Install the required dependencies from the `requirements.txt` file.

```bash
python -m pip install -r requirements.txt
```

Run the script

```bash
python main.py
```

## Setup

In case some dependencies need to be updated it's 

```bash
python -m pip install pur
```

To update the `requirements.txt` file just run

```bash
pur -r requirements.txt
```

From here it's still needed to actually install the packages

```bash
python -m pip install -r requirements.txt
```

## File structure

For the script to work properly an structured `files/` directory must be arranged.

```bash
files
├───AP
│   ├───IM1.DCM
│   ├───IM2.DCM
│   └───....DCM
├───FH
│   ├───IM1.DCM
│   ├───IM2.DCM
│   └───....DCM
├───M
│   ├───IM1.DCM
│   ├───IM2.DCM
│   └───....DCM
└───RL
    ├───IM1.DCM
    ├───IM2.DCM
    └───....DCM
```