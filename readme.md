# 🩸Magflow

![GitHub repo size](https://img.shields.io/github/repo-size/lewinkoon/magflow)

> Visualize **velocity** image series from a phase contrast **magnetic resonance** imaging study as a three-dimensional vector field.

## Setup

### 1. Clone the repository

```shell
git clone https://github.com/lewinkoon/magflow
```

### 2. Set up the project directory and environment

```shell
cd magflow
python -m venv .venv
# Activate virtual environment:
# On Windows:
.venv\Scripts\activate  
# On Unix or MacOS:
source .venv/bin/activate
```

### 3. Install dependencies

- Using pip:
```shell
pip install -r requirements.txt
```
- Or, if you prefer uv:
```shell
uv sync
```

## Usage

Load the required dicom files.

```shell
magflow load PATH
```

In total, three series of images should be loaded and they should correspond with each one of the CT axis directions.

- **FH** - Feet to head flow images.
- **AP** - Anterior to posterior flow images.
- **RL** - Right to left flow images.

The files directory should look similar to this.

```
files/
├───fh
│   ├───img0001.dcm
│   ├───img0002.dcm
│   ├───...
│   └───img9999.dcm
├───ap
│   ├───img0001.dcm
│   ├───img0002.dcm
│   ├───...
│   └───img9999.dcm
└───rl
    ├───img0001.dcm
    ├───img0002.dcm
    ├───...
    └───img9999.dcm
```

Finally, build volumetric the velocity field from dicom files.

```shell
magflow build
```

> [!TIP]
> Use the option `--parallel` to to make it faster.
>
> ```shell
> magflow build --parallel
> ```

Data files in `vtk` format will be created for each timestep in `output/` folder.

```
output/
├───data.vts.0
├───data.vts.26
├───...
└───data.vts.603
```

> [!TIP]
> It is possible to output the data in `.csv` format instead of `.vts`.
>
> ```shell
> magflow build --raw
> ```

Each dataframe will have cordinates and velocity components for each axis.

> [!IMPORTANT]
> Velocity components are meant to be in **cm/s**.

# License

This project is licensed under the **GPL-2.0 License**. See `license.txt` file for details.