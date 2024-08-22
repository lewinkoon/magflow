# 🩸Hemoflow

![GitHub repo size](https://img.shields.io/github/repo-size/lewinkoon/hemoflow)

> Visualize **velocity** image series from a phase contrast **magnetic resonance** imaging study as a three-dimensional vector field.

## Setup

> [!IMPORTANT]  
> [Poetry](https://python-poetry.org/) dependency manager is required to run this project.

Clone the repository

```shell
git clone https://github.com/lewinkoon/hemoflow
```

Move into the project directory

```shell
cd hemoflow
```

> [!NOTE]
> If needed a custom virtual environment can be created inside project directory.
> 
> ```shell
> python -m venv .venv
> ```
> 
> Activate the previously created virtual environment.
> 
> ```shell
> .venv\Scripts\activate # on Windows
> source .venv/bin/activate # on Unix or MacOS
> ```

Install the required dependencies for the project

```shell
poetry install
```

## Usage

Load the required dicom files.

```shell
hemoflow load PATH
```

> [!IMPORTANT]
> To load multiframe `dicom` files use the option `--multiframe`
> 
> ```shell
> hemoflow load --multiframe PATH
> ```

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
hemoflow build
```

> [!TIP]
> Use the option `--parallel` to to make it faster.
>
> ```shell
> hemoflow build --parallel
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
> hemoflow build --raw
> ```

Each dataframe will have cordinates and velocity components for each axis.

> [!IMPORTANT]
> Velocity components are meant to be in **cm/s**.

# License

This project is licensed under the **GPL-2.0 License**. See `license.txt` file for details.