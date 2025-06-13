# 🩸 Magflow

![GitHub repo size](https://img.shields.io/github/repo-size/lewinkoon/magflow)

> Convert DICOM files from 4D flow magnetic resonance imaging (MRI) studies into a three-dimensional velocity field in VTK format.

## Overview

Magflow is a Python tool designed to process 4D flow MRI data, converting DICOM files into VTK format for visualization and analysis of blood flow dynamics. This tool is particularly useful for cardiovascular research and clinical applications requiring detailed flow field analysis.

## Features

- **DICOM to VTK Conversion**: Seamlessly convert 4D flow MRI DICOM files to VTK format
- **3D Velocity Field Generation**: Create comprehensive three-dimensional velocity fields
- **Visualization Tools**: Generate flow visualizations for analysis

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Make sure you have uv installed:

```bash
pip install uv
```

Clone the repository and install dependencies:

```bash
git clone https://github.com/lewinkoon/magflow.git
cd magflow
uv sync
```

## Usage

### Command Line Interface

Run Magflow commands using the just task runner:

```bash
# Run magflow with specific command
just run <COMMAND>

# Generate visualizations
just run visualize
```

### Development

Update dependencies:

```bash
just update
```

Build visualizations:

```bash
just build
```

## Requirements

- Python 3.8+
- uv package manager
- DICOM files from 4D flow MRI studies

## Support

For questions, issues, or feature requests, please open an issue on GitHub.