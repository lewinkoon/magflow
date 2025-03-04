# 🩸Magflow

![GitHub repo size](https://img.shields.io/github/repo-size/lewinkoon/magflow)

!!! info

    This project provides a Python script for converting DICOM files from 4D flow magnetic resonance imaging (MRI) studies into a three-dimensional velocity field in VTK format.

<figure markdown="span">
    ![Scheme](assets/scheme-dark.png#only-dark)
    ![Scheme](assets/scheme-light.png#only-light)
    <figcaption>DICOM to VTK workflow</figcaption>
</figure>

## Motivation

Commercial solutions for analyzing 4D flow MRI data are often prohibitively expensive, limiting accesibility for researchers, clinicians, and engineers. This script ensures that anyone can process and analyze 4D flow studies without requiring costly propietary software.

## Features

- Converts DICOM 4D flow MRI data to VTK format for visualization and analysis.
- Option to export raw velocity field data as a CSV file.
- Compatible with open-source tools like ParaView and Python-based analysis libraries.
- Simple and efficient workflow for researchers and engineers.

By providing an open-source alternative, this project enables greater accessibility and flexibility in 4D flow MRI analysis.