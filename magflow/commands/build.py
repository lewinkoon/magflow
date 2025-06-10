import tempfile
from pathlib import Path
from typing import Annotated

import typer

import magflow.utils.dicom as dcm
from magflow.utils.logger import logger

app = typer.Typer()


@app.command("build")
def build(
    pathname: Annotated[Path, typer.Argument()],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o", "--output-dir", help="Output directory for generated files."
        ),
    ] = Path("data"),
    raw: Annotated[bool, typer.Option(help="Export comma delimited values.")] = False,
):
    """Create volumetric velocity field from dicom files."""

    # Always search recursively for DICOM files
    files = dcm.find_dicom_files(pathname, recursive=True)

    with tempfile.TemporaryDirectory() as tmp_path:
        temp_dir = Path(tmp_path)
        logger.info(f"Using temporary directory: {temp_dir}")

        # Process DICOM files
        dcm.batch_process_dicoms(files, temp_dir)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        # Load and validate DICOM series
        logger.info(f"Parsing DICOM files from {temp_dir}...")
        try:
            axes_data = dcm.parse_dicom_directory(temp_dir)
        except FileNotFoundError as e:
            logger.error(f"Error loading DICOM files: {e}")
            raise typer.Exit(code=1) from e

        # Validate and extract data
        fh, rl, ap = dcm.validate_axis_data(axes_data)
        volume, voxel = dcm.get_volume_metadata(fh)

        # Get unique timeframes
        timeframes = sorted({item["time"] for item in rl})
        logger.info(f"Timeframes: {timeframes}")

        # Process all timeframes
        dcm.export_all_timeframes(timeframes, fh, rl, ap, voxel, output_dir, raw)

    logger.info("Build process completed successfully.")
