import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress

from magflow.utils.data import tabulate, tocsv, tovtk
from magflow.utils.dicom import extract_dicom_images, process_file
from magflow.utils.logger import logger

app = typer.Typer()


@app.command("build")
def build(
    pathname: Annotated[Path, typer.Argument()],
    output_dir: Annotated[
        str, typer.Option(help="Output directory for generated files.")
    ] = "data",
    raw: Annotated[bool, typer.Option(help="Export comma delimited values.")] = False,
):
    """Create volumetric velocity field from dicom files."""
    files = [
        f for f in pathname.iterdir() if f.is_file() and f.suffix.lower() == ".dcm"
    ]

    if not files:
        logger.error(f"No DICOM files found in {pathname}")
        raise typer.Exit(code=1)

    total_frames = 0

    # Use a context manager for the temporary directory if no specific temp_dir is provided
    with tempfile.TemporaryDirectory() as tmp_path:
        # Use the provided temp_dir or the automatically created one
        temp_dir = Path(tmp_path)

        logger.info(f"Using temporary directory: {temp_dir}")

        with Progress() as progress:
            file_task = progress.add_task(
                "[bold green]Processing DICOM files...", total=len(files)
            )

            # Process each file in the directory
            for filename in files:
                frames_processed = process_file(filename, temp_dir, progress, file_task)
                total_frames += frames_processed
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

            # Load DICOM series using the new nested directory structure
            logger.info(f"Parsing DICOM files from {temp_dir}...")
            try:
                axes_data = extract_dicom_images(temp_dir)
            except FileNotFoundError as e:
                logger.error(f"Error loading DICOM files: {e}")
                raise typer.Exit(code=1) from e

            # Extract data for easier access
            fh, rl, ap = axes_data["fh"], axes_data["rl"], axes_data["ap"]

            # Validate data consistency
            if not (fh and rl and ap):
                logger.error("One or more series are empty. Cannot proceed.")
                raise typer.Exit(code=1)

            # List unique trigger times
            timeframes = sorted({item["time"] for item in rl})
            logger.info(f"Timeframes: {timeframes}")

            # Get volume dimensions
            volume = (fh[0]["pxl"].shape[0], fh[0]["pxl"].shape[1])
            logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px)")

            # Get voxel spacing
            voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
            logger.info(
                f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
            )

            # Sequential processing with nested progress bars
            overall_task = progress.add_task(
                "[bold green]Overall Progress", total=len(timeframes)
            )

            for time in timeframes:
                # Create a subtask for this timeframe with 3 steps
                subtask = progress.add_task(f"[cyan]Timeframe {time:03d}", total=3)

                try:
                    # Filter data for this timeframe
                    progress.update(
                        subtask, description=f"[cyan]Timeframe {time:03d} (Filtering)"
                    )
                    fh_filtered = [item["val"] for item in fh if item["time"] == time]
                    rl_filtered = [item["val"] for item in rl if item["time"] == time]
                    ap_filtered = [item["val"] for item in ap if item["time"] == time]
                    # Check that filtering produced valid data
                    if not fh_filtered or not rl_filtered or not ap_filtered:
                        logger.warning(f"No data found for timeframe {time}. Skipping.")
                        success = False
                        continue
                    progress.update(subtask, advance=1)

                    # Convert to tabular format
                    progress.update(
                        subtask, description=f"[cyan]Timeframe {time:03d} (Tabulating)"
                    )
                    data = tabulate(fh_filtered, rl_filtered, ap_filtered, voxel, time)
                    progress.update(subtask, advance=1)

                    # Export to requested format
                    progress.update(
                        subtask, description=f"[cyan]Timeframe {time:03d} (Exporting)"
                    )
                    if raw:
                        tocsv(data, time, output_dir)
                    else:
                        tovtk(data, time, output_dir)

                    logger.info(f"Trigger time {time} exported with {len(data)} rows.")
                    success = True
                except Exception as e:
                    logger.error(f"Failed processing timeframe {time}: {e}")
                    success = False
                finally:
                    # Complete the subtask and advance the overall progress
                    progress.update(subtask, completed=3)
                    progress.update(overall_task, advance=1)

                    if not success:
                        logger.warning(f"Failed processing timeframe {time:03d}")

            logger.info("Sequential processing complete.")

    logger.info("Build process completed successfully.")
