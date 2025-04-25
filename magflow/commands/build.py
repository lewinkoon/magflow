import typer
from typing_extensions import Annotated
import os

from magflow.utils.logger import logger
from magflow.utils.dicom import parse_dicom_files
from magflow.utils.data import tabulate, tovtk, tocsv
from rich.progress import Progress

app = typer.Typer()


@app.command("build")
def build(
    raw: Annotated[bool, typer.Option(help="Export comma delimited values.")] = False,
    output_dir: Annotated[
        str, typer.Option(help="Output directory for generated files.")
    ] = "output",
    input_dir: Annotated[
        str, typer.Option(help="Input directory containing DICOM files.")
    ] = ".tmp",
):
    """Create volumetric velocity field from dicom files."""
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Load DICOM series using the new nested directory structure
        logger.info(f"Parsing DICOM files from {input_dir}...")
        try:
            axes_data = parse_dicom_files(input_dir)
        except FileNotFoundError as e:
            logger.error(f"Error loading DICOM files: {e}")
            raise typer.Exit(code=1)

        # Extract data for easier access
        fh, rl, ap = axes_data["fh"], axes_data["rl"], axes_data["ap"]

        # Validate data consistency
        if not (fh and rl and ap):
            logger.error("One or more series are empty. Cannot proceed.")
            raise typer.Exit(code=1)

        # List unique trigger times
        timeframes = sorted(set(item["time"] for item in rl))
        logger.info(f"Timeframes: {timeframes}")

        # Get volume dimensions
        volume = (fh[0]["pxl"].shape[0], fh[0]["pxl"].shape[1])
        logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px)")

        # Get voxel spacing
        voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
        logger.info(
            f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
        )

        logger.info(
            "Parallel processing has been disabled. Using sequential processing."
        )

        # Sequential processing with nested progress bars
        with Progress() as progress:
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

    except Exception as e:
        logger.error(f"Build process failed: {e}")
        raise typer.Exit(code=1)

    logger.info("Build process completed successfully.")
