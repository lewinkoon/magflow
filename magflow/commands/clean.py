import os

import typer
from magflow.logger import logger

app = typer.Typer()


@app.command("clean")
def clean():
    """
    Remove exported data.
    """
    path = "output"

    # check first if path exists
    if not os.path.exists(path):
        logger.info("Output files not exported yet.")
    else:
        try:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed {file_path}")
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                        logger.info(f"Removed directory {file_path}")
                except Exception as e:
                    logger.error(f"Error while deleting {file_path}: {e}")
            os.rmdir(path)
            logger.info(f"Removed {path}")
        except Exception as e:
            logger.error(f"Output directory cannot be removed: {e}")
