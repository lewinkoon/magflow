import logging
from rich.logging import RichHandler


logger = logging.getLogger(__name__)

# handler = logging.StreamHandler()
handler = RichHandler()

logger.setLevel(logging.WARNING)

fmt = "%(message)s"
formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
handler.setFormatter(formatter)

logger.addHandler(handler)
