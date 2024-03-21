import logging

logger = logging.getLogger(__name__)

handler = logging.StreamHandler()

logger.setLevel(logging.DEBUG)


fmt = "%(asctime)s - %(message)s"
formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
handler.setFormatter(formatter)

logger.addHandler(handler)
