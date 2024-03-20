import logging

logger = logging.getLogger(__name__)

handler = logging.StreamHandler()

logger.setLevel(logging.DEBUG)


fmt = "%(message)s"
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)

logger.addHandler(handler)
