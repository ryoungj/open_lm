import logging


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket

        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f"%(asctime)s |  {hostname} | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d,%H:%M:%S",
        )
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S")
    
    # Remove all existing handlers from the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
