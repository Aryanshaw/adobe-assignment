import logging
import sys

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m"  # Bold Red
    }
    RESET = "\033[0m"
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def format(self, record):
        log_fmt = self.COLORS.get(record.levelno, "") + self.FORMAT + self.RESET
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()
