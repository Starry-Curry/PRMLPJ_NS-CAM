import logging
import sys
import os
from pythonjsonlogger import jsonlogger
import colorama

def setup_logging(
    log_file_path: str = "logs/nscam.json",
    level: int = logging.INFO,
    console_output: bool = True
):
    """
    Setup structured logging with JSON output to file and colored output to console.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler (JSON)
    file_handler = logging.FileHandler(log_file_path)
    json_formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler (Colored)
    if console_output:
        colorama.init()
        console_handler = logging.StreamHandler(sys.stdout)
        
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                logging.DEBUG: colorama.Fore.CYAN,
                logging.INFO: colorama.Fore.GREEN,
                logging.WARNING: colorama.Fore.YELLOW,
                logging.ERROR: colorama.Fore.RED,
                logging.CRITICAL: colorama.Fore.RED + colorama.Style.BRIGHT,
            }

            def format(self, record):
                color = self.COLORS.get(record.levelno, colorama.Fore.WHITE)
                reset = colorama.Style.RESET_ALL
                return f"{color}[{record.levelname}] {record.name}: {record.msg}{reset}"

        console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)

    return root_logger

def get_logger(name: str):
    return logging.getLogger(name)
