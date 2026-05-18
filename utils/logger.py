"""Logging utilities."""
import os
import sys
import time


def setup_logger(log_dir: str, name: str = "transgap"):
    """Create a simple logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, 'a')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    print(f"Logging to: {log_file}")
    return log_file
