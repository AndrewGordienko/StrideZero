import sys
import time
from colorama import Fore, Style, init

# Initialize colorama for Windows support
init(autoreset=True)

# Define log levels
def log(level, msg):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{current_time} {level} {msg}")

def info(msg):
    log(f"{Fore.CYAN}[INFO]{Style.RESET_ALL}", msg)

def success(msg):
    log(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL}", msg)

def warning(msg):
    log(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}", msg)

def error(msg):
    log(f"{Fore.RED}[ERROR]{Style.RESET_ALL}", msg)

def debug(msg):
    log(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL}", msg)

# Optional for progress updates or steps
def step(step_msg):
    print(f"{Fore.BLUE}==> {Style.RESET_ALL}{step_msg}")

