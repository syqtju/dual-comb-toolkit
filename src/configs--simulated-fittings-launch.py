import platform
import shutil
import subprocess
import sys
from pathlib import Path
from time import strftime

this_script_dir = Path(__file__).resolve().parent
script = str(this_script_dir / "fit-simulated-measurement.py")
python = sys.executable

timestr = strftime("%Y%m%d-%H%M%S")


def command(base_name: str, n: int) -> str:
    """
    Generate the command to launch the fitting script with the n-th configuration pack.
    """
    config_name = f"{base_name}_{n}"
    return f'"{python}" "{script}" --config {config_name}.txt --report "report-{timestr}-{config_name}"'


def select_linux_terminal() -> str:
    """
    Select a terminal emulator available on the Linux system.
    """
    terminals = ["gnome-terminal"]

    for terminal in terminals:
        if shutil.which(terminal):
            return terminal

    print(f"No supported terminal found. Install one of: {', '.join(terminals)}.")
    sys.exit(1)


def launch_command_windows(base_name: str, n: int) -> None:
    """
    Launch the fitting script in separate terminals on Windows.
    """
    for i in range(n):
        cmd = command(base_name, i + 1)
        full_cmd = f'start "Config Pack {i + 1}" {cmd}'
        subprocess.Popen(full_cmd, shell=True)
        print(f"Launched terminal {i + 1} with command: {cmd}")


def launch_command_linux(base_name: str, n: int) -> None:
    """
    Launch the fitting script in separate terminals on Linux.
    """
    terminal = select_linux_terminal()

    for i in range(n):
        cmd = command(base_name, i + 1)
        full_cmd = f"{cmd}; exec bash"
        if terminal == "gnome-terminal":
            subprocess.Popen([terminal, "--", "bash", "-c", full_cmd])
        print(f"Launched terminal {i + 1} with command: {cmd}")


DEFAULT_BASE_NAME = "config_pack"
DEFAULT_N = 1


def parse_args():
    """
    Parse command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch fitting of simulated measurements in separate terminals."
    )
    parser.add_argument(
        "-b",
        "--base-name",
        type=str,
        default=DEFAULT_BASE_NAME,
        help="Base name for the configuration packs to simulate and fit. "
        "The n-th configuration pack is constructed as `<base_name>_<n>.txt`.",
    )
    parser.add_argument(
        "-n",
        "--number-configs",
        type=int,
        default=DEFAULT_N,
        help="Number of configuration packs to simulate and fit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_name = getattr(args, "base_name", DEFAULT_BASE_NAME)
    n = getattr(args, "number_configs", DEFAULT_N)

    system = platform.system()

    if system == "Windows":
        launch_command_windows(base_name, n)
    elif system == "Linux":
        launch_command_linux(base_name, n)
    else:
        print(f"Unsupported OS: {system}")
        sys.exit(1)


if __name__ == "__main__":
    main()
