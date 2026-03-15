import argparse
import ast
import contextlib
import logging
import os
import platform
import sys
import warnings

from src.utils import load_toml

CONFIG_FILE = "config.toml"


def configure_startup_logging():
    """Reduce noisy third-party startup logs without changing app behavior."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GLOG_logtostderr", "0")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message="dropout option adds dropout after all but last recurrent layer.*",
        category=UserWarning,
        module=r"torch\.nn\.modules\.rnn",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
        category=FutureWarning,
        module=r"torch\.nn\.utils\.weight_norm",
    )


def get_current_config(config_file=CONFIG_FILE):
    config = load_toml(config_file)
    current_os = platform.system().lower()

    if current_os == "darwin":
        return config["config_macos"]
    if current_os == "linux":
        return config["config_linux"]
    return config["config_common"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen",
        nargs="*",
        help=(
            "Generate pose images with landmarks. "
            "Use --gen for default poses, or pass poses (e.g. --gen mountain)."
        ),
    )
    return parser.parse_args()


def normalize_gen_poses(gen_values):
    if gen_values is None:
        return None
    if not gen_values:
        return []

    if len(gen_values) == 1:
        candidate = gen_values[0].strip()
        if candidate.startswith("[") and candidate.endswith("]"):
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                pass

    return gen_values


@contextlib.contextmanager
def silence_native_output():
    """Temporarily redirect OS stdout/stderr to suppress native C/C++ library logs."""
    with open(os.devnull, "w") as devnull:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()

        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
