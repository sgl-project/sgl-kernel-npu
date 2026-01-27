import os
import re
import subprocess
import sys

import setuptools
from setuptools import find_packages
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

# Eliminate timestamp differences in whl compressed packages
os.environ["SOURCE_DATE_EPOCH"] = "0"

current_version = os.getenv("VERSION", "1.0.0")


class CustomBuildPy(build_py):
    def run(self):
        logging_type = (
            "DEBUG" if os.environ.get("DEBUG_MODE", "OFF") == "ON" else "INFO"
        )
        config_content = (
            "import logging\nlogging.basicConfig(level=logging.%s)" % logging_type
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, "deep_ep", "build_config.py")

        with open(config_file, "w") as f:
            f.write(config_content)

        super().run()


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


def get_git_revision():
    """
    Get the short (8 characters) hash value of the current Git repository
    Returns:
        str: A string with a '+' prefix and an 8-character hash value, or an empty string if retrieval fails
    """
    try:
        cmd = ["git", "rev-parse", "--short=8", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).strip().decode("utf-8")
    except Exception:
        revision = ""
    return revision


def get_cann_version():
    """
    Get the CANN version information of the current environment
    Returns:
        str: CANN version string, format like 'cann.8.2.rc1.b231'
             Returns an empty string if retrieval fails
    """
    try:
        ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", "")
        if not ascend_home:
            return ""

        arch = subprocess.check_output(["uname", "-m"]).decode().strip()
        arch = arch.lower()

        info_file = os.path.join(
            ascend_home, f"{arch}-linux", "ascend_toolkit_install.info"
        )
        if not os.path.exists(info_file):
            return ""

        with open(info_file, "r") as f:
            lines = f.readlines()

        version = ""
        b_version = ""
        for line in lines:
            line = line.strip()
            if line.startswith("version="):
                version = line.split("=")[1]
            elif line.startswith("innerversion="):
                match = re.search(r"[Bb](\d+)", line)
                if match:
                    b_version = match.group(1)

        if version and b_version:
            version = version.lower()
            b_version = b_version.lower()
            return f".cann.{version}.b{b_version}"
        return ""

    except Exception:
        return ""


def get_npu_chip_type():
    """
    Detect the NPU chip type (910B or 910C) using npu-smi command
    Returns:
        str: "910b" if running on Ascend 910B, "910c" if running on Ascend 910C, empty string otherwise
    """
    try:
        # Execute npu-smi command to get board information
        cmd = ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        output = result.stdout

        # Check for 910C pattern: NPU Name : 9382
        if re.search(r"NPU\s+Name\s*:\s*9382", output, re.IGNORECASE) and re.search(
            r"Chip\s+Name\s*:\s*Ascend910", output, re.IGNORECASE
        ):
            return "910c"

        # Check for 910B pattern: Chip Name : 910B
        if re.search(r"Chip\s+Name\s*:\s*910B", output, re.IGNORECASE):
            return "910b"

        # Fallback: check for keywords in the output
        output_lower = output.lower()
        if "ascend910" in output_lower:
            return "910C"
        elif "910b" in output_lower:
            return "910b"

        print(
            f"Warning: Could not determine NPU chip type from output:\n{output}",
            file=sys.stderr,
        )
        return ""

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if isinstance(e, FileNotFoundError):
            print(
                "Warning: npu-smi command not found. Cannot determine NPU chip type.",
                file=sys.stderr,
            )
        else:
            print(f"Warning: Failed to execute npu-smi command: {e}", file=sys.stderr)
            print(f"Command output: {e.output}", file=sys.stderr)
        return ""
    except Exception as e:
        print(
            f"Warning: Unexpected error detecting NPU chip type: {e}", file=sys.stderr
        )
        return ""


git_rev = get_git_revision()
cann_ver = get_cann_version()
chip_type = get_npu_chip_type()

# Build chip suffix based on detected chip type
chip_suffix = f"_{chip_type}" if chip_type else ""

# Construct version suffix with git revision, CANN version, and chip suffix
version_suffix = (
    f"{git_rev}{cann_ver}{chip_suffix}"
    if git_rev
    else f"+{cann_ver.lstrip('.')}{chip_suffix}" if cann_ver else ""
)

setuptools.setup(
    name="deep_ep",
    version=current_version + version_suffix,
    author="",
    author_email="",
    description="python api for deep_ep",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=["torch"],
    python_requires=">=3.7",
    package_data={"deep_ep": ["deep_ep_cpp.cpython*.so", "vendors", "vendors/**"]},
    distclass=BinaryDistribution,
    cmdclass={
        "build_py": CustomBuildPy,
    },
)
