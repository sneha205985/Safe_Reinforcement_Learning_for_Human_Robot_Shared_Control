"""
Setup script for Safe RL Human-Robot Shared Control package.

This package implements Constrained Policy Optimization (CPO) methods
for safe reinforcement learning in human-robot collaborative systems.
"""

from setuptools import setup, find_packages
import os
import re


def read(fname):
    """Read file contents."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version(*file_paths):
    """Find version string in source files."""
    version_file = read(os.path.join(*file_paths))
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            lines = f.readlines()
        # Filter out comments and empty lines
        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-f'):
                # Handle conditional dependencies
                if 'torch>=1.9.0+cu111' in line or 'torchvision>=0.10.0+cu111' in line:
                    continue  # Skip CUDA-specific versions
                requirements.append(line)
        return requirements
    return []


# Core requirements (essential for basic functionality)
CORE_REQUIREMENTS = [
    "torch>=1.9.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "gym>=0.21.0",
    "pyyaml>=6.0",
    "matplotlib>=3.5.0",
    "tqdm>=4.62.0"
]

# Development requirements (for testing, linting, documentation)
DEV_REQUIREMENTS = [
    "pytest>=6.2.0",
    "pytest-cov>=3.0.0",
    "pytest-mock>=3.6.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "isort>=5.10.0",
    "mypy>=0.910",
    "sphinx>=4.3.0",
    "sphinx-rtd-theme>=1.0.0"
]

# Experiment tracking requirements
TRACKING_REQUIREMENTS = [
    "tensorboard>=2.8.0",
    "tensorboardX>=2.4",
    "wandb>=0.12.0"
]

# Robotics simulation requirements
ROBOTICS_REQUIREMENTS = [
    "pybullet>=3.2.0",
    "stable-baselines3>=1.5.0"
]

# Full requirements (everything)
FULL_REQUIREMENTS = read_requirements()

setup(
    name="safe-rl-human-robot",
    version=find_version("src", "__init__.py"),
    author="Safe RL Team",
    author_email="team@saferlhumanrobot.ai",
    description="Safe Reinforcement Learning for Human-Robot Shared Control using CPO",
    long_description=read("README.md") if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/safe-rl-team/safe-rl-human-robot",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "tracking": TRACKING_REQUIREMENTS,
        "robotics": ROBOTICS_REQUIREMENTS,
        "full": FULL_REQUIREMENTS
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    keywords="reinforcement-learning safety robotics human-robot-interaction constrained-optimization",
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "safe-rl-train=safe_rl_human_robot.scripts.train:main",
            "safe-rl-eval=safe_rl_human_robot.scripts.evaluate:main",
            "safe-rl-demo=safe_rl_human_robot.scripts.demo:main"
        ]
    },
    
    # Package data
    package_data={
        "safe_rl_human_robot": [
            "config/*.yaml",
            "config/*.json"
        ]
    },
    
    # Test configuration
    test_suite="tests",
    tests_require=DEV_REQUIREMENTS,
    
    # Project URLs
    project_urls={
        "Documentation": "https://safe-rl-human-robot.readthedocs.io/",
        "Source": "https://github.com/safe-rl-team/safe-rl-human-robot",
        "Bug Reports": "https://github.com/safe-rl-team/safe-rl-human-robot/issues",
        "Funding": "https://github.com/sponsors/safe-rl-team",
        "Say Thanks!": "https://saythanks.io/to/safe-rl-team"
    },
    
    # Zip safety
    zip_safe=False,
    
    # License
    license="MIT",
    
    # Additional metadata
    platforms=["any"],
    maintainer="Safe RL Team",
    maintainer_email="maintainers@saferlhumanrobot.ai"
)


# Post-installation message
def post_install_message():
    """Display post-installation message."""
    print("\n" + "="*60)
    print("Safe RL Human-Robot Shared Control Installation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run tests: pytest tests/")
    print("2. Check examples: python examples/basic_usage.py")
    print("3. Read documentation: https://safe-rl-human-robot.readthedocs.io/")
    print("\nFor development installation:")
    print("pip install -e .[dev,tracking,robotics]")
    print("\nFor questions and issues:")
    print("https://github.com/safe-rl-team/safe-rl-human-robot/issues")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run setup
    setup()
    
    # Display post-installation message
    import sys
    if "install" in sys.argv:
        post_install_message()