#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "torch",
    "torch_geometric",
    "lightning",
    "kaggle",
    "polars",
    "numpy",
    "huggingface_hub",
    "seaborn",
    "networkx",
    "optuna",
    "wandb>=0.12.1",
    "optuna-integration",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Marcel Boersma",
    author_email="boersma.marcel@gmail.com",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    description="AI4FinTech project repository",
    entry_points={
        "console_scripts": [
            "fintorch=fintorch.cli:fintorch",
            "fintrainer=fintorch.cli:trainer",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords="fintorch",
    name="fintorch",
    packages=find_packages(include=["fintorch", "fintorch.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/AI4FinTech/fintorch",
    version="0.1.8",
    zip_safe=False,
)
