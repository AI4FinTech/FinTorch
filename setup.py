#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["Click>=7.0", "torch", "pytorch_lightning", "kaggle", "polars", "numpy"]

test_requirements = [
    "pytest>=3",
]

dev_requirements = ["sphinx"]

setup(
    author="Marcel Boersma",
    author_email="boersma.marcel@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="AI4FinTech project repository",
    entry_points={
        "console_scripts": [
            "fintorch=fintorch.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="fintorch",
    name="fintorch",
    packages=find_packages(include=["fintorch", "fintorch.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"test": test_requirements, "dev": dev_requirements},
    url="https://github.com/boersmamarcel/fintorch",
    version="0.3.0",
    zip_safe=False,
)
