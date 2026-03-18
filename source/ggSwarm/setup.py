# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Installation script for the 'ggSwarm' python package."""

import os

import toml
from setuptools import setup

# Obtain the extension data from the extension.toml file
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # h5py>=3.12 bundles HDF5 2.x DLLs that conflict with Isaac Sim's
    # bundled HDF5 1.12.x DLLs on Windows (fatal DLL entry-point error).
    # Pin to <3.12 until Isaac Sim ships a compatible HDF5 version.
    "h5py>=3.9.0,<3.12",
    "psutil",
    "skrl>=1.1.0",
    "toml",
]


# Installation operation
setup(
    name="ggSwarm",
    packages=["ggSwarm"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
