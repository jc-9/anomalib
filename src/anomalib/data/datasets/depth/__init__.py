# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch Dataset Implementations of Anomalib Depth Datasets.

This module provides data implementations for working with RGB-D (depth) data in
anomaly detection tasks. The following datasets are available:

- ``Folder3DDataset``: Custom data for loading RGB-D data from a folder structure
- ``MVTec3DDataset``: Implementation of the MVTec 3D-AD data

Example:
    >>> from anomalib.data.datasets import Folder3DDataset
    >>> data = Folder3DDataset(
    ...     name="custom",
    ...     root="datasets/custom",
    ...     normal_dir="normal",
    ...     normal_depth_dir="normal_depth"
    ... )

    >>> from anomalib.data.datasets import MVTec3DDataset
    >>> data = MVTec3DDataset(
    ...     root="datasets/MVTec3D",
    ...     category="bagel"
    ... )
"""

from .folder_3d import Folder3DDataset
from .mvtec_3d import MVTec3DDataset

__all__ = ["Folder3DDataset", "MVTec3DDataset"]
