# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Dataset implementations for anomaly detection in images.

This module provides data implementations for various image anomaly detection
datasets:

- ``BTechDataset``: BTech data containing industrial objects
- ``DatumaroDataset``: Dataset in Datumaro format (Intel Geti™ export)
- ``FolderDataset``: Custom data from folder structure
- ``KolektorDataset``: Kolektor surface defect data
- ``MVTecADDataset``: MVTec AD data with industrial objects
- ``MVTecLOCODataset``: MVTec LOCO data with logical and structural anomalies
- ``TabularDataset``: Custom tabular data with image paths and labels
- ``VAD``: Valeo Anomaly Detection Dataset
- ``VisaDataset``: Visual Anomaly data

Example:
    >>> from anomalib.data.datasets import MVTecADDataset
    >>> data = MVTecADDataset(
    ...     root="./datasets/MVTec",
    ...     category="bottle",
    ...     split="train"
    ... )
"""

from .btech import BTechDataset
from .datumaro import DatumaroDataset
from .folder import FolderDataset
from .kolektor import KolektorDataset
from .mpdd import MPDDDataset
from .mvtec_loco import MVTecLOCODataset
from .mvtecad import MVTecADDataset, MVTecDataset
from .mvtecad2 import MVTecAD2Dataset
from .realiad import RealIADDataset
from .tabular import TabularDataset
from .vad import VADDataset
from .visa import VisaDataset

__all__ = [
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MPDDDataset",
    "MVTecDataset",
    "MVTecADDataset",
    "MVTecAD2Dataset",
    "MVTecLOCODataset",
    "RealIADDataset",
    "TabularDataset",
    "VADDataset",
    "VisaDataset",
]
