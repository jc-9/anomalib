# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Dataset implementations for anomaly detection.

This module provides data implementations for various anomaly detection tasks:

Base Classes:
    - ``AnomalibDataset``: Base class for all Anomalib datasets
    - ``AnomalibDepthDataset``: Base class for 3D/depth datasets
    - ``AnomalibVideoDataset``: Base class for video datasets

Depth Datasets:
    - ``Folder3DDataset``: Custom RGB-D data from folder structure
    - ``MVTec3DDataset``: MVTec 3D AD data with industrial objects

Image Datasets:
    - ``BTechDataset``: BTech data containing industrial objects
    - ``DatumaroDataset``: Dataset in Datumaro format (Intel Geti™ export)
    - ``FolderDataset``: Custom data from folder structure
    - ``KolektorDataset``: Kolektor surface defect data
    - ``MPDDDataset``: Metal Parts Defect Detection data
    - ``MVTecADDataset``: MVTec AD data with industrial objects
    - ``TabularDataset``: Custom tabular data with image paths and labels
    - ``VAD``: Valeo Anomaly Detection Dataset
    - ``VisaDataset``: Visual Anomaly data

Video Datasets:
    - ``AvenueDataset``: CUHK Avenue data for abnormal event detection
    - ``ShanghaiTechDataset``: ShanghaiTech Campus surveillance data
    - ``UCSDpedDataset``: UCSD Pedestrian data for anomaly detection

Example:
    >>> from anomalib.data.datasets import MVTecADDataset
    >>> data = MVTecADDataset(
    ...     root="./datasets/MVTec",
    ...     category="bottle",
    ...     split="train"
    ... )
"""

from .base import AnomalibDataset, AnomalibDepthDataset, AnomalibVideoDataset
from .depth import Folder3DDataset, MVTec3DDataset
from .image import (
    BTechDataset,
    DatumaroDataset,
    FolderDataset,
    KolektorDataset,
    MPDDDataset,
    MVTecADDataset,
    TabularDataset,
    VADDataset,
    VisaDataset,
)
from .video import AvenueDataset, ShanghaiTechDataset, UCSDpedDataset

__all__ = [
    # Base
    "AnomalibDataset",
    "AnomalibDepthDataset",
    "AnomalibVideoDataset",
    # Depth
    "Folder3DDataset",
    "MVTec3DDataset",
    # Image
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MPDDDataset",
    "MVTecADDataset",
    "TabularDataset",
    "VADDataset",
    "VisaDataset",
    # Video
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
]
