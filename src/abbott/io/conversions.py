import warnings
from typing import TYPE_CHECKING

import h5py
import itk
import numpy as np

if TYPE_CHECKING:
    from itk.support.types import ImageBase as ITKImage

DTYPE_CONVERSION = {
    np.dtype("uint64"): np.dtype("uint16"),
    np.dtype("uint32"): np.dtype("uint16"),
    np.dtype("uint16"): np.dtype("uint16"),
    np.dtype("uint8"): np.dtype("uint8"),
    np.dtype("int64"): np.dtype("uint16"),
    np.dtype("int32"): np.dtype("uint16"),
    np.dtype("int16"): np.dtype("int16"),
    np.dtype("float64"): np.dtype("float64"),
    np.dtype("float32"): np.dtype("float32"),
    np.dtype("float16"): np.dtype("float16"),
    np.dtype("bool"): np.dtype("uint8"),
}


def to_itk(
    img: np.ndarray | ITKImage | h5py.Dataset,
    scale: tuple[int, ...] | None = None,
    conversion_warning: bool = True,
) -> ITKImage:
    """Convert something image-like to `itk.Image`.

    Args:
        img: Image to convert.
        scale: Image scale in numpy (!) conventions ([z], y, x).
        conversion_warning: Warning when data types are converted. Defaults to True.

    Raises:
        ValueError: No `scale` provided with np.ndarray.
        TypeError: Unknown image type.

    Returns:
        ITK image.
    """
    if isinstance(img, np.ndarray):
        if scale is None:
            raise ValueError(
                """You need to explicitly specify an image `scale` when converting from
                numpy.ndarray to itk.Image!"""
            )
        new_dtype = DTYPE_CONVERSION[img.dtype]
        if conversion_warning and img.dtype != new_dtype:
            warnings.warn(f"Converting {img.dtype} to {new_dtype}", stacklevel=2)
        img = img.astype(new_dtype)
        trans_img = itk.GetImageFromArray(img)
        trans_img.SetSpacing(scale[::-1])
    elif isinstance(img, itk.Image):
        trans_img = img
        if scale is None:
            scale = tuple(img.GetSpacing())[::1]
        trans_img.SetSpacing(scale[::-1])
    elif isinstance(img, h5py.Dataset):
        img_dset = img
        img = to_numpy(img_dset)
        new_dtype = DTYPE_CONVERSION[img.dtype]
        if conversion_warning:
            warnings.warn(f"Converting {img.dtype} to {new_dtype}", stacklevel=2)

        trans_img = itk.GetImageFromArray(img.astype(new_dtype))
        if scale is None:
            scale = tuple(
                img_dset.attrs.get("element_size_um", img_dset.ndim * [1.0]).astype(
                    np.float64
                )
            )
        trans_img.SetSpacing(scale[::-1])
    else:
        raise TypeError(f"Unknown image type: {type(img)}")
    return trans_img


def to_numpy(img: np.ndarray | ITKImage | h5py.Dataset) -> np.ndarray:
    """Convert to numpy.

    Args:
        img: Image.

    Raises:
        ValueError: Unknown image type.

    Returns:
        Numpy array.
    """
    if isinstance(img, (itk.Image, itk.VectorImage)):
        trans_img = itk.GetArrayFromImage(img)
    elif isinstance(img, np.ndarray):
        trans_img = img
    elif isinstance(img, h5py.Dataset):
        trans_img = img[...]
    else:
        raise ValueError(f"Unknown image type: {type(img)}")
    return trans_img
