# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Calculates translation for image-based registration."""
import logging
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import dask.array as da
import itk
import numpy as np
import zarr
from abbott.io.conversions import to_itk, to_numpy
from abbott.registration.itk_elastix import register_transform_only
from fractal_tasks_core.channels import OmeroChannel, get_channel_from_image_zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from pydantic.decorator import validate_arguments
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@validate_arguments
def compute_registration_elastix(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    wavelength_id: str,
    parameter_files: list[str],
    roi_table: str = "FOV_ROI_table",
    reference_cycle: str = "0",
    level: int = 2,
    intensity_normalization: bool = True,
    registration_folder: str = "transforms",
) -> dict[str, Any]:
    """
    Calculate registration based on images.

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating the transformation for that ROI
    3. Storing the calculated transformation in the ROI table

    Parallelization level: image

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: Dictionary containing metadata about the OME-Zarr. This task
            requires the following elements to be present in the metadata.
            `coarsening_xy (int)`: coarsening factor in XY of the downsampling
            when building the pyramid. (standard argument for Fractal tasks,
            managed by Fractal server).
        wavelength_id: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        parameter_files: TBD
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        reference_cycle: Which cycle to register against. Defaults to 0,
            which is the first OME-Zarr image in the well (usually the first
            cycle that was provided).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.
        intensity_normalization: TBD
        registration_folder: TBD

    """
    # TODO: Optionally turn on/off Elastix logging (very verbose)
    logger.info(
        f"Running for {input_paths=}, {component=}. \n"
        f"Calculating translation registration per {roi_table=} for "
        f"{wavelength_id=}."
    )
    # Set OME-Zarr paths
    zarr_img_cycle_x = Path(input_paths[0]) / component

    # If the task is run for the reference cycle, exit
    # TODO: Improve the input for this: Can we filter components to not
    # run for itself?
    alignment_cycle = zarr_img_cycle_x.name
    if alignment_cycle == str(reference_cycle):
        logger.info(
            "Calculate registration image-based is running for "
            f"cycle {alignment_cycle}, which is the reference_cycle."
            "Thus, exiting the task."
        )
        return {}
    else:
        logger.info(
            "Calculate registration image-based is running for "
            f"cycle {alignment_cycle}"
        )

    zarr_img_ref_cycle = zarr_img_cycle_x.parent / str(reference_cycle)

    # Read some parameters from Zarr metadata
    ngff_image_meta = load_NgffImageMeta(str(zarr_img_ref_cycle))
    coarsening_xy = ngff_image_meta.coarsening_xy

    # Get channel_index via wavelength_id.
    # Intially only allow registration of the same wavelength
    channel_ref: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=str(zarr_img_ref_cycle),
        wavelength_id=wavelength_id,
    )
    channel_index_ref = channel_ref.index

    channel_align: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=str(zarr_img_cycle_x),
        wavelength_id=wavelength_id,
    )
    channel_index_align = channel_align.index

    # Lazily load zarr array
    data_reference_zyx = da.from_zarr(f"{zarr_img_ref_cycle}/{level}")[
        channel_index_ref
    ]
    data_alignment_zyx = da.from_zarr(f"{zarr_img_cycle_x}/{level}")[
        channel_index_align
    ]

    # Read ROIs
    ROI_table_ref = ad.read_zarr(f"{zarr_img_ref_cycle}/tables/{roi_table}")
    ROI_table_x = ad.read_zarr(f"{zarr_img_cycle_x}/tables/{roi_table}")
    logger.info(f"Found {len(ROI_table_x)} ROIs in {roi_table=} to be processed.")

    # Check that table type of ROI_table_ref is valid. Note that
    # "ngff:region_table" and None are accepted for backwards compatibility
    valid_table_types = [
        "roi_table",
        "masking_roi_table",
        "ngff:region_table",
        None,
    ]

    ROI_table_ref_group = zarr.open_group(
        f"{zarr_img_ref_cycle}/tables/{roi_table}",
        mode="r",
    )
    ref_table_attrs = ROI_table_ref_group.attrs.asdict()
    ref_table_type = ref_table_attrs.get("type")
    if ref_table_type not in valid_table_types:
        raise ValueError(
            f"Table '{roi_table}' (with type '{ref_table_type}') is "
            "not a valid ROI table."
        )

    # For each cycle, get the relevant info
    # TODO: Add additional checks on ROIs?
    if (ROI_table_ref.obs.index != ROI_table_x.obs.index).all():
        raise ValueError(
            "Registration is only implemented for ROIs that match between the "
            "cycles (e.g. well, FOV ROIs). Here, the ROIs in the reference "
            "cycles were {ROI_table_ref.obs.index}, but the ROIs in the "
            "alignment cycle were {ROI_table_x.obs.index}"
        )

    # For each cycle, get the relevant info
    # TODO: Add additional checks on ROIs?
    if (ROI_table_ref.obs.index != ROI_table_x.obs.index).all():
        raise ValueError(
            "Registration is only implemented for ROIs that match between the "
            "cycles (e.g. well, FOV ROIs). Here, the ROIs in the reference "
            "cycles were {ROI_table_ref.obs.index}, but the ROIs in the "
            "alignment cycle were {ROI_table_x.obs.index}"
        )
    # TODO: Make this less restrictive? i.e. could we also run it if different
    # cycles have different FOVs? But then how do we know which FOVs to match?
    # If we relax this, downstream assumptions on matching based on order
    # in the list will break.

    # Read pixel sizes from zarr attributes
    ngff_image_meta_cycle_x = load_NgffImageMeta(str(zarr_img_cycle_x))
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx_cycle_x = ngff_image_meta_cycle_x.get_pixel_sizes_zyx(level=0)

    if pxl_sizes_zyx != pxl_sizes_zyx_cycle_x:
        raise ValueError("Pixel sizes need to be equal between cycles for registration")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices_ref = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_ref, roi_table)

    list_indices_cycle_x = convert_ROI_table_to_indices(
        ROI_table_x,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices_cycle_x, roi_table)

    num_ROIs = len(list_indices_ref)
    compute = True
    # FIXME: Loop again
    for i_ROI in range(num_ROIs):
        # if True:
        #     i_ROI = 0
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} " f"for channel {channel_align}."
        )
        img_ref = load_region(
            data_zyx=data_reference_zyx,
            region=convert_indices_to_regions(list_indices_ref[i_ROI]),
            compute=compute,
        )
        img_cycle_x = load_region(
            data_zyx=data_alignment_zyx,
            region=convert_indices_to_regions(list_indices_cycle_x[i_ROI]),
            compute=compute,
        )

        ##############
        #  Calculate the transformation
        ##############
        # Basic version (no padding, no internal binning)
        if img_ref.shape != img_cycle_x.shape:
            raise NotImplementedError(
                "This registration is not implemented for ROIs with "
                "different shapes between cycles"
            )

        logger.info(f"Pixel sizes: {tuple(pxl_sizes_zyx)}")

        ref = to_itk(img_ref, scale=tuple(pxl_sizes_zyx))
        # print(ref.GetPixel(500, 500, 1))
        move = to_itk(img_cycle_x, scale=tuple(pxl_sizes_zyx_cycle_x))
        if intensity_normalization:
            # FIXME: Decide on how to make quantile rescaling more robust
            ref_norm = quantile_rescale_exp(ref, rejected_planes=(0, -1))
            move_norm = quantile_rescale_exp(move, rejected_planes=(0, -1))
        else:
            ref_norm = ref
            move_norm = move
        trans = register_transform_only(ref_norm, move_norm, parameter_files)

        # Write transform parameter files
        # TODO: Add overwrite check (it overwrites by default)
        for i in range(trans.GetNumberOfParameterMaps()):
            trans_map = trans.GetParameterMap(i)
            fn = (
                Path(output_path)
                / "registration"
                / registration_folder
                / (f"{component}_roi_{i_ROI}_t{i}.txt")
            )
            fn.parent.mkdir(exist_ok=True, parents=True)
            trans.WriteParameterFile(trans_map, fn.as_posix())

    return {}


def quantile_rescale_exp(
    img_itk: itk.Image,
    q: tuple[float, float] = (0.01, 0.999),
    rejected_planes: tuple[int, int] = (15, 50),  # FIXME: Expose this
) -> itk.Image:
    """TBD."""
    scale = tuple(img_itk.GetSpacing())[::-1]
    img = to_numpy(img_itk)
    lower, upper = np.quantile(img, q, axis=(1, 2))
    x = np.arange(len(lower))
    lower_r = lower[rejected_planes[0] : -rejected_planes[1]]
    upper_r = upper[rejected_planes[0] : -rejected_planes[1]]
    x_r = x[rejected_planes[0] : -rejected_planes[1]]

    lm_lower = linregress(x_r, np.log1p(lower_r))
    lm_upper = linregress(x_r, np.log1p(upper_r))

    lower_bounds = np.expm1(lm_lower.intercept + lm_lower.slope * x)
    upper_bounds = np.expm1(lm_upper.intercept + lm_upper.slope * x)

    res = (img - np.expand_dims(lower_bounds, axis=(1, 2))) / np.expand_dims(
        upper_bounds - lower_bounds, axis=(1, 2)
    )

    out_itk = to_itk(res, scale=scale)
    # out_itk.SetSpacing(img_itk.GetSpacing())
    out_itk.SetOrigin(img_itk.GetOrigin())
    return out_itk


if __name__ == "__main__":
    # from fractal_tasks_core.tasks._utils import run_fractal_task

    # run_fractal_task(
    #     task_function=compute_registration_elastix,
    #     logger_name=logger.name,
    # )
    input_paths = [
        "/Users/joel/shares/dataShareJoel/jluethi/Fractal/"
        "20230906-zebrafish-registration/full/"
    ]
    output_path = (
        "/Users/joel/shares/dataShareJoel/jluethi/Fractal/"
        "20230906-zebrafish-registration/full/"
    )
    component = "AssayPlate_Greiner_#655090.zarr/B/02/1"
    metadata = {"coarsening_xy": 2}

    wavelength_id = "A03_C03"
    parameter_files = ["/Users/joel/Desktop/params_translation_level0.txt"]
    level = 0

    compute_registration_elastix(
        input_paths=input_paths,
        output_path=output_path,
        component=component,
        metadata=metadata,
        wavelength_id=wavelength_id,
        parameter_files=parameter_files,
        level=level,
    )
