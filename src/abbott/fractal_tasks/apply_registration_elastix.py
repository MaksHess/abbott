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
"""Calculates translation for 2D image-based registration."""
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from abbott.io.conversions import to_itk, to_numpy
from abbott.registration.itk_elastix import apply_transform, load_parameter_files
from fractal_tasks_core.channels import OmeroChannel, get_omero_channel_list
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
)

# from fractal_tasks_core.roi import is_standard_roi_table
# from fractal_tasks_core.roi import load_region
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.utils import get_table_path_dict
from pydantic.decorator import validate_arguments

logger = logging.getLogger(__name__)


@validate_arguments
def apply_registration_elastix(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    roi_table: str = "FOV_ROI_table",
    reference_cycle: str = "0",
    overwrite_input: bool = False,
    registration_folder: str = "transforms",
):
    """
    Apply registration calculated by Compute Registration Elastix task.

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
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table: Name of the ROI table over which the registration was
            calculated. The task will loop over these ROIs and apply the
            transformation.
        reference_cycle: Which cycle to register against. Defaults to 0,
            which is the first OME-Zarr image in the well, usually the first
            cycle that was provided
        overwrite_input: Whether the old image data should be replaced with the
            newly registered image data.
        registration_folder: The folder in which the calculated transformations
            were saved by the compute_registration_elastix task
            (relative to the input_path).

    """
    logger.info(component)

    input_path = Path(input_paths[0])
    # Set OME-Zarr paths
    zarr_img_cycle_x = input_path / component

    logger.info(
        f"Running `apply_registration_to_image` on {input_path=}, "
        f"{component=}, {roi_table=} and {reference_cycle=}. "
        f"Using {overwrite_input=}"
    )

    new_component = "/".join(
        component.split("/")[:-1] + [component.split("/")[-1] + "_registered"]
    )

    # If the task is run for the reference cycle, exit
    if zarr_img_cycle_x.name == str(reference_cycle):
        logger.info(
            f"Applying registration for cycle {zarr_img_cycle_x.name}, which "
            "is the reference_cycle. Not changing anything."
        )
        # Should the reference cycle be renamed?
        # if not overwrite_input:
        #     os.rename(f"{input_path / component}", f"{input_path / new_component}")

        return {}

    ROI_table_cycle = ad.read_zarr(f"{input_path / component}/tables/{roi_table}")

    ngff_image_meta = load_NgffImageMeta(str(input_path / component))
    coarsening_xy = ngff_image_meta.coarsening_xy
    num_levels = ngff_image_meta.num_levels

    ####################
    # Process images
    ####################
    logger.info("Write the registered Zarr image to disk")
    write_registered_zarr(
        input_path=input_path,
        component=component,
        new_component=new_component,
        ROI_table=ROI_table_cycle,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.mean,
        registration_folder=registration_folder,
    )

    ####################
    # Process labels
    ####################
    # TODO: Can labels actually be registered with ITK?
    # e.g. will it do interpolation or not? Other issues with labels?
    # Currently turned off
    try:
        labels_group = zarr.open_group(f"{input_path / component}/labels", "r")
        label_list = labels_group.attrs["labels"]
    except (zarr.errors.GroupNotFoundError, KeyError):
        label_list = []

    if label_list:
        logger.warn("Labels are currently not written into the registered image")

    # if label_list:
    #     logger.info(f"Processing the label images: {label_list}")
    #     labels_group = zarr.group(f"{input_path / new_component}/labels")
    #     labels_group.attrs["labels"] = label_list

    #     for label in label_list:
    #         label_component = f"{component}/labels/{label}"
    #         label_component_new = f"{new_component}/labels/{label}"
    #         write_registered_zarr(
    #             input_path=input_path,
    #             component=label_component,
    #             new_component=label_component_new,
    #             ROI_table=ROI_table_cycle,
    #             ROI_table_ref=ROI_table_ref,
    #             num_levels=num_levels,
    #             coarsening_xy=coarsening_xy,
    #             aggregation_function=np.max,
    #         )

    ####################
    # Copy all the tables from the existing image
    ####################
    table_dict = get_table_path_dict(input_path, component)

    if table_dict:
        logger.info(f"Processing the tables: {table_dict}")
        table_group = zarr.group(f"{input_path / new_component}")

        for table in table_dict.keys():
            logger.info(f"Copying table: {table}")
            old_table_group = zarr.open_group(table_dict[table], mode="r")
            # Write the Zarr table
            curr_table = ad.read_zarr(table_dict[table])
            write_table(
                table_group,
                table,
                curr_table,
                table_attrs=old_table_group.attrs.asdict(),
                overwrite=True,
            )

    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info("Replace original zarr image with the newly created Zarr image")
        # Potential for race conditions: Every cycle reads the
        # reference cycle, but the reference cycle also gets modified
        # See issue #516 for the details
        os.rename(f"{input_path / component}", f"{input_path / component}_tmp")
        os.rename(f"{input_path / new_component}", f"{input_path / component}")
        shutil.rmtree(f"{input_path / component}_tmp")
    else:
        # FIXME: Add the new zarr image to the list of images in the well
        pass
        # The thing that would be missing in this branch is that Fractal
        # isn't aware of the new component. If there's a way to add it back,
        # that's the only thing that would be required here

    # TODO: Return changed metadata if no overwrite was used?
    return {}


def write_registered_zarr(
    input_path: Path,
    component: str,
    new_component: str,
    ROI_table: ad.AnnData,
    num_levels: int,
    coarsening_xy: int = 2,
    aggregation_function: Callable = np.mean,
    registration_folder: str = "transforms",
):
    """
    Write registered zarr array based on ROI tables.

    This function loads the image or label data from a zarr array based on the
    ROI bounding-box coordinates and stores them into a new zarr array.
    The new Zarr array has the same shape as the original array, but will have
    0s where the ROI tables don't specify loading of the image data.
    The ROIs loaded from `list_indices` will be written into the
    `list_indices_ref` position, thus performing translational registration if
    the two lists of ROI indices vary.

    Args:
        input_path: Base folder where the Zarr is stored
            (does not contain the Zarr file itself)
        component: Path to the OME-Zarr image that is processed. For example:
            `"20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/1"`
        new_component: Path to the new Zarr image that will be written
            (also in the input_path folder). For example:
            `"20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr/B/03/1_registered"`
        ROI_table: Fractal ROI table for the component
        ROI_table_ref: Fractal ROI table for the reference cycle
        num_levels: Number of pyramid layers to be created (argument of
            `build_pyramid`).
        coarsening_xy: Coarsening factor between pyramid levels
        aggregation_function: Function to be used when downsampling (argument
            of `build_pyramid`).
        registration_folder: The folder in which the calculated transformations
            were saved by the compute_registration_elastix task
            (relative to the input_path).

    """
    # Read pixel sizes from Zarr attributes
    ngff_image_meta = load_NgffImageMeta(str(input_path / component))
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    # Create list of indices for 3D ROIs
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )

    old_image_group = zarr.open_group(f"{input_path / component}", mode="r")
    new_image_group = zarr.group(f"{input_path / new_component}")
    new_image_group.attrs.put(old_image_group.attrs.asdict())

    # Loop over all channels. For each channel, write full-res image data.
    data_array = da.from_zarr(old_image_group["0"])
    # Create dask array with 0s of same shape
    # new_array = da.zeros_like(data_array)
    zarrurl_new = input_path / new_component

    new_zarr_array = zarr.create(
        shape=data_array.shape,
        chunks=data_array.chunksize,
        dtype=data_array.dtype,
        store=zarr.storage.FSStore(f"{zarrurl_new}/0"),
        overwrite=True,  # FIXME: Overwrite preexisting output?
        dimension_separator="/",
    )
    # new_image_group.attrs.put(old_image_group.attrs.asdict())

    channels: list[OmeroChannel] = get_omero_channel_list(
        image_zarr_path=input_path / component
    )

    for i_ROI, roi_indices in enumerate(list_indices):
        region = convert_indices_to_regions(roi_indices)
        parameter_path = Path(input_path) / "registration" / registration_folder
        fn_pattern = f"{component}_roi_{i_ROI}_t*.txt"

        # FIXME: Improve sorting to always achieve correct order (above 9 items)
        parameter_files = sorted(parameter_path.glob(fn_pattern))
        parameter_object = load_parameter_files([str(x) for x in parameter_files])

        for i_c, channel in enumerate(channels):
            logger.info(f"Processing ROI {i_ROI}, channel {channel}")
            # Define region
            channel_region = (slice(i_c, i_c + 1), *region)
            itk_img = to_itk(
                np.squeeze(data_array[channel_region].compute()),
                scale=tuple(pxl_sizes_zyx),
            )
            registered_roi = apply_transform(
                itk_img,
                parameter_object,
            )
            # Write to disk
            img = np.expand_dims(to_numpy(registered_roi), 0)
            da.array(img).to_zarr(
                url=new_zarr_array,
                region=channel_region,
                compute=True,
            )

    # Starting from on-disk highest-resolution data, build and write to
    # disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{input_path / new_component}",
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_array.chunksize,
        aggregation_function=aggregation_function,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_elastix,
        logger_name=logger.name,
    )
