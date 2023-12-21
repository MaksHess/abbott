from typing import Sequence, Tuple

import itk


def cast(img: itk.Image, out_type: type):
    """Cast an image to `out_type`.

    Args:
        img: Image to cast.
        out_type: Output type.

    Returns:
        Casted image.
    """
    castImageFilter = itk.ClampImageFilter[type(img), out_type].New()
    castImageFilter.SetInput(img)
    castImageFilter.Update()
    return castImageFilter.GetOutput()


def register(
    fixed: itk.Image,
    moving: itk.Image,
    parameter_files: Sequence[str],
    fixed_mask: itk.Image = None,
    moving_mask: itk.Image = None,
) -> Tuple[itk.Image, itk.ParameterObject]:
    """Register an image and apply the resulting transform.

    Args:
        fixed: Fixed image.
        moving: Moving image.
        parameter_files: Paths to parameter files.
        fixed_mask: Mask to apply to fixed image. Defaults to None.
        moving_mask: Mask to apply to moving image. Defaults to None.

    Returns:
        The registered moving image and transform parameters.
    """
    parameter_object = load_parameter_files(parameter_files)

    elastix_object = itk.ElastixRegistrationMethod.New(
        cast(fixed, itk.Image[itk.F, fixed.GetImageDimension()]),
        cast(moving, itk.Image[itk.F, fixed.GetImageDimension()]),
    )
    if fixed_mask is not None:
        elastix_object.SetFixedMask(fixed_mask)
    if moving_mask is not None:
        elastix_object.SetMovingMask(moving_mask)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(True)
    elastix_object.UpdateLargestPossibleRegion()

    out = elastix_object.GetOutput()
    trans = elastix_object.GetTransformParameterObject()
    return cast(out, type(fixed)), trans


def register_transform_only(
    fixed: itk.Image,
    moving: itk.Image,
    parameter_files: Sequence[str],
    fixed_mask: itk.Image = None,
    moving_mask: itk.Image = None,
) -> itk.ParameterObject:
    """Register an image and apply the resulting transform.

    Args:
        fixed: Fixed image.
        moving: Moving image.
        parameter_files: Paths to parameter files.
        fixed_mask: Mask to apply to fixed image. Defaults to None.
        moving_mask: Mask to apply to moving image. Defaults to None.

    Returns:
        Transform parameters.
    """
    parameter_object = load_parameter_files(parameter_files)

    elastix_object = itk.ElastixRegistrationMethod.New(
        cast(fixed, itk.Image[itk.F, fixed.GetImageDimension()]),
        cast(moving, itk.Image[itk.F, fixed.GetImageDimension()]),
    )
    if fixed_mask is not None:
        elastix_object.SetFixedMask(fixed_mask)
    if moving_mask is not None:
        elastix_object.SetMovingMask(moving_mask)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(True)
    elastix_object.UpdateLargestPossibleRegion()

    trans = elastix_object.GetTransformParameterObject()
    return trans


def load_parameter_files(parameter_files: Sequence[str]) -> itk.ParameterObject:
    """Load one or multiple parameter files into a parameter object.

    Args:
        parameter_files: Path(s) to parameter file(s).

    Returns:
        Parameter object.
    """
    parameter_object = itk.ParameterObject.New()
    for parameter_file in parameter_files:
        parameter_object.AddParameterFile(parameter_file)
    return parameter_object


def apply_transform(moving: itk.Image, trans: itk.ParameterObject) -> itk.Image:
    """Apply the transform on an image.

    Args:
        moving: Moving image.
        trans: Transform parameters.

    Returns:
        Transformed image.
    """
    transformix_object = itk.TransformixFilter.New(
        cast(moving, itk.Image[itk.F, moving.GetImageDimension()])
    )
    transformix_object.SetTransformParameterObject(trans)
    transformix_object.UpdateLargestPossibleRegion()

    out = transformix_object.GetOutput()
    return cast(out, type(moving))


def copy_parameter_map(parameter_map: itk.ParameterObject) -> itk.ParameterObject:
    """Copy (clone) a parameter object.

    Args:
        parameter_map: Map to copy.

    Returns:
        Copy.
    """
    parameter_map_out = parameter_map.Clone()
    for i in range(parameter_map.GetNumberOfParameterMaps()):
        parameter_map_out.AddParameterMap(parameter_map.GetParameterMap(i))
    return parameter_map_out
