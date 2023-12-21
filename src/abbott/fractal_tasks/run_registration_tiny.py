from abbott.fractal_tasks.compute_registration_elastix import (
    compute_registration_elastix,
)

# input_paths = [
#     "/Users/joel/shares/workShareJoel/fractal-demos/examples/"
#     "03_cardio_multiplexing/tmp_multiplex-0140/"
# ]
# component = "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr/B/03/1"

input_paths = [
    "/Users/joel/shares/dataShareJoel/jluethi/Fractal/"
    "20231221-zebrafish-registration/subset/"
]
component = "AssayPlate_Greiner_#655090.zarr/B/02/1"
metadata = {"coarsening_xy": 2}
output_path = input_paths[0]
parameter_files = ["params_translation_level0.txt"]
# Task-specific arguments
wavelength_id = "A01_C01"
roi_table = "FOV_ROI_table"
reference_cycle = "0"
level = 0
intensity_normalization = False

compute_registration_elastix(
    input_paths=input_paths,
    output_path=output_path,
    component=component,
    metadata=metadata,
    wavelength_id=wavelength_id,
    roi_table=roi_table,
    reference_cycle=reference_cycle,
    parameter_files=parameter_files,
    level=level,
    intensity_normalization=intensity_normalization,
)
