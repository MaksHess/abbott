from abbott.fractal_tasks.compute_registration_elastix import (
    compute_registration_elastix,
)

input_paths = [
    "/Users/joel/shares/dataShareJoel/jluethi/Fractal/"
    "20230906-zebrafish-registration/F002"
]

metadata = {"coarsening_xy": 2}
output_path = "test_output"
component = "AssayPlate_Greiner_#655090.zarr/B/02/1"
# Task-specific arguments
wavelength_id = "A01_C01"
roi_table = "FOV_ROI_table"
reference_cycle = 0
level = 2

compute_registration_elastix(
    input_paths=input_paths,
    output_path=output_path,
    component=component,
    metadata=metadata,
    wavelength_id=wavelength_id,
    roi_table=roi_table,
    reference_cycle=reference_cycle,
    level=level,
)
