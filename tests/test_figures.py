import os
import pytest

output_dir = os.path.join(os.path.dirname(__file__), '..', 'article_assets')

@pytest.mark.parametrize("file_name", [
    "Maule_map.png",
    "Canterbury_map.png",
    "maule_gdp_paths.png",
    "nz_gdp_paths.png",
    "maule_gap.png",
    "nz_gap.png",
    "nz_population_paths.png",
    "nz_population_gap.png",
    "nz_total_gdp_paths.png",
    "nz_total_gdp_gap.png",
    "nz_outcome_extension_tables.xlsx",
    "nz_outcome_extension_summary.csv",
    "maule_placebos.png",
    "nz_placebos.png",
    "scm_uniform_confidence_sets.png",
    "chile_uniform_threshold_sensitivity.png",
    "nz_uniform_treatment_timing_sensitivity.png",
    "scm_uniform_confidence_sets.csv",
    "scm_uniform_confidence_summary.csv",
    "nz_scm_Construction.png",
    "nz_scm_Other_Sectors.png",
    "chile_jacknife.png",
    "nz_jacknife.png",
    "sdid_bias_corrected_summary.csv",
    "sdid_bias_corrected_gaps.csv",
    "sdid_bias_corrected_gaps.png",
    "sdid_penalized_lambda_grid.csv",
    "sdid_diagnostics.txt",
    "ntl_validation_paths_gaps.png",
    "ntl_sensor_processing_robustness.png",
    "ntl_spatial_sensitivity.png",
    "ntl_validation_summary.csv",
    "ntl_scm_summary.csv",
    "ntl_scm_gaps.csv",
    "ntl_scm_weights.csv",
    "ntl_maule_urban_mask_series.csv",
    "ntl_nz_buffer_sensitivity.csv",
])
def test_output_file_exists(file_name):
    file_path = os.path.join(output_dir, file_name)
    assert os.path.exists(file_path), f"{file_name} does not exist in the output directory"
    assert os.path.getsize(file_path) > 0, f"{file_name} is empty"
