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
    "maule_placebos.png",
    "nz_placebos.png",
    "nz_scm_Construction.png",
    "nz_scm_Other_Sectors.png",
    "chile_jacknife.png",
    "nz_jacknife.png",
    # Denominator effect
    "maule_pop_paths.png",
    "nz_pop_paths.png",
    "maule_gdp_total_paths.png",
    "nz_gdp_total_paths.png",
    # In-time placebos
    "maule_in_time_placebos.png",
    "nz_in_time_placebos.png",
    # Maule sectoral
    "maule_scm_Construction.png",
    "maule_scm_Other_Sectors.png",
    # Construction placebo
    "nz_scm_Construction_placebos.png",
    # Treatment timing sensitivity
    "nz_gdp_paths_t2010.png",
])
def test_output_file_exists(file_name):
    file_path = os.path.join(output_dir, file_name)
    assert os.path.exists(file_path), f"{file_name} does not exist in the output directory"
    assert os.path.getsize(file_path) > 0, f"{file_name} is empty"
