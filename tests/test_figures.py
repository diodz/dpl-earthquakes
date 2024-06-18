import os
import pytest

output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')

@pytest.mark.parametrize("file_name", [
    "chile_scm.png",
    "nz_scm.png",
    "chile_placebo.png",
    "nz_placebo.png",
    "chile_jacknife.png",
    "nz_jacknife.png",
    "chile_regions_gdp.png",
    "nz_regions_gdp.png"
])
def test_output_file_exists(file_name):
    file_path = os.path.join(output_dir, file_name)
    assert os.path.exists(file_path), f"{file_name} does not exist in the output directory"
