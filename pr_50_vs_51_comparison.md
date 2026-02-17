# PR #50 vs PR #51 Comparison: Nighttime Lights Validation

Both PRs address the same reviewer request (Issue #37): corroborate GDP-based SCM findings with nighttime-light (NTL) data. They take fundamentally different approaches.

---

## Summary Table

| Dimension | PR #50 | PR #51 |
|---|---|---|
| **Branch** | `cursor/nighttime-lights-validation-2f34` | `cursor/nighttime-lights-validation-fcc7` |
| **Lines added/removed** | +1,540 / -52 | +2,234 / -10 |
| **Files changed** | 18 | 19 |
| **NTL data source** | Synthetic proxy derived from GDP | Real satellite rasters (Zenodo: DMSP-OLS/VIIRS) |
| **SCM implementation** | Custom gradient-descent with simplex projection | `pysyncon` library (established, peer-reviewed) |
| **New dependencies** | None | `rasterio`, relaxed `shapely` pin |
| **Merge status** | MERGEABLE | CONFLICTING (has merge conflicts) |
| **CI / Bugbot** | Pending (still reviewing) | Passed |
| **Bugs found by Bugbot** | 3 (circularity, duplication, edge cases) | 1 (corrupt raster cache -- fixed) |
| **Reviewer response updated** | No | Yes (`response_to_reviewers.tex`) |
| **README updated** | No | Yes |
| **Tests updated** | No | Yes (`tests/test_figures.py`) |
| **run_analysis.py integrated** | No | Yes |

---

## 1. Data Approach (Most Important Difference)

### PR #50: GDP-derived proxy (circular)
PR #50 generates NTL proxy series **from the GDP data itself** using the Henderson et al. (2012) log-linear GDP-NTL elasticity (~0.5) with sensor-realistic measurement noise added. This means the "validation" data is a deterministic transformation of the very outcome being validated, plus random noise.

- Bugbot flagged this as a **high-severity bug**: "NTL validation is circular: proxy derived from GDP."
- After the fix, the manuscript text was corrected to call this a "pipeline consistency demonstration" rather than "independent validation."
- While conceptually useful for showing the SCM pipeline works on a transformed outcome, it does **not** address the reviewer's actual request for an **independent, objective proxy** to bypass reporting biases.

### PR #51: Actual satellite data (independent)
PR #51 downloads real harmonized DMSP-OLS/VIIRS annual rasters from Zenodo, aggregates pixel values to ADM1 regional boundaries using `rasterio` + `shapely`, and runs SCM on the resulting NTL panel. This is genuinely independent data from a completely different measurement system (satellite radiance vs. national accounts).

- The data pipeline includes: boundary downloads from geoBoundaries, raster downloads from Zenodo with caching, zonal statistics extraction, and geographic union of sub-regions.
- Two NTL products are used: `pcnl_harmonized` (DMSP+VIIRS harmonized) and `viirs_extrapolated` (alternative processing).
- This directly satisfies the reviewer's request for an objective proxy that bypasses GDP reporting biases.

**Verdict: PR #51 is clearly superior on data methodology.** PR #50's approach was correctly identified as circular; even after the fix it only claims "pipeline consistency," which is not what the reviewer asked for.

---

## 2. Scientific Findings

### PR #50
- Maule: NTL gap near zero (~+0.8%), confirming GDP-null finding.
- Canterbury: Noisy but directionally consistent with GDP-positive finding under some specifications.
- These results are expected since the NTL data is derived from GDP -- they cannot diverge much by construction.

### PR #51
- Canterbury: NTL confirms GDP-positive result (gap ~+7.8% average post-treatment, ~+10% in 2016).
- Maule: NTL **diverges** from GDP-null (positive NTL gap ~+13%), which is an interesting and honest finding.
- The divergence is explicitly discussed in the manuscript: NTL may capture "broader spatial re-lighting and infrastructure restoration dynamics that do not map one-to-one to measured value-added in regional GDP accounts."
- Urban-core mask analysis shows Maule's positive NTL signal is broader spatial re-lighting rather than urban production.

**Verdict: PR #51 produces more scientifically valuable and honest results.** The Maule divergence between NTL and GDP is genuinely informative and adds nuance to the paper. PR #50's results are tautological.

---

## 3. Code Quality and Engineering

### PR #50
- Refactored shared constants and `project_to_simplex` into `src/math_utils.py` (good code hygiene).
- Custom SCM implementation (gradient descent) -- more code to maintain, less tested than library.
- Generates pre-built CSV data files committed to the repo (337 + 301 rows).
- 11 PNG figures generated and committed.
- No integration with `run_analysis.py`, no tests, no README updates.

### PR #51
- Uses `pysyncon` library for SCM (well-tested, peer-reviewed).
- Full geospatial pipeline: boundary downloading, raster processing, zonal statistics.
- Atomic download with temp-file pattern (Bugbot fix) for robustness.
- Local cache with offline support (avoids re-downloading on subsequent runs).
- Properly integrated into project:
  - `run_analysis.py` updated to include NTL step.
  - `tests/test_figures.py` updated with 9 new expected output files.
  - `README.md` updated with NTL documentation.
  - `.gitignore` updated to exclude large raster caches.
  - `requirements.txt` updated with new dependency.
- Outputs rich intermediate data (regional panel, SCM weights, gaps CSVs).

**Verdict: PR #51 has significantly better engineering practices** -- proper testing, documentation, integration, and use of established libraries.

---

## 4. Manuscript Quality

### PR #50
- Adds 73 lines to `main.tex` in the robustness section.
- Includes: NTL validation paragraph, NTL vs GDP comparison figure, robustness table with 10 specifications, sensor robustness figure, spatial buffer figure.
- After Bugbot fix, correctly disclaims proxy nature -- but this weakens the validation claim considerably.
- Adds 2 bibliography entries (Henderson 2012, Chen & Nordhaus 2011).
- Does **not** update `response_to_reviewers.tex`.

### PR #51
- Adds 36 lines to `main.tex` as a new subsection "Independent validation using night-time lights."
- Includes: validation figure (paths + gaps), sensor robustness figure, spatial sensitivity figure.
- Honestly discusses the Maule NTL-GDP divergence and interprets it substantively.
- Updates `response_to_reviewers.tex` with 16 lines explaining exactly what was done and how.
- Fixes an unrelated cross-reference from "Figure 5" to `\ref{fig:sector_analysis}`.
- Updates the reviewer issue tracking document.

**Verdict: PR #51 has a more complete and honest manuscript contribution**, especially because it updates the reviewer response and honestly discusses divergent findings.

---

## 5. Risks and Concerns

### PR #50 Risks
- **Fundamental methodological flaw**: Even after the "circularity" fix, using GDP-derived NTL doesn't meaningfully address the reviewer's concern.
- If a reviewer scrutinizes the approach, they will likely reject the "validation" as uninformative.
- The committed CSV data files (638 rows of synthetic data) are not real satellite data.

### PR #51 Risks
- **Merge conflicts**: Currently in CONFLICTING state (would need conflict resolution).
- **External dependencies**: Requires downloading ~multi-GB raster files on first run (mitigated by caching).
- **New dependency**: Adds `rasterio` which has system-level dependencies (GDAL).
- **Network requirement**: First run needs internet access to download rasters and boundaries.
- **Reproducibility**: Depends on external Zenodo records remaining available (but uses DOI-based stable records).

---

## 6. Overall Recommendation

**PR #51 is the clearly better PR**, despite its merge conflicts and added complexity.

### Reasons:
1. **Methodological soundness**: Uses actually independent satellite data rather than a circular GDP-derived proxy. This is the single most important differentiator.
2. **Scientific value**: Discovers an interesting NTL-GDP divergence for Maule and discusses it honestly, adding real analytical value to the paper.
3. **Completeness**: Updates the reviewer response, README, tests, run script, and .gitignore -- a production-ready contribution.
4. **Engineering quality**: Uses established `pysyncon` library, has proper caching, atomic downloads, and test coverage.
5. **Reviewer satisfaction**: Actually delivers what the reviewer asked for (an independent, objective proxy), rather than a pipeline consistency check.

### PR #50's only advantages:
- Currently mergeable (no conflicts).
- Simpler to run (no large raster downloads needed).
- Refactored shared constants/utilities into `math_utils.py` (good code hygiene that PR #51 doesn't do).

### Suggested path forward:
Merge PR #51 (after resolving conflicts), and optionally cherry-pick the `math_utils.py` refactoring from PR #50 as a separate code-quality improvement.
