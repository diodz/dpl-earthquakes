# Comparison: PR #60 vs PR #61

Both PRs address **Issue #33** (SUTVA/spillover diagnostics and geographic donor-exclusion robustness) for the earthquake SCM analysis. They were created within minutes of each other by the same author (`diodz`) using Cursor background agents. Both are currently **OPEN** against `main`.

---

## Summary at a Glance

| Dimension | PR #60 (`cursor/issue-33-fix-d494`) | PR #61 (`cursor/issue-33-fix-f382`) |
|---|---|---|
| **Title** | Issue 33 fix | Issue 33 fix |
| **Commits** | 1 | 2 |
| **Files changed** | 11 | 20 |
| **Lines added** | 872 | 1,301 |
| **Lines deleted** | 14 | 4 |
| **Analysis script** | `src/spillover_diagnostics_analysis.py` (542 lines) | `src/spillover_diagnostics.py` (787 lines) |
| **CI status** | Pending (Bugbot) | Pending (Bugbot) |

---

## 1. Architecture & Code Structure

### PR #60 — Compact, single-module approach
- Single new file: `src/spillover_diagnostics_analysis.py` (542 lines)
- Flat procedural design with a `main()` function
- Uses named scenarios via `@dataclass` (`ExclusionScenario`)
- Defines 2 exclusion rings for Chile (O'Higgins; O'Higgins + Araucanía) and 2 for NZ (Auckland + Wellington; North-Island corridor)
- Builds a combined flow-diagnostics table in one function (`_build_flow_diagnostics`)

### PR #61 — More comprehensive, modular approach
- Single new file: `src/spillover_diagnostics.py` (787 lines — 45% larger)
- Uses `@dataclass` (`_ExclusionResult`) to store full ring results including gap series
- Defines **4 exclusion rings for Chile** and **5 for NZ** (more granular geographic rings)
- Separates diagnostic concerns into distinct functions:
  - `_compute_population_flow_diagnostics()`
  - `_compute_sector_spillover_diagnostics()`
  - `_compute_economic_linkage_index()`
- Returns a dictionary of DataFrames rather than just printing
- Exports a consolidated Excel workbook (`spillover_diagnostics_workbook.xlsx`) via `xlsxwriter`

---

## 2. Output Artifacts

### PR #60 (4 new files)
| File | Type |
|---|---|
| `spillover_sensitivity_summary.csv` | Exclusion summary table |
| `spillover_gap_paths.csv` | Year-by-year gap series |
| `spillover_flow_diagnostics.csv` | Combined flow/linkage diagnostics |
| `spillover_sensitivity_paths.png` | Side-by-side gap path figure |

### PR #61 (14 new files)
| File | Type |
|---|---|
| `spillover_exclusion_summary.csv` | Exclusion summary table |
| `spillover_exclusion_gap_series.csv` | Year-by-year gap series |
| `spillover_sensitivity_statement.csv` | Automated sensitivity conclusion |
| `chile_population_flow_diagnostics.csv` | Chile population flow proxy |
| `chile_sector_spillover_diagnostics.csv` | Chile construction-share proxy |
| `chile_economic_linkage_index.csv` | Chile sector-correlation linkage |
| `nz_population_flow_diagnostics.csv` | NZ population flow proxy |
| `nz_sector_spillover_diagnostics.csv` | NZ construction-share proxy |
| `nz_economic_linkage_index.csv` | NZ sector-correlation linkage |
| `spillover_geographic_exclusion_gaps.png` | Side-by-side gap paths |
| `spillover_sensitivity_bar.png` | Sensitivity bar chart |
| `chile_spillover_paths.png` | Chile treated-vs-synthetic per ring |
| `nz_spillover_paths.png` | NZ treated-vs-synthetic per ring |
| `spillover_diagnostics_workbook.xlsx` | Consolidated Excel workbook |

**PR #61 produces 3.5x more output artifacts**, with country-specific diagnostic CSVs and additional figure types.

---

## 3. Diagnostic Depth

### Spillover Proxy Construction

| Proxy | PR #60 | PR #61 |
|---|---|---|
| Population flow (migration) | Combined in one table (pre/post growth, delta) | Separate per-country CSVs with adjacent/non-adjacent grouping |
| Construction-sector share | Combined in same table | Separate per-country CSVs with adjacent/non-adjacent grouping |
| Tradables exposure | Included in combined table | Not separately broken out |
| Pre-treatment growth correlation | Included in combined table | Separate "economic linkage index" per country |
| Sensitivity statement | Not automated | Auto-generated CSV with sign-consistency check |

PR #60 puts everything in one wide table (per-region rows). PR #61 disaggregates diagnostics into grouped summaries (treated vs adjacent vs non-adjacent) which is arguably more directly useful for the SUTVA argument.

### Geographic Exclusion Rings

| Country | PR #60 | PR #61 |
|---|---|---|
| **Chile** | 2 rings (baseline + 2 exclusions) | 4 rings (baseline + 3 progressively stricter exclusions) |
| **New Zealand** | 2 rings (baseline + 2 exclusions) | 5 rings (baseline + 4 progressively stricter exclusions) |

PR #61's additional rings include the most restrictive specifications (e.g., Chile Ring 3 excludes 5 donors; NZ Ring 4 excludes Auckland alone). This allows testing whether the convex hull breaks down under extreme restriction — PR #61 explicitly discusses this in the reviewer response.

---

## 4. Manuscript Changes (`main.tex`)

### PR #60 — 65 lines added
- Replaces one paragraph in Section 4.4 (Robustness) with a new summary referencing the appendix
- Adds a new appendix section "Spillover diagnostics and geographic donor-exclusion robustness" with:
  - 1 diagnostics table (Table: spillover flow/linkage diagnostics)
  - 1 sensitivity table (Table: baseline vs spillover-robust exclusions)
  - 1 figure (gap paths under progressive exclusions)
- Hardcoded specific numeric results in text (e.g., "-1.71%", "+7.52%")

### PR #61 — 174 lines added
- Adds a cross-reference sentence to Section 3.1 (Data and Variables) pointing to the appendix
- Adds a cross-reference sentence to Section 4.4 (Robustness)
- Adds a much more detailed appendix section "SUTVA and spillover diagnostics" with:
  - Population flow diagnostic tables (Chile and NZ, Tables B.1-B.2)
  - Sector spillover diagnostic tables (Chile and NZ, Tables B.3-B.4)
  - Economic linkage index tables (Chile and NZ, Tables B.5-B.6)
  - Progressive exclusion summary table (Table B.7)
  - Multiple figures (gap paths, sensitivity bar, individual country paths)
  - Detailed textual interpretation of each diagnostic
- Uses `\label{sec:spillover_appendix}` for cross-referencing

PR #61's manuscript additions are **2.7x larger** and more thoroughly structured with sub-sections, multiple tables, and interpretation text.

---

## 5. Reviewer Response (`response_to_reviewers.tex`)

### PR #60 — 25 lines added
- Addresses Reviewer #1 Comment 3.4
- References the single new analysis module
- Lists the three components and key quantitative results
- Does not add a separate Reviewer #2 response

### PR #61 — 37 lines added
- Addresses Reviewer #1 Comment 3.4 with detailed enumerated response
- **Also adds Reviewer #2 Major Comment 1** response (O'Higgins/Auckland contamination)
- Provides specific quantitative evidence for each concern
- Discusses the Auckland exclusion result interpretively (including Auckland understates the effect)

PR #61 addresses **both reviewers explicitly**, while PR #60 only addresses Reviewer #1.

---

## 6. Pipeline Integration (`run_analysis.py`)

### PR #60
- Inserts spillover diagnostics as **step 6** (before nighttime lights)
- Renumbers subsequent steps (7-10)
- Runs script from `PROJECT_ROOT` as cwd

### PR #61
- Inserts spillover diagnostics as **step 6b** (after nighttime lights)
- Does not renumber other steps
- Runs script from `src/` as cwd (which may cause path issues since the script constructs paths relative to `_PROJECT_ROOT`)

PR #60's pipeline ordering is cleaner (full renumbering). PR #61's "6b" insertion is less disruptive but slightly less clean.

---

## 7. Test Coverage (`tests/test_figures.py`)

| | PR #60 | PR #61 |
|---|---|---|
| New test assertions | 4 files checked | 14 files checked |
| Total test count | ~60 | ~70 |

PR #61's test coverage is significantly more thorough, checking all 14 new artifacts.

---

## 8. Reported Results Comparison

Both PRs report similar qualitative conclusions but differ in specific numbers due to different exclusion specifications and SCM fit parameters:

| Metric | PR #60 | PR #61 |
|---|---|---|
| Chile baseline mean post-gap | -1.71% | -4.5% |
| Chile Ring 1 | -5.42% | -5.8% |
| NZ baseline mean post-gap | +7.52% | +7.8% |
| NZ most restrictive ring | +5.63% | +12.7% |
| Core conclusion | Sign pattern preserved | Sign pattern preserved |

The differences in baseline estimates (e.g., -1.71% vs -4.5% for Chile) likely stem from different fit windows, optimizer settings (`optim_initial="ols"` vs `"equal"`), or year ranges included in the post-treatment average.

---

## 9. Strengths & Weaknesses

### PR #60 Strengths
- Simpler, more concise implementation (easier to review)
- Clean pipeline renumbering
- Consistent hardcoded results in manuscript text match the script output format
- Smaller diff, easier to merge

### PR #60 Weaknesses
- Fewer exclusion rings (2 per country vs 3-4)
- Only addresses Reviewer #1, not Reviewer #2
- Fewer output artifacts (no Excel workbook, no per-country diagnostic CSVs)
- Less thorough test coverage (4 vs 14 assertions)

### PR #61 Strengths
- More comprehensive diagnostic battery (population, sector, linkage separately)
- More exclusion rings provide a fuller robustness picture
- Addresses both Reviewer #1 and Reviewer #2 explicitly
- Consolidated Excel workbook for referee convenience
- More figures (sensitivity bar, per-country paths)
- More thorough test coverage
- Richer manuscript appendix with sub-tables and interpretation
- Auto-generated sensitivity statement with sign-consistency logic

### PR #61 Weaknesses
- Larger, more complex implementation (harder to review)
- "6b" pipeline numbering is awkward
- `cwd=os.path.join(PROJECT_ROOT, "src")` in run_analysis.py may cause issues
- Chile sector spillover diagnostics CSV has missing post-treatment values
- Slightly more dependencies (xlsxwriter for Excel export)

---

## 10. Recommendation

**PR #61 is the stronger implementation** for the purposes of addressing reviewer concerns comprehensively. It provides:
- More granular geographic exclusion rings
- Separate diagnostic CSVs that map directly to individual manuscript tables
- Explicit Reviewer #2 response
- A consolidated Excel workbook for referee convenience
- More thorough test coverage

However, PR #61 would benefit from:
1. Fixing the missing post-treatment construction-share values in `chile_sector_spillover_diagnostics.csv`
2. Changing the pipeline step from "6b" to a proper renumbered sequence
3. Verifying the `cwd` setting in `run_analysis.py` does not break path resolution

If only one PR should be merged, **PR #61 is recommended**. If elements of both are desirable, PR #60's cleaner pipeline integration could be adopted into PR #61's more comprehensive analysis.
