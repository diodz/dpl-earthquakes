import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Helper functions

def compute_covariates(df, region, pre_years, covar_cols, outcome_col='GDP_per_capita'):
    sub = df[(df['Region'] == region) & (df['Year'].isin(pre_years))]

    def _mean_with_fallback(column):
        local_mean = sub[column].astype(float).mean(skipna=True)
        if pd.isna(local_mean):
            return float(df[column].astype(float).mean(skipna=True))
        return float(local_mean)

    covars = [_mean_with_fallback(outcome_col)]
    for col in covar_cols:
        covars.append(_mean_with_fallback(col))
    return np.array(covars, dtype=float)

def select_donors(df, treated, pre_years, covar_cols, num_donors=5):
    regions = df['Region'].dropna().unique().tolist()
    cov_treated = compute_covariates(df, treated, pre_years, covar_cols)
    distances = {}
    for region in regions:
        if region == treated:
            continue
        cov_reg = compute_covariates(df, region, pre_years, covar_cols)
        distances[region] = np.linalg.norm(cov_reg - cov_treated)
    sorted_regions = sorted(distances, key=lambda r: distances[r])
    selected = sorted_regions[:num_donors]
    inv = np.array([1 / distances[r] for r in selected])
    weights = inv / inv.sum()
    return selected, weights

def synthetic_series(df, treated, selected, weights, outcome_col='GDP_per_capita'):
    years = sorted(df['Year'].unique())
    series = {}
    for region in df['Region'].unique():
        s = df[df['Region'] == region].set_index('Year').reindex(years)[outcome_col].astype(float).values
        series[region] = s
    y_treated = series[treated]
    y_synth = np.zeros(len(years), dtype=float)
    for i, region in enumerate(selected):
        y_synth += weights[i] * series[region]
    return years, y_treated, y_synth


def _safe_inverse_distances(selected, distances):
    adjusted = []
    for region in selected:
        distance = distances[region]
        adjusted.append(1.0 / max(distance, 1e-8))
    inv = np.array(adjusted)
    return inv / inv.sum()


def run_in_time_placebo_inference(
    df,
    treated,
    actual_treatment_year,
    placebo_years,
    covar_cols,
    outcome_col,
    baseline_start_year,
    country_label,
    output_dir='article_assets',
    max_donors=5,
):
    """Run in-time placebo inference by shifting treatment timing through the pre-period.

    For each candidate treatment year, SCM is re-estimated using data from
    baseline_start_year up to the year before the candidate treatment year.
    The function exports a tidy CSV (post-treatment gaps + RMSPE statistics)
    and a compact two-panel figure contrasting placebo distributions with the
    actual treatment-year estimate.
    """
    work_df = df.copy()
    years = sorted(work_df['Year'].dropna().unique())
    regions = work_df['Region'].dropna().unique().tolist()

    if treated not in regions:
        raise ValueError(f'Treated region {treated} not found in Region column.')

    donors_all = [region for region in regions if region != treated]
    y_treated = (
        work_df[work_df['Region'] == treated]
        .set_index('Year')
        .reindex(years)[outcome_col]
        .astype(float)
        .values
    )

    records = []
    effect_rows = []
    candidate_years = sorted(set(placebo_years + [actual_treatment_year]))

    for candidate_year in candidate_years:
        pre_years = [y for y in years if baseline_start_year <= y < candidate_year]
        if not pre_years:
            continue

        cov_treated = compute_covariates(work_df, treated, pre_years, covar_cols, outcome_col=outcome_col)
        distances = {}
        for donor in donors_all:
            donor_cov = compute_covariates(work_df, donor, pre_years, covar_cols, outcome_col=outcome_col)
            distances[donor] = np.linalg.norm(donor_cov - cov_treated)

        selected = sorted(distances, key=lambda region: distances[region])[:min(max_donors, len(distances))]
        weights = _safe_inverse_distances(selected, distances)

        y_synth = np.zeros(len(years), dtype=float)
        for idx, donor in enumerate(selected):
            donor_series = (
                work_df[work_df['Region'] == donor]
                .set_index('Year')
                .reindex(years)[outcome_col]
                .astype(float)
                .values
            )
            y_synth += weights[idx] * donor_series

        gap = y_treated - y_synth
        pre_idx = [i for i, y in enumerate(years) if y in pre_years]
        post_idx = [i for i, y in enumerate(years) if y >= candidate_year]

        rmspe_pre = float(np.sqrt(np.mean(np.square(gap[pre_idx])))) if pre_idx else np.nan
        rmspe_post = float(np.sqrt(np.mean(np.square(gap[post_idx])))) if post_idx else np.nan
        rmspe_ratio = rmspe_post / rmspe_pre if rmspe_pre and not np.isclose(rmspe_pre, 0.0) else np.nan
        mean_post_gap = float(np.mean(gap[post_idx])) if post_idx else np.nan

        analysis_type = 'actual' if candidate_year == actual_treatment_year else 'placebo'
        effect_rows.append(
            {
                'Country': country_label,
                'AnalysisType': analysis_type,
                'CandidateTreatmentYear': candidate_year,
                'MeanPostGap': mean_post_gap,
                'RMSPE_Pre': rmspe_pre,
                'RMSPE_Post': rmspe_post,
                'RMSPE_Ratio': rmspe_ratio,
                'NumPreYears': len(pre_years),
            }
        )

        for i in post_idx:
            records.append(
                {
                    'Country': country_label,
                    'AnalysisType': analysis_type,
                    'CandidateTreatmentYear': candidate_year,
                    'Year': years[i],
                    'Gap': float(gap[i]),
                    'RMSPE_Pre': rmspe_pre,
                    'RMSPE_Post': rmspe_post,
                    'RMSPE_Ratio': rmspe_ratio,
                }
            )

    tidy_df = pd.DataFrame(records)
    effect_df = pd.DataFrame(effect_rows)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f'{country_label.lower()}_in_time_placebo_summary.csv'
    fig_path = output_path / f'{country_label.lower()}_in_time_placebo_compact.png'

    tidy_df.to_csv(csv_path, index=False)

    placebo_effects = effect_df[effect_df['AnalysisType'] == 'placebo']
    actual_effect = effect_df[effect_df['AnalysisType'] == 'actual'].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].hist(placebo_effects['MeanPostGap'].dropna(), bins=10, color='lightgray', edgecolor='black')
    axes[0].axvline(actual_effect['MeanPostGap'], color='red', linestyle='--', linewidth=1.5, label='Actual treatment year')
    axes[0].set_title('Post-gap distribution')
    axes[0].set_xlabel('Average post-treatment gap')
    axes[0].legend(loc='best', fontsize=8)

    axes[1].hist(placebo_effects['RMSPE_Ratio'].replace([np.inf, -np.inf], np.nan).dropna(), bins=10, color='lightgray', edgecolor='black')
    axes[1].axvline(actual_effect['RMSPE_Ratio'], color='red', linestyle='--', linewidth=1.5, label='Actual treatment year')
    axes[1].set_title('RMSPE ratio distribution')
    axes[1].set_xlabel('Post/Pre RMSPE')
    axes[1].legend(loc='best', fontsize=8)

    fig.suptitle(f'In-time placebo inference: {country_label}')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    return tidy_df, effect_df, str(csv_path), str(fig_path)


def run_in_time_placebo_nz(output_dir='article_assets'):
    df = pd.read_csv('inter/nz.csv').rename(columns={'GDP per capita': 'GDP_per_capita'})
    covar_cols = [
        'Agriculture',
        'Manufacturing',
        'Construction',
        'Financial and Insurance Services',
        'Rental, Hiring and Real Estate Services',
        'Tertiary',
    ]
    covar_cols = [col for col in covar_cols if col in df.columns]
    return run_in_time_placebo_inference(
        df=df,
        treated='Canterbury',
        actual_treatment_year=2011,
        placebo_years=list(range(2000, 2010)),
        covar_cols=covar_cols,
        outcome_col='GDP_per_capita',
        baseline_start_year=2000,
        country_label='NZ',
        output_dir=output_dir,
    )


def run_in_time_placebo_chile(output_dir='article_assets'):
    df = pd.read_csv('inter/processed_chile.csv').rename(columns={'year': 'Year', 'region_name': 'Region', 'gdp_cap': 'GDP_per_capita'})
    covar_cols = ['agropecuario', 'industria_m', 'construccion', 'comercio', 'servicios_financieros']
    covar_cols = [col for col in covar_cols if col in df.columns]
    return run_in_time_placebo_inference(
        df=df,
        treated='VII Del Maule',
        actual_treatment_year=2010,
        placebo_years=list(range(1990, 2010)),
        covar_cols=covar_cols,
        outcome_col='GDP_per_capita',
        baseline_start_year=1990,
        country_label='Chile',
        output_dir=output_dir,
    )

def jackknife_tests(df, treated, pre_years, covar_cols, outcome_col='GDP_per_capita'):
    # Select donors once to determine the donor set
    donors, weights = select_donors(df, treated, pre_years, covar_cols)
    results = []
    years = sorted(df['Year'].unique())
    y_treated = df[df['Region'] == treated].set_index('Year').reindex(years)[outcome_col].astype(float).values
    for excluded in donors:
        sub_donors = [d for d in donors if d != excluded]
        # recompute weights from distances using the sub-donor set
        cov_treated = compute_covariates(df, treated, pre_years, covar_cols)
        distances = {}
        for region in sub_donors:
            cov_reg = compute_covariates(df, region, pre_years, covar_cols)
            distances[region] = np.linalg.norm(cov_reg - cov_treated)
        inv = np.array([1 / distances[r] for r in sub_donors])
        w = inv / inv.sum()
        # synthetic series for sub-donor set
        series = {}
        for region in df['Region'].unique():
            s = df[df['Region'] == region].set_index('Year').reindex(years)[outcome_col].astype(float).values
            series[region] = s
        y_synth_sub = np.zeros(len(years), dtype=float)
        for i, reg in enumerate(sub_donors):
            y_synth_sub += w[i] * series[reg]
        gap = y_treated - y_synth_sub
        results.append({'Excluded': excluded, 'Gap': gap})
    return years, results

def placebo_tests(df, treated, pre_years, covar_cols, outcome_col='GDP_per_capita'):
    # treat each control region as pseudo-treated
    regions = df['Region'].dropna().unique().tolist()
    placebo_results = []
    years = sorted(df['Year'].unique())
    for pseudo in regions:
        if pseudo == treated:
            continue
        # donors for pseudo-treated exclude that region
        donors = [r for r in regions if r != pseudo]
        # compute covariates for pseudo-treated
        cov_pseudo = compute_covariates(df, pseudo, pre_years, covar_cols)
        distances = {}
        for r in donors:
            cov_reg = compute_covariates(df, r, pre_years, covar_cols)
            distances[r] = np.linalg.norm(cov_reg - cov_pseudo)
        # select donors (still five or fewer)
        sorted_regions = sorted(distances, key=lambda r: distances[r])
        selected = sorted_regions[:min(5, len(sorted_regions))]
        inv = np.array([1 / distances[r] for r in selected])
        w = inv / inv.sum()
        # synthetic series
        series = {}
        for region in regions:
            s = df[df['Region'] == region].set_index('Year').reindex(years)[outcome_col].astype(float).values
            series[region] = s
        y_pseudo = series[pseudo]
        y_synth = np.zeros(len(years), dtype=float)
        for i, reg in enumerate(selected):
            y_synth += w[i] * series[reg]
        gap = y_pseudo - y_synth
        # compute mean gap in pre- and post-periods
        pre_gap = gap[[i for i, y in enumerate(years) if y in pre_years]].mean()
        post_gap = gap[[i for i, y in enumerate(years) if y not in pre_years]].mean()
        placebo_results.append({'Region': pseudo, 'PreGap': pre_gap, 'PostGap': post_gap})
    return placebo_results


def sector_level_plot(df, treated, selected, weights, sector_cols, event_year, title_prefix, outfile_prefix):
    years = sorted(df['Year'].unique())
    sector_series_treated = {}
    sector_series_synth = {col: np.zeros(len(years), dtype=float) for col in sector_cols}
    # Build series for each region
    for region in df['Region'].unique():
        sub = df[df['Region'] == region].set_index('Year').reindex(years)
        for col in sector_cols:
            if region == treated:
                sector_series_treated.setdefault(col, []).append(sub[col].astype(float).values)
            else:
                pass
    # For treated region we have a list of arrays for each sector
    for col in sector_cols:
        # there should be only one array
        sector_series_treated[col] = sector_series_treated[col][0]
    # Build synthetic sector series
    for i, region in enumerate(selected):
        sub = df[df['Region'] == region].set_index('Year').reindex(years)
        for col in sector_cols:
            sector_series_synth[col] += weights[i] * sub[col].astype(float).values
    # Plot each sector
    for col in sector_cols:
        plt.figure(figsize=(10,4))
        plt.plot(years, sector_series_treated[col], label=f'{title_prefix} (Actual)')
        plt.plot(years, sector_series_synth[col], label=f'{title_prefix} (Synthetic)', linestyle='--')
        plt.axvline(event_year, color='k', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel(col)
        plt.title(f'{title_prefix}: {col} share')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{outfile_prefix}_{col}.png')
        plt.close()

# Main function to perform analysis and generate robustness plots

def run_full_analysis_nz():
    df = pd.read_csv('nz_dataset_final.csv')
    df = df.rename(columns={'Year ended March': 'Year'})
    treated = 'Canterbury'
    pre_years = list(range(2000, 2011))
    covar_cols = ['Primary_share', 'Manufacturing_share', 'Construction_share', 'Services_share', 'Utilities_share', 'HigherDegreePct']
    selected, weights = select_donors(df, treated, pre_years, covar_cols)
    years, y_treated, y_synth = synthetic_series(df, treated, selected, weights)
    # Plot overall GDP per capita
    plt.figure(figsize=(10,5))
    plt.plot(years, y_treated, label='Canterbury (Actual)')
    plt.plot(years, y_synth, label='Synthetic Canterbury', linestyle='--')
    plt.axvline(2011, color='k', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('GDP per capita (NZD)')
    plt.title('Canterbury vs Synthetic Control')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nz_canterbury_synthetic.png')
    plt.close()
    # Jackknife
    years_j, jack_results = jackknife_tests(df, treated, pre_years, covar_cols)
    # Plot gaps for jackknife
    for res in jack_results:
        plt.plot(years_j, res['Gap'], label=f'Excl {res["Excluded"]}')
    plt.axvline(2011, color='k', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Gap (Actual - Synth)')
    plt.title('Jackknife Gaps for Canterbury')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nz_canterbury_jackknife_gaps.png')
    plt.close()
    # Placebo tests summary
    placebo_res = placebo_tests(df, treated, pre_years, covar_cols)
    placebo_df = pd.DataFrame(placebo_res)
    placebo_df.to_csv('nz_canterbury_placebos.csv', index=False)
    # Sector-level plots
    sector_cols = ['Primary_share', 'Manufacturing_share', 'Services_share']
    sector_level_plot(df, treated, selected, weights, sector_cols, 2011, 'Canterbury', 'nz_canterbury_sector')
    return selected, weights

def run_full_analysis_chile():
    df = pd.read_csv('chile_regional_data_1990_2019.csv')
    treated = 'Maule'
    pre_years = list(range(1990, 2010))
    covar_cols = ['ag_share', 'man_share', 'serv_share']
    selected, weights = select_donors(df, treated, pre_years, covar_cols)
    years, y_treated, y_synth = synthetic_series(df, treated, selected, weights)
    plt.figure(figsize=(10,5))
    plt.plot(years, y_treated, label='Maule (Actual)')
    plt.plot(years, y_synth, label='Synthetic Maule', linestyle='--')
    plt.axvline(2010, color='k', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('GDP per capita (CLP)')
    plt.title('Maule vs Synthetic Control')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cl_maule_synthetic.png')
    plt.close()
    years_j, jack_results = jackknife_tests(df, treated, pre_years, covar_cols, outcome_col='GDP_per_capita')
    for res in jack_results:
        plt.plot(years_j, res['Gap'], label=f'Excl {res["Excluded"]}')
    plt.axvline(2010, color='k', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Gap (Actual - Synth)')
    plt.title('Jackknife Gaps for Maule')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cl_maule_jackknife_gaps.png')
    plt.close()
    placebo_res = placebo_tests(df, treated, pre_years, covar_cols)
    placebo_df = pd.DataFrame(placebo_res)
    placebo_df.to_csv('cl_maule_placebos.csv', index=False)
    # Sector-level plots (agriculture, manufacturing and services)
    sector_cols = ['ag_share', 'man_share', 'serv_share']
    sector_level_plot(df, treated, selected, weights, sector_cols, 2010, 'Maule', 'cl_maule_sector')
    return selected, weights

if __name__ == '__main__':
    run_full_analysis_nz()
    run_full_analysis_chile()
