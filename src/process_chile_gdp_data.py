import pandas as pd

REGION_MAPPING = {
        'GDP per capita I De Tarapacá Growth Rate': 'I De Tarapacá',
        'GDP per capita II De Antofagasta Growth Rate': 'II De Antofagasta',
        'GDP per capita III De Atacama Growth Rate': 'III De Atacama',
        'GDP per capita IV De Coquimbo Growth Rate': 'IV De Coquimbo',
        'GDP per capita V De Valparaíso Growth Rate': 'V De Valparaíso',
        'GDP per capita RMS Región Metropolitana de Santiago Growth Rate': 'RMS Región Metropolitana de Santiago',
        'GDP per capita VI Del Libertador General Bernardo OHiggins Growth Rate': 'VI Del Libertador General Bernardo OHiggins',
        'GDP per capita VII Del Maule Growth Rate': 'VII Del Maule',
        'GDP per capita VIII Del Biobío Growth Rate': 'VIII Del Biobío',
        'GDP per capita IX De La Araucanía Growth Rate': 'IX De La Araucanía',
        'GDP per capita X De Los Lagos Growth Rate': 'X De Los Lagos',
        'GDP per capita XI Aysén del General Carlos Ibáñez del Campo Growth Rate': 'XI Aysén del General Carlos Ibáñez del Campo',
        'GDP per capita XII De Magallanes y de la Antártica Chilena Growth Rate': 'XII De Magallanes y de la Antártica Chilena'
    }

REGIONS = [
        'I De Tarapacá', 'II De Antofagasta', 'III De Atacama',
        'IV De Coquimbo', 'V De Valparaíso',
        'RMS Región Metropolitana de Santiago',
        'VI Del Libertador General Bernardo OHiggins', 'VII Del Maule',
        'VIII Del Biobío', 'IX De La Araucanía', 'X De Los Lagos',
        'XI Aysén del General Carlos Ibáñez del Campo',
        'XII De Magallanes y de la Antártica Chilena'
    ]

REVERSED_REGION_MAPPING = {
        '2.Región de Tarapacá': 'I De Tarapacá',
        '3.Región de Antofagasta': 'II De Antofagasta',
        '4.Región de Atacama': 'III De Atacama',
        '5.Región de Coquimbo': 'IV De Coquimbo',
        '6.Región de Valparaíso': 'V De Valparaíso',
        '7.Región Metropolitana': 'RMS Región Metropolitana de Santiago',
        '8.Región del Libertador General Bernardo O\'Higgins': 'VI Del Libertador General Bernardo OHiggins',
        '9.Región del Maule': 'VII Del Maule',
        '10.Región del Biobío': 'VIII Del Biobío',
        '12.Región de La Araucanía': 'IX De La Araucanía',
        '14.Región de Los Lagos': 'X De Los Lagos',
        '15.Región de Aysén del General Carlos Ibáñez del Campo': 'XI Aysén del General Carlos Ibáñez del Campo',
        '16.Región de Magallanes y la Antártica Chilena': 'XII De Magallanes y de la Antártica Chilena'
    }

GDP_COLUMNS_MAPPING = {
        'PIB Región de Antofagasta': 'II De Antofagasta',
        'PIB Región de Atacama': 'III De Atacama',
        'PIB Región de Coquimbo': 'IV De Coquimbo',
        'PIB Región de Valparaíso': 'V De Valparaíso',
        'PIB Región Metropolitana de Santiago': 'RMS Región Metropolitana de Santiago',
        'PIB Región del Libertador General Bernardo OHiggins': 'VI Del Libertador General Bernardo OHiggins',
        'PIB Región del Maule': 'VII Del Maule',
        'PIB Región de La Araucanía': 'IX De La Araucanía',
        'PIB Región de Aysén del General Carlos Ibáñez del Campo': 'XI Aysén del General Carlos Ibáñez del Campo',
        'PIB Región de Magallanes y de la Antártica Chilena': 'XII De Magallanes y de la Antártica Chilena',
        'I de Tarapacá': 'I De Tarapacá',
        'VIII Del Biobío': 'VIII Del Biobío',
        'X De Los Lagos': 'X De Los Lagos'
    }

def read_updated_gdp_chile():
    file_path = '../data/pib sectorial y regional.xlsx'
    # Correctly load the dataset again, this time ensuring to skip only the first 2 rows
    df = pd.read_excel(file_path, skiprows=2)

    selected_columns = ['Periodo'] + [col for col in df.columns if col.startswith('PIB Reg')]
    df = df[selected_columns]
    def shorten_column_names(col):
        return col.split(',')[0]

# Apply the function to rename the columns
    df.columns = [shorten_column_names(col) for col in df.columns]
    df['Year'] = pd.to_datetime(df['Periodo']).dt.year
    df = df.drop('Periodo', axis=1)
    df = df[1:]
    df = agg_regions(df)
    return df


def agg_regions(df):
    df['I de Tarapacá'] = df['PIB Región de Arica y Parinacota'] + df['PIB Región de Tarapacá']
    df['VIII Del Biobío'] = df['PIB Región de Ñuble'] + df['PIB Región del Biobío']
    df['X De Los Lagos'] = df['PIB Región  de los Ríos'] + df['PIB Región de Los Lagos']
    df = df.drop(['PIB Región de Arica y Parinacota', 'PIB Región de Tarapacá', 'PIB Región de Ñuble',
                  'PIB Región del Biobío', 'PIB Región  de los Ríos', 'PIB Región de Los Lagos'], axis=1)
    return df


def read_updated_population_chile():
    file_path = '../data/poblacion regional.xlsx'
    # Correctly load the dataset again, this time ensuring to skip only the first 2 rows
    df = pd.read_excel(file_path, skiprows=2)
    # Apply the function to rename the columns
    df['Year'] = pd.to_datetime(df['Periodo']).dt.year
    df = df.drop('Periodo', axis=1)
    df = df.rename(columns=REVERSED_REGION_MAPPING)
    df['I De Tarapacá'] = df['1.Región de Arica y Parinacota'] + df['I De Tarapacá']
    df['VIII Del Biobío'] = df['11.Región de Ñuble'] + df['VIII Del Biobío']
    df['X De Los Lagos'] = df['13.Región de Los Ríos'] + df['X De Los Lagos']
    df = df.drop(['1.Región de Arica y Parinacota', '11.Región de Ñuble', '13.Región de Los Ríos'], axis=1)
    return df


def read_and_clean_data():
    # Read the datasets
    gdp = read_updated_gdp_chile()
    pop = read_updated_population_chile()

def clean_gdp_column_names(df):
    df = df.rename(columns=GDP_COLUMNS_MAPPING)
    return df


def calculate_gdp_per_capita():
    gdp = read_updated_gdp_chile()
    pop = read_updated_population_chile()
    gdp_clean = clean_gdp_column_names(gdp)
    # Merge datasets on 'Year'
    merged_df = pd.merge(gdp_clean, pop, on='Year', suffixes=('_gdp', '_pop'))
    # Calculate GDP per capita
    gdp_per_capita = pd.DataFrame()
    gdp_per_capita['Year'] = merged_df['Year']
    # Calculate GDP per capita for each region
    for region in REGIONS:
        gdp_per_capita['GDP per capita ' + region] = merged_df[region + "_gdp"] * 10E8 / merged_df[region + "_pop"]
    return gdp_per_capita


def calculate_growth_rates():
    ch = calculate_gdp_per_capita()
    for col in ch.columns[1:]:
        ch[f'{col} Growth Rate'] = ch[col].pct_change() * 100
    gr = ch.filter(like='Growth').dropna()
    gr['year'] = range(2014, 2024)
    return gr


def fix_encoding(text):
    return text.encode('latin1').decode('utf-8')


def _population_long_format():
    """Return population data in long format (region_name, Year, Population)."""
    pop = read_updated_population_chile()
    pop_long = pop.melt(id_vars=['Year'], var_name='region_name', value_name='Population')
    pop_long = pop_long.rename(columns={'Year': 'year'})
    return pop_long


def process_data_for_synth():
    df = pd.read_excel('../data/scm_chile_2010.xlsx')
    # Apply the function to the column with apply
    df['region_name'] = df['region_name'].apply(fix_encoding)
    ch = calculate_growth_rates()
    # Melt the DataFrame to long format
    df_long = ch.melt(id_vars=['year'], var_name='region_name', value_name='growth_rate')
    # Map to the new region names
    df_long['region_name'] = df_long['region_name'].map(REGION_MAPPING)
    df_long = df_long[df_long['year'] > 2015]
    # Merge the dataframes
    merged_df = pd.merge(df, df_long, on=['year', 'region_name'], how='outer')
    # Sort by region and year
    merged_df = merged_df.sort_values(by=['region_name', 'year']).reset_index(drop=True)
    # Forward fill the gdp_cap data
    for i in range(len(merged_df)):
        if pd.isnull(merged_df.loc[i, 'gdp_cap']):
            previous_year_gdp = merged_df.loc[i - 1, 'gdp_cap']
            growth_rate = merged_df.loc[i, 'growth_rate']
            merged_df.loc[i, 'gdp_cap'] = previous_year_gdp * (1 + growth_rate / 100)
    # Add Population and gdp_total (gdp_cap * Population) for denominator-effect SCM
    pop_long = _population_long_format()
    merged_df = pd.merge(merged_df, pop_long, on=['year', 'region_name'], how='left')
    merged_df['gdp_total'] = merged_df['gdp_cap'] * merged_df['Population']
    return merged_df