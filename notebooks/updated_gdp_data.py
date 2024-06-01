import pandas as pd

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
    #df = agg_regions(df)
    reversed_new_region_names = {
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

    df = df.rename(columns=reversed_new_region_names)
    df['I De Tarapacá'] = df['1.Región de Arica y Parinacota'] + df['I De Tarapacá']
    df['VIII Del Biobío'] = df['11.Región de Ñuble'] + df['VIII Del Biobío']
    df['X De Los Lagos'] = df['13.Región de Los Ríos'] + df['X De Los Lagos']
    df = df.drop(['1.Región de Arica y Parinacota', '11.Región de Ñuble', '13.Región de Los Ríos'], axis=1)
    return df

def read_and_clean_data():
    # Read the datasets
    gdp = read_updated_gdp_chile()
    pop = read_updated_population_chile()

# Clean and align the column names to keep only the region numbers and the 'Year' column
def clean_gdp_column_names(df):
    df = df.rename(columns={
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
    })
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
    regions = [
        'I De Tarapacá', 'II De Antofagasta', 'III De Atacama',
        'IV De Coquimbo', 'V De Valparaíso',
        'RMS Región Metropolitana de Santiago',
        'VI Del Libertador General Bernardo OHiggins', 'VII Del Maule',
        'VIII Del Biobío', 'IX De La Araucanía', 'X De Los Lagos',
        'XI Aysén del General Carlos Ibáñez del Campo',
        'XII De Magallanes y de la Antártica Chilena'
    ]

    for region in regions:
        gdp_per_capita['GDP per capita ' + region] = merged_df[region + "_gdp"] * 10E8 / merged_df[region + "_pop"]
    return gdp_per_capita

