import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Print area cannot be set to Defined name")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

SECTORIAL_GDP_VARIABLES = ['Agriculture', 'Administrative and Support Services', 'Construction', 
                        'Education and Training', 'Financial and Insurance Services', 
                        'Food and beverage services', 'Health Care and Social Assistance', 
                        'Information Media, Telecommunications and Other Services', 
                        'Manufacturing', 'Professional, Scientific, and Technical Services', 
                        'Public Administration and Safety', 'Rental, Hiring and Real Estate Services', 
                        'Retail Trade', 'Transport, Postal and Warehousing', 'Wholesale Trade']


def clean_data_for_synthetic_control():
    df = read_and_clean_nz_data()
    pop = read_and_clean_nz_population_data()    
    df_wide = pd.pivot_table(df, index=['Period', 'Region'], columns='Industry', values='value').reset_index()
    df_wide.rename(columns={'Period': 'Year'}, inplace=True)
    merged_df = pd.merge(df_wide, pop, on=['Year', 'Region'], how='outer')
    merged_df.drop(columns=['Total All Industries'], inplace=True)
    merged_df.rename(columns={'Total': 'GDP per capita'}, inplace=True)
    rv = order_data_and_calculate_per_capita(merged_df)
    rv = add_tertiary_education_data(rv)
    rv['gdp_per_capita'] = rv['GDP per capita']
    rv['total_population'] = rv['Population']
    rv['total_gdp'] = rv['Gross Domestic Product']
    return rv


def read_and_clean_nz_data():
    file_path = os.path.join(_DATA_DIR, 'regional-gross-domestic-product-year-ended-march-2023.csv')
    df = pd.read_csv(file_path)
    df_cleaned = df[['Period', 'Data_value', 'MAGNTUDE','Subject', 'Group', 'Series_title_2', 'Series_title_3']]    
    rv = df_cleaned.rename(columns={
    'Series_title_3': 'Industry',
    'Series_title_2': 'Region',
    'Data_value': 'value'
    }).drop(columns=['Subject']).copy(deep=True)
    rv['Industry'] = rv['Industry'].fillna('Total')
    rv['Period'] = rv['Period'].astype(int)
    return rv

def read_and_clean_nz_population_data():
    spreadsheet_path = os.path.join(_DATA_DIR, 'regional-gross-domestic-product-year-ended-march-2023.xlsx')
    table_3 = pd.read_excel(spreadsheet_path, sheet_name='Table 3')
    start_row = table_3.index[table_3.iloc[:, 0] == "Region"].tolist()[0]

    # Load the data again from the correct starting row, and set the header accordingly
    table_3_data = pd.read_excel(spreadsheet_path, sheet_name='Table 3', skiprows=start_row)

    # Rename the columns to reflect the years correctly
    table_3_data.columns = ['Region'] + table_3_data.iloc[0, 1:].tolist()

    # Drop the first two rows as they are headers
    table_3_data = table_3_data.drop([0, 1]).reset_index(drop=True)

    # Keep only regions and the total for the country, exclude island totals
    regions_to_keep = [
        'Northland', 'Auckland', 'Waikato', 'Bay of Plenty', 'Gisborne', "Hawke's Bay",
        'Taranaki', 'Manawatū-Whanganui', 'Wellington' ,'Total North Island',
        'Tasman / Nelson(2)', 'Marlborough', 'West Coast', 'Canterbury(3)', 'Otago',
        'Southland', 'Total South Island', 'Total New Zealand'
    ]
    # Filter the dataframe
    filtered_data = table_3_data[table_3_data['Region'].isin(regions_to_keep)]
    population_df = pd.melt(filtered_data, id_vars=['Region'], var_name='Year', value_name='Population')
    population_df['Year'] = population_df['Year'].astype(str).str.replace(r'\D', '', regex=True)
    
    # Convert 'Year' to integers
    population_df['Population'] = population_df['Population'].astype(float)
    population_data_cleaned = population_df.replace({
    'Region': {
        'Manawatū-Whanganui': 'Manawatu-Whanganui',
        'Tasman / Nelson(2)': 'Tasman/Nelson',
        'Canterbury(3)': 'Canterbury',
        'Total New Zealand': 'New Zealand'
    }
    })
    population_data_cleaned['Year'] = population_data_cleaned['Year'].apply(lambda x: str(x)[:-1] if len(str(x)) == 5 else str(x))
    population_data_cleaned['Year'] = population_data_cleaned['Year'].astype(int)
    return population_data_cleaned


def order_data_and_calculate_per_capita(df):
    # Select the columns related to the specific sectors mentioned
    # Rearrange the columns so that the selected columns are at the end
    reordered_columns = ['Year', 'Region', 'Gross Domestic Product', 'GDP per capita', 'Population'] + SECTORIAL_GDP_VARIABLES
    # Create the reordered dataframe
    df_reordered = df[reordered_columns].copy(deep=True)
    df_reordered[SECTORIAL_GDP_VARIABLES] = df_reordered[SECTORIAL_GDP_VARIABLES].div(df_reordered['Gross Domestic Product'], axis=0)
    return df_reordered

def read_and_process_tertiary_education_data():
    # Define the region mapping
    ter = pd.read_csv(os.path.join(_DATA_DIR, 'nz_tertiary_attainment.csv'))
    region_mapping = {
        'northland': 'Northland',
        'auckland': 'Auckland',
        'waikato': 'Waikato',
        'bayofplenty': 'Bay of Plenty',
        'gisborne': 'Gisborne',
        'hawke\'sbay': 'Hawke\'s Bay',
        'taranaki': 'Taranaki',
        'Manawatu-Wanganui': 'Manawatu-Whanganui',
        'wellington': 'Wellington',
        'tasmannelson': 'Tasman/Nelson',
        'Marlborough': 'Marlborough',
        'westcoast': 'West Coast',
        'canterbury': 'Canterbury',
        'otago': 'Otago',
        'southland': 'Southland'
    }

    ter['Region'] = ter['Region'].map(region_mapping)
    return ter

def add_tertiary_education_data(data):
    ter = read_and_process_tertiary_education_data()
    df = data.merge(ter.drop('regioncode', axis=1), on=['Region', 'Year'], how='left')
    regioncodes = ter[['regioncode', 'Region']].drop_duplicates().copy(deep=True)
    df = df.merge(regioncodes, on='Region', how='left')
    return df
