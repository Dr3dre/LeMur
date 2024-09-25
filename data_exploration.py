import pandas as pd
import warnings

if __name__ == '__main__':
    # Files
    articles_file_path = 'lista_articoli.xlsx'
    history_file_path = 'storico_filtrato.xlsx'

    # Read
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        articles_df = pd.read_excel(articles_file_path, engine="openpyxl")
        history_df = pd.read_excel(history_file_path, engine="openpyxl")
    
    '''
    Data cleaning
    '''
    # Store apart indexes of rows indicating an year / month switch
    year_switches = history_df[history_df.iloc[:,1].str.contains("Anno Inizio", case=False, na=False)].iloc[:,1].index.tolist()
    month_switches = history_df[history_df.iloc[:,2].str.contains("Mese Inizio", case=False, na=False)].iloc[:,2].index.tolist()
    # Store the starting year and month
    year_start = history_df.iloc[year_switches[0]].iloc[1]
    month_start = history_df.iloc[month_switches[0]].iloc[2]
    print(year_start)
    print(month_start)
    
    # Re-arrange the dataframe
    history_df = history_df.iloc[2:, 3:]
    history_df.rename(columns={history_df.columns[0]: 'MC'}, inplace=True)
    history_df['MC'] = history_df['MC'].fillna(-1).astype(int)
    print(history_df)