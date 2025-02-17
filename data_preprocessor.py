import pandas as pd
from sklearn.preprocessing import StandardScaler

#this python code will be used for preprocessing of the data sets
#I transformed the volume datapoints into numbers (K, M, and B)
#I standardized all the datasets (mean = 0 and std = 1) in order to facilitate analysis...
#..This has to be done before the analysis of any kind to make it more accurate


class DataPreprocessor:
    def __init__(self):
        #scaler method to standardize all datasets
        self.scaler = StandardScaler()
        #columns (features) of all datasets. they all contain the same columns
        self.numerical_columns = ['Price', 'Open', 'High', 'Low', 'volume', 'price_change', 'range']

    def process_vol(self, vol_series):
        #this function will handle volume (originally in string), converting K, M and B to their proper float values
        def convert_volume(x):
            if pd.isna(x) or x == '':
                return 0.0
            if isinstance(x, (int, float)):
                return float(x)
            x = str(x).upper().strip()
            try:
                if 'K' in x:
                    return float(x.replace('K', '')) * 1000
                elif 'M' in x:
                    return float(x.replace('M', '')) * 1000000
                elif 'B' in x:
                    return float(x.replace('B', '')) * 1000000000
                else:
                    return float(x.replace(',', ''))
            except:
                return 0.0
        
        return vol_series.apply(convert_volume)

    def standardize_dataframe(self, df):
        #function to standardize all dataframes in order to facilitate data analysis
        df_standardized = df.copy()
        
        #fetching numerical columns that exist in this dataframe
        #this is important to reduce the chance of an error, ensuring we are standardizing only columns that actually exist in the datasets
        cols_to_standardize = [col for col in self.numerical_columns if col in df.columns]
        
        if cols_to_standardize:
            #fitting the data using the scaler and transforming it
            scaled_data = self.scaler.fit_transform(df_standardized[cols_to_standardize])
            
            #fit_transform is a built-in function from the StandardScaler library
            #it is responsible for computing the mean and std dev. of each column and transforming them using the z-score formula
            #this will create standardized values (with mean = 0 and std = 1)
            
            #now we are adding the standardized values to the new data_frame
            df_standardized[cols_to_standardize] = scaled_data
            
        return df_standardized

    def preprocess_pair(self, btc_df, tradfi_df):
        #here i am cleaning all the columns and transforming them into float, and also aligning dates of dfs
        #the reason for why I am using a pair here is to make sure all tradfi assets have the same dates (index) as the btc
        #this includes the volume part, as well as the price change (initially as a string with %)
        #this is crucial, as at the end I will use the standardize_dataframe method in this cleaned data 

        #creating a copy in order to not mess with the actual, original datasets
        btc_clean = btc_df.copy()
        tradfi_clean = tradfi_df.copy()

        #convert all numeric data to float 
        for df in [btc_clean, tradfi_clean]:
            
            #converting price columns to float
            price_cols = ['Price', 'Open', 'High', 'Low']
            df[price_cols] = df[price_cols].replace(',', '', regex=True).astype(float)

            #converting price change to decimal
            df['price_change'] = df['Change %'].str.rstrip('%').astype(float) / 100

            #process volume using the volume method we created earlier
            df['volume'] = self.process_vol(df['Vol.'])

            #calculating range, adding a new feature to the new dataset
            df['range'] = df['High'] - df['Low']

            #converting date into datatime and seting it as index
            #this is important for organizing datasets and aligning them with each other
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            #dropping original string columns
            df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

        #aligning dates between both datasets
        common_dates = btc_clean.index.intersection(tradfi_clean.index)

        #using the aligned dates for the new index of the datasets
        btc_clean = btc_clean.loc[common_dates]
        tradfi_clean = tradfi_clean.loc[common_dates]

        #standardizing numerical features using the standardize_dataframe method created earlier
        btc_standardized = self.standardize_dataframe(btc_clean)
        tradfi_standardized = self.standardize_dataframe(tradfi_clean)

        return btc_standardized, tradfi_standardized

    def preprocess_and_save_all(self, timeframe):
        #saving and processing standardized datasets
        #taking timeframe as a parameter, which will be part of the name of the files of the datasets
        print(f"Processing {timeframe} data...")
        
        #loading data
        #I have parametrized the names of the csvs in order to facilitate this part of the code
        btc_df = pd.read_csv(f'btc_{timeframe}.csv')
        nyse_df = pd.read_csv(f'NYSE_{timeframe}.csv')
        dxy_df = pd.read_csv(f'DXY_{timeframe}.csv')
        brent_df = pd.read_csv(f'Brent_{timeframe}.csv')
        gold_df = pd.read_csv(f'Gold_{timeframe}.csv')

        
        #using dictionary to fetch the dataframes of each asset
        assets = {'BTC': btc_df, 'NYSE': nyse_df,'DXY': dxy_df,'Brent': brent_df,'Gold': gold_df}
        
        processed_data = {}
        for asset_name, df in assets.items():
            if asset_name == 'BTC':
                continue
            #will process all datasets inside the assets dictionary

            #using preprocess_pair method to align the dates of datasets
            btc_processed, asset_processed = self.preprocess_pair(btc_df, df)
            processed_data[asset_name] = asset_processed
            
            if asset_name == 'NYSE': #saving only the first btc processed data (the one with NYSE)
                processed_data['BTC'] = btc_processed
        
        #btc will be processed multiple times but only saved once (when processed with NYSE)

        #save standardized data to a new file
        for asset_name, df in processed_data.items():
            output_path = f'standardized_{asset_name.lower()}_{timeframe}.csv'
            
            #saving it to a csv file
            df.to_csv(output_path)
            print(f"Saved standardized data to {output_path}")
        
        return processed_data

def main():
    #preprocessing all datasets for all timeframes
    preprocessor = DataPreprocessor()
    timeframes = ['daily', 'weekly', 'monthly']
    
    for timeframe in timeframes:
        preprocessor.preprocess_and_save_all(timeframe)
        print(f"Completed processing {timeframe} data")

if __name__ == "__main__":
    main()