import pandas as pd
import ta

class StandardizedFeatureGenerator:
    def __init__(self):
        #creating windows for each timeframe
        #notice the windows will contain a the same relative period of time (short, medium and long-term),...
        #... ensuring compatibility between features
        self.windows = {
            'daily': { 'volatility': [20, 50, 150], 'ma': [20, 50, 150] },
            'weekly': { 'volatility': [4, 10, 30], 'ma': [4, 10, 30] },
            'monthly': { 'volatility': [2, 4, 12], 'ma': [2, 4, 12]}
        }
    
    def generate_features(self, standardized_df, asset_name, timeframe):
        #method to generate features based on the standardized dataframes created on the preprocessing part
        features = pd.DataFrame(index=standardized_df.index)
        
        #standardized prices, fetching only the price column of each standardized_df
        std_prices = standardized_df['Price']
        
        #log return (price movement)
        #diff function computes the difference between consecutive elements in the series
        features[f'{asset_name}_log_returns'] = std_prices.diff()
        
        
        #volatility over certain windows (market behavior)
        for window in self.windows[timeframe]['volatility']:
            rolling_std = std_prices.diff().rolling(window=window).std()
            #rolling() function creates a rolling window that allows me to do calculations over this specified window
            #std() function (called on the rolling window object) calculats the standard deviation of the values within each rolling window
            #in other words, finding the std of a rolling window made of the log returns
            features[f'{asset_name}_volatility_{window}'] = rolling_std
        
        #relative strength indicator (RSI - technical indicator)
        features[f'{asset_name}_rsi'] = ta.momentum.RSIIndicator( std_prices + abs(std_prices.min()) + 1 ).rsi()
        
        #moving average and price-to-moving-average spread (technical indicators)
        for window in self.windows[timeframe]['ma']:
            ma = std_prices.rolling(window=window).mean()
            features[f'{asset_name}_ma_{window}'] = ma
            features[f'{asset_name}_price_to_ma_{window}'] = std_prices - ma
        
        return features
    
    def process_timeframe(self, timeframe):
        #processing each asset for every timeframe (daily, weekly, monthly)
        print(f"\nProcessing {timeframe} timeframe...")
        
        #(initially empty) dataframe that will contain all newly generated features
        all_features = pd.DataFrame()
        
        for asset in ['BTC', 'NYSE', 'DXY', 'Brent', 'Gold']:
            
            #loading the standardized dataframe and setting the index to datetime
            df = pd.read_csv(f'standardized_{asset.lower()}_{timeframe}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            #generating engineered features using hte generate_features method
            print(f"Generating features for {asset}...")
            features = self.generate_features(df, asset, timeframe)
            
            #adding newly generated features to the dataframe (that was initially empty)
            all_features = pd.concat([all_features, features], axis=1)
        
        
        #saving features using the method save_feature_types
        self.save_feature_types(all_features, timeframe)
        
        return all_features
    
    def save_feature_types(self, features_df, timeframe):
        #saving features separately by type and timeframe
        #each df will have specific features from a particular timeframe
        
        #log returns
        returns_cols = [col for col in features_df.columns if 'log_returns' in col]
        features_df[returns_cols].to_csv(f'standardized_returns_{timeframe}.csv')
        
        #volatility
        vol_cols = [col for col in features_df.columns if 'volatility' in col]
        features_df[vol_cols].to_csv(f'standardized_volatility_{timeframe}.csv')
        
        #rsi
        rsi_cols = [col for col in features_df.columns if 'rsi' in col]
        features_df[rsi_cols].to_csv(f'standardized_rsi_{timeframe}.csv')
        
        #ma and price-to-ma spread
        ma_cols = [col for col in features_df.columns if 'ma' in col]
        features_df[ma_cols].to_csv(f'standardized_ma_{timeframe}.csv')

def main():
    #generating features from all timestamps based on standardized data
    generator = StandardizedFeatureGenerator()
    
    for timeframe in ['daily', 'weekly', 'monthly']:
        generator.process_timeframe(timeframe)
    
    print("\nFeature generation from standardized data complete.")

if __name__ == "__main__":
    main()