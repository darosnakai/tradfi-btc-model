import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#this file is making the use of linear regression, using more than one dependent variable...
#...for analyzing bitcoin's correlation with tradfi assets
#in this file I am not using engineered features, only standard features from the 'clean' datasets

class MultivarAnalyzer:
    def __init__(self):
        self.results = {}
        
    def preprocess_data(self, df):
        #preprocessing (data is already standardized)
        #creating copy to avoide modifying the original dataset
        df_clean = df.copy()
        
        #this part is only converting date and setting to index to make sure df is correct, as its already standardized
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean.set_index('Date', inplace=True)
        
        return df_clean

    def align_datasets(self, *dataframes):
        #finding common dates accross all dataframs and returning the aligned versions
        common_dates = dataframes[0].index
        for df in dataframes[1:]:
            common_dates = common_dates.intersection(df.index)

        #will return only aligned versions of all dataframes, in which dates are all matching,
        #this ensures data compatibility for further analysis
        return [df.loc[common_dates] for df in dataframes]

    def run_regression(self, X, y, feature_names):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,random_state=42,)
        #fitting regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        #making prediction based on fitted model
        y_pred = model.predict(X_test)
        
        #calculating metrics (r2, mse, rmse)
        mse = mean_squared_error(y_test, y_pred)
        
        results = {'r2': r2_score(y_test, y_pred),'mse': mse,'rmse': np.sqrt(mse),
            'coefficients': dict(zip(feature_names, model.coef_)), 
            #pairing feature name with its corresponding coefficient and converting these pairs into dictionary to optimize usage in the future
            'intercept': model.intercept_}
        
        return results

    def analyze_all(self, btc_df, nyse_df, dxy_df, brent_df, gold_df, timeframe):
        #preprocessing method
        dfs = {'BTC': self.preprocess_data(btc_df),'NYSE': self.preprocess_data(nyse_df),'DXY': self.preprocess_data(dxy_df),
            'Brent': self.preprocess_data(brent_df),'Gold': self.preprocess_data(gold_df)}
        #assignining each df to its accoridng variable to optimize the analysis

        #aligning dates
        aligned_dfs = self.align_datasets(*dfs.values())
        btc, nyse, dxy, brent, gold = aligned_dfs
        
        results = {}
        
        #price-based features analysis
        X_all_features = pd.DataFrame()
        feature_names = []
        #combining all price-based features into a single dataframe
        for asset, df in [('NYSE', nyse), ('DXY', dxy), ('Brent', brent), ('Gold', gold)]:
            for feature in ['Price', 'Open', 'High', 'Low']:
                col_name = f'{asset}_{feature}'
                X_all_features[col_name] = df[feature]
                feature_names.append(col_name)
                #this loop will extract the specified features (Price, OPen, High, Low) and will add them to the x_all_features dataframe
                #it also maintians the names of features (ie NYSE_price or Brent_Low) for reference
        
        #running the model using the price-based features generated with the btc price feature as the dependent variable
        results['all_features'] = self.run_regression(X_all_features, btc['Price'], feature_names)
        
        #closing price features analysis
        X_closing = pd.DataFrame({'NYSE_Price': nyse['Price'],'DXY_Price': dxy['Price'],'Brent_Price': brent['Price'],'Gold_Price': gold['Price']})
        #creating dataframe only containing closing prices of tradfi assets
        
        #running the model based using the closnig prices of tradfi assets and btc closing price as 
        results['closing_prices'] = self.run_regression(X_closing, btc['Price'], X_closing.columns)
        
        #price range features analysis 
        X_ranges = pd.DataFrame({'NYSE_Range': nyse['High'] - nyse['Low'],'DXY_Range': dxy['High'] - dxy['Low'],
            'Brent_Range': brent['High'] - brent['Low'],'Gold_Range': gold['High'] - gold['Low']})
        #notice price range is calculated by the difference between high and low price of all assets
        
        #running the model based on ranges
        results['ranges'] = self.run_regression(X_ranges, btc['High'] - btc['Low'], X_ranges.columns)
        
        #paired market analysis (using nyse and gold features only)
        X_paired = pd.DataFrame()
        paired_features = []
        for asset, df in [('NYSE', nyse), ('Gold', gold)]:
            for feature in ['Price', 'Open', 'High', 'Low']:
                col_name = f'{asset}_{feature}'
                X_paired[col_name] = df[feature]
                paired_features.append(col_name)
                #similar to the loop in the price-based features part, this will extract all price-based features from gold and the nyse
        
        results['paired_markets'] = self.run_regression(X_paired, btc['Price'], paired_features)
        
        return results

    def save_results(self, results, timeframe):
        #saving metrics
        metrics_data = []
        for analysis_type, result in results.items():
            metrics_data.append({'timeframe': timeframe,'analysis_type': analysis_type,'r2': result['r2'],
                'mse': result['mse'],'rmse': result['rmse'],'intercept': result['intercept']})
        
        #saving coefficients
        coef_data = []
        for analysis_type, result in results.items():
            for feature, coef in result['coefficients'].items():
                coef_data.append({'timeframe': timeframe,'analysis_type': analysis_type,
                    'feature': feature,'coefficient': coef})
        
        #saving to newly created csv files
        pd.DataFrame(metrics_data).to_csv(f'multivar_metrics_{timeframe}.csv', index=False)
        pd.DataFrame(coef_data).to_csv(f'multivar_coefficients_{timeframe}.csv', index=False)

def main():
    analyzer = MultivarAnalyzer()
    timeframes = ['daily', 'weekly', 'monthly']
    
    for timeframe in timeframes:
        print(f"\nProcessing {timeframe} data...")
        
        #loading standardized data
        btc_df = pd.read_csv(f'standardized_btc_{timeframe}.csv')
        nyse_df = pd.read_csv(f'standardized_nyse_{timeframe}.csv')
        dxy_df = pd.read_csv(f'standardized_dxy_{timeframe}.csv')
        brent_df = pd.read_csv(f'standardized_brent_{timeframe}.csv')
        gold_df = pd.read_csv(f'standardized_gold_{timeframe}.csv')
        #notice the csvs were all saved like this in the generator file, to ensure efficiency in fetching them
        
        #running analysis method
        results = analyzer.analyze_all(btc_df, nyse_df, dxy_df, brent_df, gold_df, timeframe)
        
        #saving results method
        analyzer.save_results(results, timeframe)
        print(f"Results saved for {timeframe}")

if __name__ == "__main__":
    main()