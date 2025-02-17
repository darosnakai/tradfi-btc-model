import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#this file will make use of a linear regression model in order to find relationship between single variables accross Bitcoin and TradFi assets
#it will analyse all variables of Bitcoin individually against all the same variables of tradfi assets
#for example (BTC price change X DXY price change; BTC volume X Brent volume)...
#important for us to understand how each asset, individually, may or may not impact Bitcoin

class SingleVarAnalyzer:
    def __init__(self):
        self.results = {}
        self.features = ['Price', 'Open', 'High', 'Low', 'volume', 'price_change', 'range']

    def analyze_relationship(self, btc_data, tradfi_data, feature):
        #relationship between matched features (btc open x nyse open), using standardized data
        try:
            if feature not in btc_data.columns or feature not in tradfi_data.columns:
                raise ValueError(f"Feature '{feature}' not found in data")
            
            #creating a new dataframe with matched features from both assets (BTC open price x NYSE open price), in order to analyze them later on
            #these will be used for creating the linear regression model, as we are comparing matched features only
            aligned_data = pd.concat([btc_data[feature],tradfi_data[feature]], axis=1, join='inner')

            #preparing the data
            #reshape transforms the data into the format required by scikit learn module
            X = aligned_data.iloc[:, 1].values.reshape(-1, 1)  #tradfi
            y = aligned_data.iloc[:, 0].values.reshape(-1, 1)  #btc
            #using aligned_data here as we will evaluate the correlation between one tradfi asset against btc (same feature on both)

            #splitting data for linear regression analysis
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            #notice test data is 0.2 (20%) of the dataset

            #fitting model using scikit learn module WHY???
            model = LinearRegression()
            model.fit(X_train, y_train)

            #test predictions
            y_pred_test = model.predict(X_test)

            #metrics of the predicted data compared to the actual (test) data
            return {
                'r2_test': r2_score(y_test, y_pred_test),
                'mse_test': mean_squared_error(y_test, y_pred_test),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'coefficient': model.coef_[0][0],
                'intercept': model.intercept_[0],
                'n_samples': len(aligned_data)
            }
        
        except Exception as e:
            print(f"Error in analyze function: {str(e)}")
            return None

    def analyze_all(self, timeframe, asset_name):
        #loading data
        btc_df = pd.read_csv(f'standardized_btc_{timeframe}.csv')
        tradfi_df = pd.read_csv(f'standardized_{asset_name.lower()}_{timeframe}.csv')
        
        #converting data to datetime and setting as index
        for df in [btc_df, tradfi_df]:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        #storing results in a hierarchial structure
        if timeframe not in self.results:
            self.results[timeframe] = {}
        if asset_name not in self.results[timeframe]:
            self.results[timeframe][asset_name] = {}

        #analyzing each matched feature one by one, from all tradfi assets in all timeframes
        for feature in self.features:
            results = self.analyze_relationship(btc_df, tradfi_df, feature)
            if results:
                self.results[timeframe][asset_name][feature] = results

        return self.results[timeframe][asset_name]

    def get_results(self, as_dataframe=True):
        #getting results 
        if not as_dataframe:
            return self.results
        
        rows = []
        
        #transforming dictionary into a list of rows, in order to convert to a pandas df and saving it to csv later on
        for timeframe in self.results:
            for asset in self.results[timeframe]:
                for feature in self.results[timeframe][asset]:
                    row = {
                        'timeframe': timeframe,
                        'asset': asset,
                        'feature': feature,
                        #** will unpack all metrics into the row dictionary
                        **self.results[timeframe][asset][feature]
                    }
                    rows.append(row)
        return pd.DataFrame(rows)

def main():
    #running the methods
    analyzer = SingleVarAnalyzer()
    timeframes = ['daily', 'weekly', 'monthly']
    assets = ['NYSE', 'DXY', 'Brent', 'Gold']
    
    #iterates each timeframe and ensures all combinations are analyzed accordingly
    for timeframe in timeframes:
        print(f"\nAnalyzing {timeframe} data...")
        for asset in assets:
            print(f"Analyzing {asset}...")
            analyzer.analyze_all(timeframe, asset)
    
    #saving all results to csv
    results_df = analyzer.get_results()
    results_df.to_csv('single_var_regression_results.csv', index=False)
    print("\nResults saved to 'single_var_regression_results.csv'")

if __name__ == "__main__":
    main()