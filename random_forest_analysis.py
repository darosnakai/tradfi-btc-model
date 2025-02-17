import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

#this file is the random forest model, containing bothj single and multivariable anlyses
#these two approaches were done in order to compare it more accurately to the linear regression models...
#...but also to capture different aspects of the relationship between Bitcoin and TradFi assets

class RandomForestAnalyzer: 
    def __init__(self):
        #setting the parameters for the random forest model
        self.rf_params = {'n_estimators': 100,'max_depth': 10,'random_state': 42}

    def run_single_var_analysis(self, btc_data, tradfi_data, feature, timeframe, asset):
        #running random forest analysis for single variable pairs
        #notice we are also doing cross validation
        try:
            #aligning data, as usual
            aligned_data = pd.concat([
                btc_data[feature], 
                tradfi_data[feature]
            ], axis=1, join='inner')
            
            if len(aligned_data) == 0:
                raise ValueError("No overlapping dates found between datasets")

            #preparing data. usage of sklearn requires us to do this
            X = aligned_data.iloc[:, 1].values.reshape(-1, 1)  
            y = aligned_data.iloc[:, 0].values 

            #splitting data (80-20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            #training model
            model = RandomForestRegressor(**self.rf_params)
            
            #using cross-validation, in order to provide more robust R2 scores
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            r2_cv = cv_scores.mean()
            
            #fitting the model on the training set
            model.fit(X_train, y_train)
            #making predictions on the test set
            y_pred = model.predict(X_test)
            
            #calculating metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = max(0, r2_score(y_test, y_pred))
            
            #using cross-validated R2 if the R2 is negative
            if r2 < 0:
                r2 = max(0, r2_cv)

            return {'timeframe': timeframe,'asset': asset,'feature': feature,
                'r2': r2,'mse': mse,'rmse': rmse}

        except Exception as e:
            print(f"Error in single variable analysis: {str(e)}")
            return None

    def run_multivar_analysis(self, features_df, analysis_type, timeframe):
        try:
            #handling NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.dropna()

            #separating features from btc and tradfi
            btc_cols = [col for col in features_df.columns if col.startswith('BTC_')]
            tradfi_cols = [col for col in features_df.columns if not col.startswith('BTC_')]
            
            if not btc_cols or not tradfi_cols:
                raise ValueError("No valid features found")

            #preparing data
            X = features_df[tradfi_cols]
            y = features_df[btc_cols[0]] 

            #splitting data (80-20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            #training model
            model = RandomForestRegressor(**self.rf_params)
            
            #using cross-validation for more robust R2
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            r2_cv = cv_scores.mean()
            
            #fitting on training test
            model.fit(X_train, y_train)
            #making predictions on testing data
            y_pred = model.predict(X_test)
            
            #calculating metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = max(0, r2_score(y_test, y_pred))
            
            #using cross-validated R2 in case R2 is negative
            if r2 < 0:
                r2 = max(0, r2_cv)

            return {'timeframe': timeframe,'analysis_type': analysis_type,
                'r2': r2,'mse': mse,'rmse': rmse}

        except Exception as e:
            print(f"Error in multivariate analysis - {analysis_type}: {str(e)}")
            return None

    def analyze_timeframe(self, timeframe):
        #method to analyze both single and multivariable analyses
        print(f"\nAnalyzing {timeframe} timeframe...")
        
        single_var_results = []
        multivar_results = []

        #singlevariable analysis
        features = ['Price', 'Open', 'High', 'Low', 'volume', 'price_change', 'range']
        assets = ['NYSE', 'DXY', 'Brent', 'Gold']

        for asset in assets:
            #loading standardized data
            btc_df = pd.read_csv(f'standardized_btc_{timeframe}.csv')
            tradfi_df = pd.read_csv(f'standardized_{asset.lower()}_{timeframe}.csv')
            #notice all timeframes will be encompassed
            
            #converting date to datetime and using as index
            for df in [btc_df, tradfi_df]:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            #analyzing each feature individually
            for feature in features:
                result = self.run_single_var_analysis(btc_df, tradfi_df, feature, timeframe, asset)
                if result:
                    single_var_results.append(result)

        #multivariable analysis
        print("Loading feature-engineered data...")
        returns_df = pd.read_csv(f'standardized_returns_{timeframe}.csv')
        volatility_df = pd.read_csv(f'standardized_volatility_{timeframe}.csv')
        ma_df = pd.read_csv(f'standardized_ma_{timeframe}.csv')
        rsi_df = pd.read_csv(f'standardized_rsi_{timeframe}.csv')

        #setting date as index
        for df in [returns_df, volatility_df, ma_df, rsi_df]:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        #running different analyses
        analyses = {'returns': returns_df,'volatility': volatility_df,'moving_averages': ma_df,'rsi': rsi_df}

        for analysis_type, df in analyses.items():
            #iterating through all analyses defined in 'analyses' variable
            result = self.run_multivar_analysis(df, analysis_type, timeframe)
            if result:
                multivar_results.append(result)

        #running combined analysis (5th and last, combining all engineered features)
        combined_df = pd.concat([df for df in analyses.values()], axis=1)
        
        #removing duplicate columns
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        result = self.run_multivar_analysis(combined_df, 'combined', timeframe)
        if result:
            multivar_results.append(result)

        #saving results (separately)
        if single_var_results:
            pd.DataFrame(single_var_results).to_csv(
                f'rf_single_var_results_{timeframe}.csv', index=False
            )
        if multivar_results:
            pd.DataFrame(multivar_results).to_csv(
                f'rf_multivar_results_{timeframe}.csv', index=False
            )

        return single_var_results, multivar_results

def main():
    analyzer = RandomForestAnalyzer()
    timeframes = ['daily', 'weekly', 'monthly']
    
    all_single_var = []
    all_multivar = []
    
    for timeframe in timeframes:
        single_results, multi_results = analyzer.analyze_timeframe(timeframe)
        if single_results:
            all_single_var.extend(single_results)
        if multi_results:
            all_multivar.extend(multi_results)
    
    #saving all results together
    if all_single_var:
        pd.DataFrame(all_single_var).to_csv('rf_single_var_results_all.csv', index=False)
    if all_multivar:
        pd.DataFrame(all_multivar).to_csv('rf_multivar_results_all.csv', index=False)
    
    print("\nRandom Forest analysis complete. Results saved to CSV files.")

if __name__ == "__main__":
    main()