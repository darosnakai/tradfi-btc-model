import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


#this pythong file will analyze the engineered features using linear regression
#this linear regression will be used individually against the corresponding feature from each TradFi asset
#similar to the singlevar_analyzer, but here I am making the use of engineered features

class SingleVarfeAnalyzer:
    def __init__(self):
        pass
        #class does not need any attributes as it will create results as it processes the data
    
    def clean_data(self, X, y):
        #cleaning - removing NaN and infinite values 
        combined = pd.concat([X, y], axis=1)#combining features (x) and target (y) into one dataframe for better cleaning
        combined = combined.replace([np.inf, -np.inf], np.nan) #converting infinite values to NaN
        combined = combined.dropna()#removing NaN values
        
        y_clean = combined[y.name]
        X_clean = combined.drop(y.name, axis=1)
        
        removed_rows = len(y) - len(y_clean)
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows containing NaN or inf values")
        
        return X_clean, y_clean
    
    def create_lagged_features(self, df, lag_periods=[1, 2, 3]):
        #creating lagged version of features
        lagged_df = df.copy()
        
        for col in df.columns:
            for lag in lag_periods:
                lagged_df[f"{col}_lag{lag}"] = df[col].shift(lag)
        
        return lagged_df.dropna()
    
    def run_matched_regression(self, X, y, feature_type, asset_type, timeframe):
        #running regressions

        X_clean, y_clean = self.clean_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2,random_state=42,)
        
        
        if len(X_clean) == 0:
            print(f"Error: No valid data remaining after cleaning for {feature_type} analysis")
            return None, None
        
        try:
            #fitting model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            #calculation of core metrics (r2, mse, rmse)
            mse = mean_squared_error(y_test, y_pred)
            
            #storing results 
            metrics = {'feature_type': feature_type,'asset_type': asset_type,'timeframe': timeframe,
                'r2': r2_score(y_test, y_pred),'mse': mse,'rmse': np.sqrt(mse)}
            
            #storing feature coefficients
            feature_stats = pd.DataFrame({'coefficient': model.coef_,'feature_type': feature_type,
                'asset_type': asset_type,'timeframe': timeframe},index=X_clean.columns)
            
            return metrics, feature_stats
            
        except Exception as e:
            print(f"Error in regression analysis for {feature_type}: {str(e)}")
            return None, None
    
    def analyze_timeframe(self, timeframe):
        #method to analyze features based on the timeframes
        print(f"\nAnalyzing {timeframe} timeframe...")
        
        metrics_list = []
        feature_stats_list = []
        
        try:
            #loading engineered features (created beforehand)
            #they will be used later on for matching the regression 
            returns_df = pd.read_csv(f'standardized_returns_{timeframe}.csv')
            volatility_df = pd.read_csv(f'standardized_volatility_{timeframe}.csv')
            trend_df = pd.read_csv(f'standardized_ma_{timeframe}.csv')
            
            #setting date as index as usual
            for df in [returns_df, volatility_df, trend_df]:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            #analyzing each tradfi asset separately
            for asset in ['NYSE', 'DXY', 'Brent', 'Gold']:
                print(f"\nAnalyzing {asset}...")
                
                #log returns analysis

                y_returns = returns_df['BTC_log_returns']
                X_returns = returns_df[[f'{asset}_log_returns']]
                #matching features - log returns from both Bitcoin and TradFi assets

                #using the lagged features creator method
                X_returns_lagged = self.create_lagged_features(X_returns)

                #using the matched regression method
                metrics, stats = self.run_matched_regression(X_returns_lagged, y_returns, 'returns', asset, timeframe)
                
                #ensuring metrics and stats are stored into the list
                if metrics and stats is not None:
                    metrics_list.append(metrics)
                    feature_stats_list.append(stats)
                
                #volatility analysis
                #identifying columns that contain volatility data of Bitcoin (and the specific TradFi asset we are analzing)
                #notice volatilty_df contains volatility measures for ALL assets (BTC, NYSE, DXY, Brent and Gold)...
                #...so here we are specifying only the particular tradfi asset we are evaluating
                btc_vol_cols = [col for col in volatility_df.columns if col.startswith('BTC_volatility')][0]
                asset_vol_cols = [col for col in volatility_df.columns if col.startswith(f'{asset}_volatility')]
                
                #using the columns with volatility features
                y_vol = volatility_df[btc_vol_cols]
                X_vol = volatility_df[asset_vol_cols]
                #matching features - volatility features from both Bitcoin and TradFi assets

                X_vol_lagged = self.create_lagged_features(X_vol)
                
                metrics, stats = self.run_matched_regression(
                    X_vol_lagged, y_vol, 'volatility', asset, timeframe
                )
                if metrics and stats is not None:
                    metrics_list.append(metrics)
                    feature_stats_list.append(stats)
                
                #trend (price-to-ma spread) analysis
                price_to_ma_cols = [col for col in trend_df.columns if 'price_to_ma' in col]
                
                #same as the volatility
                btc_trend = trend_df[[col for col in price_to_ma_cols if col.startswith('BTC_')][0]]
                asset_trend = trend_df[[col for col in price_to_ma_cols if col.startswith(f'{asset}_')]]
                #matching features - trend features from both Bitcoin and TradFi assets

                X_trend_lagged = self.create_lagged_features(asset_trend)
                
                metrics, stats = self.run_matched_regression(
                    X_trend_lagged, btc_trend, 'trend', asset, timeframe
                )
                if metrics and stats is not None:
                    metrics_list.append(metrics)
                    feature_stats_list.append(stats)
            
            #saving results to new csv
            if metrics_list and feature_stats_list:
                pd.DataFrame(metrics_list).to_csv(f'singlevar_fe_metrics_{timeframe}.csv', index=False)
                pd.concat(feature_stats_list).to_csv(f'singlevar_fe_coefficients_{timeframe}.csv')
            
            return metrics_list, feature_stats_list
            
        except Exception as e:
            print(f"Error analyzing {timeframe} timeframe: {str(e)}")
            return [], []

def main():
    analyzer = SingleVarfeAnalyzer()
    timeframes = ['daily', 'weekly', 'monthly']
    
    all_metrics = []
    all_feature_stats = []
    
    #processing each timeframe using the analyze_timeframe method created earlier
    for timeframe in timeframes:
        metrics_list, feature_stats_list = analyzer.analyze_timeframe(timeframe)
        if metrics_list and feature_stats_list:
            all_metrics.extend(metrics_list)
            all_feature_stats.extend(feature_stats_list)
    
    # Save combined results
    if all_metrics and all_feature_stats:
        #saving regression metrics
        pd.DataFrame(all_metrics).to_csv('singlevar_fe_metrics_all.csv', index=False)
        
        #saving regression coefficients
        pd.concat(all_feature_stats).to_csv('singlevar_fe_coefficients_all.csv')
        print("\nAnalysis complete. Results saved to CSV files.")
    else:
        print("\nNo valid results to save.")

if __name__ == "__main__":
    main()