import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#this python file will analyze several engineered features altogether...
#... in order to draw conclusions regarding bitcoin's correlation with tradfi assets
#this model will make the use of engineered features

class MultiVarfeAnalyzer:
    def __init__(self):
        pass
    
    def clean_data(self, X, y):
        #cleaning - removing NaN and infinite values 
        combined = pd.concat([X, y], axis=1) #combining features (x) and target (y) into one dataframe for better cleaning
        combined = combined.replace([np.inf, -np.inf], np.nan)#converting infinite values to NaN
        combined = combined.dropna()#removing NaN values
        
        y_clean = combined[y.name]
        X_clean = combined.drop(y.name, axis=1)
        
        return X_clean, y_clean

    #this method will be used further on the code. important to do it because will facilitate the model training part
    def run_regression(self, X, y, analysis_name, timeframe):
        #cleaning data
        X_clean, y_clean = self.clean_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2,random_state=42,)

        try:
            #fitting model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            #calculating etrics
            mse = mean_squared_error(y_test, y_pred)
            
            #metrics dictionary
            metrics = {'analysis_name': analysis_name,'timeframe': timeframe,
                'r2': r2_score(y_test, y_pred),'mse': mse,'rmse': np.sqrt(mse)}
            
            #recording coefficients in dictionary as well
            feature_stats = pd.DataFrame({'coefficient': model.coef_,'analysis_name': analysis_name,
                'timeframe': timeframe}, index=X_clean.columns)
            
            return metrics, feature_stats
        
        except Exception as e:
            print(f"Error in regression analysis for {analysis_name}: {str(e)}")
            return None, None

    def analyze_timeframe(self, timeframe):
        #main part of the code
        #here each analysis will be made and saved based on the timeframe
        print(f"\nAnalyzing {timeframe} timeframe...")
        
        metrics_list = []
        feature_stats_list = []
        
        try:
            #loading engineered features
            returns_df = pd.read_csv(f'standardized_returns_{timeframe}.csv')
            volatility_df = pd.read_csv(f'standardized_volatility_{timeframe}.csv')
            trend_df = pd.read_csv(f'standardized_ma_{timeframe}.csv')
            rsi_df = pd.read_csv(f'standardized_rsi_{timeframe}.csv')
            #notice how the csvs are organized to optimize the code
            #since I am comparing feature by feature, this will make it more efficient
            
            #setting date as index, as usual
            for df in [returns_df, volatility_df, trend_df, rsi_df]:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            #all tradfi returns
            print("Running all TradFi returns analysis...")
            #dependent variable selection
            y_returns = returns_df['BTC_log_returns']
            #independent variables selection
            X_returns = returns_df[[col for col in returns_df.columns 
                                  if col.startswith(('NYSE_', 'DXY_', 'Brent_', 'Gold_'))]]
            #notice I am getting from the returns column already, so I am only selecting the regressors here
            metrics, stats = self.run_regression(X_returns, y_returns, 
                                              'all_tradfi_returns', timeframe)
            #regression is run using the method created earlier
            
            #will only add the metrics and feature stats if they were successfully calculated by regression analysis
            if metrics and stats is not None:
                metrics_list.append(metrics)
                feature_stats_list.append(stats)
            
            #all tradfi volatility
            print("Running all TradFi volatility analysis...")
            #dependent variable selection
            btc_vol = volatility_df[[col for col in volatility_df.columns 
                                   if col.startswith('BTC_volatility')][0]]
            #independent variable selection
            X_vol = volatility_df[[col for col in volatility_df.columns 
                                 if col.startswith(('NYSE_', 'DXY_', 'Brent_', 'Gold_'))]]
            #selecting only the necessary features for each analysis.
            #since the features are separated in eadch dataset, its easier for us to choose 
            #getting the results from the volatility_df already, so only selecting the regressors
            metrics, stats = self.run_regression(X_vol, btc_vol, 
                                              'all_tradfi_volatility', timeframe)
            
            if metrics and stats is not None:
                metrics_list.append(metrics)
                feature_stats_list.append(stats)
            
            #tradfi trend analysis (price_to_ma)
            print("Running all TradFi trend analysis...")
            #dependent variable
            btc_trend = trend_df[[col for col in trend_df.columns 
                                if col.startswith('BTC_') and 'price_to_ma' in col][0]]
            #independent variable
            X_trend = trend_df[[col for col in trend_df.columns 
                              if any(asset in col for asset in ['NYSE_', 'DXY_', 'Brent_', 'Gold_'])
                              and 'price_to_ma' in col]]
            #using the trend_df already 
            metrics, stats = self.run_regression(X_trend, btc_trend, 
                                              'all_tradfi_trend', timeframe)
            if metrics and stats is not None:
                metrics_list.append(metrics)
                feature_stats_list.append(stats)
            
            #tradfi rsi analysis
            print("Running all TradFi RSI analysis...")
            #dependent variable
            btc_rsi = rsi_df['BTC_rsi']
            #independent variable
            X_rsi = rsi_df[[col for col in rsi_df.columns 
                           if col.startswith(('NYSE_', 'DXY_', 'Brent_', 'Gold_'))]]
            metrics, stats = self.run_regression(X_rsi, btc_rsi,
                                             'all_tradfi_rsi', timeframe)
            if metrics and stats is not None:
                metrics_list.append(metrics)
                feature_stats_list.append(stats)
            
            #all combined features
            print("Running combined features analysis...")
            combined_features = pd.concat([
                X_returns,  
                X_vol,      
                X_trend,    
                X_rsi       
            ], axis=1)
            #notice they were already calculated for the past analyses, therefore I am just using them again
            
            #target is Bitcoin's log returns
            y_combined = returns_df['BTC_log_returns'] 
            metrics, stats = self.run_regression(combined_features, y_combined, 
                                              'combined_features', timeframe)
            if metrics and stats is not None:
                metrics_list.append(metrics)
                feature_stats_list.append(stats)
            
            #savig results
            if metrics_list and feature_stats_list:
                pd.DataFrame(metrics_list).to_csv(
                    f'multivar_fe_metrics_{timeframe}.csv', 
                    index=False
                )
                pd.concat(feature_stats_list).to_csv(
                    f'multivar_fe_coefficients_{timeframe}.csv'
                )
            
            return metrics_list, feature_stats_list
            
        except Exception as e:
            print(f"Error analyzing {timeframe} timeframe: {str(e)}")
            return [], []

def main():
    regressor = MultiVarfeAnalyzer()
    timeframes = ['daily', 'weekly', 'monthly']
    
    all_metrics = []
    all_feature_stats = []
    
    #runing regression timeframe by timeframe
    for timeframe in timeframes:
        metrics_list, feature_stats_list = regressor.analyze_timeframe(timeframe)
        if metrics_list and feature_stats_list:
            all_metrics.extend(metrics_list)
            all_feature_stats.extend(feature_stats_list)
    
    #saving results
    if all_metrics and all_feature_stats:
        pd.DataFrame(all_metrics).to_csv('multivar_fe_metrics_all.csv', index=False)
        pd.concat(all_feature_stats).to_csv('multivar_fe_coefficients_all.csv')
        print("\nAnalysis complete. Results saved to CSV files.")
    else:
        print("\nNo valid results to save.")

if __name__ == "__main__":
    main()