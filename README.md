# Financial Modeling with Bitcoin
🔭 Linear and Non-Linear Model Comparative Analysis to evaluate the correlation between Bitcoin and Traditional Markets. 

📑 **Overview of the Project:** In this project, I examined the relationship between Bitcoin and Traditional Markets by analyzing Bitcoin's correlation to financial indicators that are crucial for global markets: the NYSE Composite Index, U.S. Dollar Index (DXY), Brent Crude Oil, and Gold. I made a model comparative analysis between Linear (Simple and Multiple Linear Regression) and Non-Linear (Random Forest) Machine Learning models to examine whether traditional market indicators can effectively predict Bitcoin price movements and to evaluate the degree of Bitcoin's integration with established financial markets.

📌 **Purpose:** This project aimed to further my understanding of Machine Learning models in Python by using data that I am particularly familiar with, the financial markets. I have always been interested in financial markets and Bitcoin, and developing a predictive model seemed like an interesting challenge for my personal development. This project has proven invaluable in helping me better understand quantitative finance and machine learning integration. The skills and methodologies I learned directly apply to various areas of financial modeling that align with my professional interests. Throughout the process, I developed several skills: understanding the importance of data preprocessing and standardization of all datasets for better results; developing a pipeline to optimize the use of datasets and get better results; developing reusable functions to make the code more efficient; deep understanding of essential modeling metrics (R2, MSE, RMSE, MAE, f-test); importance of feature selection and how to implement feature engineering.

🗂️ **Features Used:** The analysis examines standard and engineered market features. 
- Standard Features (price-based and volume features):
    - Closing Price; Opening Price; High Price; Low Price; Price Range (High - Low); Price Change Percentage and Trading Volume (standardized).
- Engineered features (return, volatility, and momentum-based features):
    - Returns feature included Log Returns (calculated from standardized prices)
    - Volatility features with daily windows (20, 50, 150 days), weekly windows (4, 10, 30 weeks), and monthly windows (2, 4, 12 months)
    - Momentum Features include Moving Averages (using the same windows as volatility) and Price-to-MA Spread (spread (difference) of price to MA)

📊 **Analysis Pipeline:**
1. Data collection and data preprocessing, including but not limited to:
- Sourced all datasets from Investing.com to maintain a consistent data structure
- Standardization of all features
2. Feature engineering, including but not limited to:
- Generating sophisticated features from standardized data
3. Model Implementation and Evaluation:
- Linear Regression (simple and multiple) and Random Forest models
- R² for explanatory power, MSE for prediction accuracy, and RMSE for scale-appropriate error measurement
4. Results Analysis and Implementation:
- Comparative analysis across models, timeframes, datasets, and features

🖋️ **Interesting Takeaways:** Results reveal significant correlations between Bitcoin and traditional markets, particularly with the NYSE Composite Index and Gold, suggesting Bitcoin's dual role as both a speculative investment and a potential store of value. The Random Forest model demonstrated superior predictive power in shorter timeframes, while linear regression models showed comparable or better performance over longer horizons.

🗒️ **Next Steps:** As a continuation of this project, I would like to develop other models, including ARCH/GARCH, ANNs, and LSTM, in order to enhance my analysis and have a more robust understanding of modeling techniques in finance. I will also use higher frequency data, such as hourly, minute-to-minute, and second-to-second data, to have more observations for analysis, and engineer other features in order to enhance my understanding of feature engineering.
