# Performance Optimization Summary

## Changes Made for Sub-30 Minute Runtime

### 1. Data Configuration
- **Tickers**: Reduced from 3 (AAPL, MSFT, GOOGL) to 1 (AAPL only)
- **History**: Reduced from 7 years to 5 years
- **Min History Requirement**: Reduced from 5 to 3 years

### 2. Feature Engineering
- **Technical Indicators**: Removed "stochastic" (5 indicators instead of 6)
- **Lag Days**: Reduced from [1, 5, 10, 21] to [1, 5, 10] (3 instead of 4)
- **Rolling Windows**: Reduced from [5, 10, 21, 63] to [5, 10, 21] (3 instead of 4)

### 3. LSTM Model
- **Lookback Window**: Reduced from 60 to 30 days
- **Forecast Horizons**: Reduced from [1, 5, 30] to [1, 5] (removed 30-day forecast)
- **Units**: Reduced from [64, 64] to [32, 32] (smaller network)
- **Batch Size**: Increased from 32 to 64 (faster training)
- **Epochs**: Reduced from 50 to 10 (5x faster)
- **Patience**: Reduced from 5 to 3 (earlier stopping)

### 4. ARIMA Model
- **Max P/Q**: Reduced from 5 to 3
- **Parameter Search**: Limited to max 2 for p/q (9 combinations instead of 25)
- **Seasonal Search**: Reduced to only 2 combinations [(0,0,0), (1,0,1)] instead of full grid
- **Optimizers**: Reduced from 4 to 2 (lbfgs, bfgs only)
- **Max Iterations**: Reduced from 200/500/1000 to 100

### 5. Traditional ML Models
- **Grid Search**: Eliminated - all parameters now have single values
  - Random Forest: n_estimators=100, max_depth=10, min_samples_leaf=2
  - Gradient Boosting: n_estimators=100, learning_rate=0.1, max_depth=3
  - SVR: kernel='rbf', C=10.0, gamma='scale', epsilon=0.1
- **Training**: Direct fit instead of GridSearchCV (instant training)

### 6. Backtesting
- **Walk Forward Window**: Reduced from 252 to 126 days (half year)
- **Rebalance Frequency**: Increased from 5 to 10 days (fewer backtests)

### 7. Logging
- **Level**: Changed from DEBUG to INFO (reduces log noise and I/O overhead)

## Expected Runtime Breakdown

- **Data Loading**: ~30 seconds
- **Feature Engineering**: ~1-2 minutes
- **LSTM Training** (2 horizons × 10 epochs): ~5-8 minutes
- **ARIMA Training** (limited search): ~3-5 minutes
- **Traditional ML** (no grid search): ~1-2 minutes
- **Backtesting**: ~2-3 minutes
- **Visualization**: ~1 minute

**Total Estimated Time: 15-25 minutes** (well under 30-minute target)

## Trade-offs

- **Accuracy**: Slightly reduced due to smaller models and less hyperparameter tuning
- **Coverage**: Single ticker instead of three
- **Forecast Horizons**: No 30-day forecast
- **Model Complexity**: Simpler models overall

These optimizations maintain the core functionality while dramatically reducing runtime.


