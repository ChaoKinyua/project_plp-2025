"""
Example: Integration with live trading broker (Alpaca)

This shows how to connect your analysis pipeline to actual trading.
Install: pip install alpaca-trade-api

DO NOT USE REAL MONEY WITHOUT THOROUGH TESTING IN PAPER TRADING
"""

import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Example 1: Alpaca Broker Integration
# ============================================================================

def execute_with_alpaca(predictions: Dict[str, float], config: Optional[Dict] = None):
    """
    Execute trades based on predictions.
    
    Args:
        predictions: {"AAPL": 0.65, "MSFT": -0.42}  (signal strength)
        config: {
            "paper_trading": True,
            "position_size": 100,  # dollars per trade
            "confidence_threshold": 0.55
        }
    
    Example:
        predictions = {
            "AAPL": 0.65,      # 65% confidence UP
            "MSFT": -0.42,     # 42% confidence DOWN
            "GOOGL": 0.51      # 51% confidence UP
        }
        execute_with_alpaca(predictions, config)
    """
    try:
        from alpaca_trade_api import REST
    except ImportError:
        logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
        return
    
    config = config or {}
    paper_trading = config.get("paper_trading", True)
    position_size = config.get("position_size", 100)
    confidence_threshold = config.get("confidence_threshold", 0.55)
    
    # Initialize Alpaca client
    api = REST(
        base_url="https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets",
        api_version='v2'
    )
    
    # Get current account info
    account = api.get_account()
    logger.info(f"Account balance: ${account.portfolio_value:.2f}")
    
    # Process each prediction
    for ticker, signal in predictions.items():
        confidence = abs(signal)
        direction = "BUY" if signal > 0 else "SELL"
        
        if confidence < confidence_threshold:
            logger.info(f"{ticker}: Signal {signal:.2f} below threshold {confidence_threshold}")
            continue
        
        try:
            # Example: Simple order execution
            if signal > confidence_threshold:
                # BUY signal
                quantity = int(position_size / 100)  # Simple position sizing
                logger.info(f"Sending BUY order: {quantity} shares of {ticker}")
                
                # Uncomment to execute (DANGEROUS - test in paper trading first!)
                # order = api.submit_order(
                #     symbol=ticker,
                #     qty=quantity,
                #     side='buy',
                #     type='market',
                #     time_in_force='day'
                # )
                # logger.info(f"Order submitted: {order.id}")
                
            elif signal < -confidence_threshold:
                # SELL signal
                quantity = int(position_size / 100)
                logger.info(f"Sending SELL order: {quantity} shares of {ticker}")
                
                # Uncomment to execute
                # order = api.submit_order(
                #     symbol=ticker,
                #     qty=quantity,
                #     side='sell',
                #     type='market',
                #     time_in_force='day'
                # )
                # logger.info(f"Order submitted: {order.id}")
                
        except Exception as e:
            logger.error(f"Failed to trade {ticker}: {e}")
    
    logger.info("Trading execution complete")


# ============================================================================
# Example 2: Extract Predictions from Pipeline
# ============================================================================

def extract_predictions_from_pipeline(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """
    Convert model metrics to trading signals.
    
    Args:
        metrics_df: DataFrame with columns [Model, RMSE, MAE, Sharpe, Direction_Accuracy]
    
    Returns:
        {"AAPL": 0.63}  (confidence in BUY direction)
    
    Example:
        metrics = pd.read_csv("data/processed/AAPL_metrics.csv")
        signal = extract_predictions_from_pipeline(metrics)
        print(signal)  # {"AAPL": 0.63}
    """
    
    # Strategy: Use ensemble average of all models
    if metrics_df.empty:
        return {}
    
    # Weight by model quality (lower RMSE = higher weight)
    metrics_df = metrics_df.copy()
    metrics_df['weight'] = 1 / (metrics_df['RMSE'] + 0.001)
    metrics_df['weight'] = metrics_df['weight'] / metrics_df['weight'].sum()
    
    # Get directional accuracy (assume > 50% confidence is positive)
    metrics_df['signal'] = (metrics_df['Directional Accuracy'] - 0.5) * 2
    
    # Weighted ensemble signal
    ensemble_signal = (metrics_df['signal'] * metrics_df['weight']).sum()
    
    return {"signal": ensemble_signal}


# ============================================================================
# Example 3: Risk Management
# ============================================================================

class PortfolioRiskManager:
    """
    Manage portfolio risk - enforce limits to prevent catastrophic losses.
    """
    
    def __init__(self, max_position_size: float = 10000, max_loss_per_trade: float = 500):
        self.max_position_size = max_position_size
        self.max_loss_per_trade = max_loss_per_trade
        self.losses_today = 0
    
    def can_trade(self, ticker: str, signal_strength: float, current_price: float, quantity: int) -> bool:
        """
        Determine if trade should be executed based on risk limits.
        
        Returns: True if trade is safe, False otherwise
        """
        
        # Position size check
        position_value = current_price * quantity
        if position_value > self.max_position_size:
            logger.warning(f"Position size {position_value:.2f} exceeds max {self.max_position_size:.2f}")
            return False
        
        # Daily loss limit
        if self.losses_today > self.max_loss_per_trade:
            logger.warning(f"Daily loss {self.losses_today:.2f} exceeds limit {self.max_loss_per_trade:.2f}")
            return False
        
        # Confidence threshold
        if abs(signal_strength) < 0.55:
            logger.info(f"Signal strength {signal_strength:.2f} too weak, skipping trade")
            return False
        
        return True
    
    def log_trade_result(self, profit_loss: float):
        """Record trade result for risk management"""
        if profit_loss < 0:
            self.losses_today += abs(profit_loss)
            logger.info(f"Trade loss: ${profit_loss:.2f}. Daily loss: ${self.losses_today:.2f}")
    
    def reset_daily(self):
        """Reset daily counters (call each market open)"""
        self.losses_today = 0
        logger.info("Daily risk counters reset")


# ============================================================================
# Example 4: Live Trading Loop
# ============================================================================

def live_trading_loop(analysis_predictions: Dict[str, float], backtest_results: Dict):
    """
    Main loop for live trading execution.
    
    Steps:
    1. Get predictions from pipeline
    2. Check risk limits
    3. Execute trades via broker
    4. Log results
    """
    
    risk_manager = PortfolioRiskManager(
        max_position_size=10000,      # $10k per position
        max_loss_per_trade=1000       # Stop after $1k loss/day
    )
    
    # Get current prices (example)
    current_prices = {
        "AAPL": 185.50,
        "MSFT": 378.92,
        "GOOGL": 140.31
    }
    
    logger.info("=" * 80)
    logger.info("LIVE TRADING LOOP")
    logger.info("=" * 80)
    
    for ticker, signal in analysis_predictions.items():
        if ticker not in current_prices:
            logger.warning(f"No price data for {ticker}, skipping")
            continue
        
        current_price = current_prices[ticker]
        
        # Position sizing (Kelly criterion or simpler fixed)
        quantity = int(5000 / current_price)  # $5k per trade
        
        # Risk check
        if not risk_manager.can_trade(ticker, signal, current_price, quantity):
            logger.info(f"{ticker}: Trade blocked by risk manager")
            continue
        
        # Execute
        direction = "BUY" if signal > 0 else "SELL"
        logger.info(f"{ticker}: {direction} signal {signal:.2f}, qty={quantity} @ ${current_price:.2f}")
        
        # TODO: Uncomment for real trading
        # execute_with_alpaca({ticker: signal}, config={"paper_trading": True})


# ============================================================================
# Example 5: Backtest Trading Strategy
# ============================================================================

def backtest_trading_signals(test_df: pd.DataFrame, predictions: pd.Series) -> Dict:
    """
    Backtest trading signals on historical data.
    
    Args:
        test_df: DataFrame with Close price
        predictions: Series with predicted prices or signals
    
    Returns:
        {"total_return": 0.05, "win_rate": 0.62, "trades": 15}
    """
    
    trades = []
    positions = []
    
    for i in range(1, len(test_df)):
        actual_change = test_df.iloc[i]['Close'] - test_df.iloc[i-1]['Close']
        predicted_signal = predictions.iloc[i-1]
        
        # Did prediction match actual movement?
        if (predicted_signal > 0 and actual_change > 0) or (predicted_signal < 0 and actual_change < 0):
            trades.append({"date": test_df.index[i], "result": "WIN", "pnl": abs(actual_change)})
        else:
            trades.append({"date": test_df.index[i], "result": "LOSS", "pnl": -abs(actual_change)})
    
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {"total_return": 0, "win_rate": 0, "trades": 0}
    
    total_pnl = trades_df['pnl'].sum()
    total_return = total_pnl / test_df.iloc[0]['Close']
    win_rate = (trades_df['result'] == "WIN").sum() / len(trades_df)
    
    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "trades": len(trades_df),
        "avg_pnl": trades_df['pnl'].mean()
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example predictions from pipeline
    predictions = {
        "AAPL": 0.65,      # 65% confidence BUY
        "MSFT": -0.42,     # 42% confidence SELL
        "GOOGL": 0.51      # 51% confidence BUY
    }
    
    # Example 1: Extract signals
    print("\n=== Example 1: Extract Predictions ===")
    example_metrics = pd.DataFrame({
        "Model": ["LSTM", "ARIMA", "RF"],
        "RMSE": [1.23, 1.45, 1.31],
        "Directional Accuracy": [0.62, 0.58, 0.61]
    })
    signal = extract_predictions_from_pipeline(example_metrics)
    print(f"Ensemble signal: {signal}")
    
    # Example 2: Risk management
    print("\n=== Example 2: Risk Manager ===")
    risk_mgr = PortfolioRiskManager()
    can_trade = risk_mgr.can_trade("AAPL", 0.65, 185.50, 50)
    print(f"Can trade: {can_trade}")
    risk_mgr.log_trade_result(-200)  # Example loss
    
    # Example 3: Live trading (paper trading only)
    print("\n=== Example 3: Live Trading Loop ===")
    live_trading_loop(predictions, backtest_results={})
    
    # Example 4: Backtest (example with random data)
    print("\n=== Example 4: Backtest ===")
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=100)
    test_data = pd.DataFrame({
        "Close": np.random.randn(100).cumsum() + 100
    }, index=dates)
    fake_predictions = pd.Series(np.random.randn(100), index=dates)
    backtest_result = backtest_trading_signals(test_data, fake_predictions)
    print(f"Backtest result: {backtest_result}")
    
    print("\n" + "=" * 80)
    print("⚠️  DISCLAIMER: This is educational code only!")
    print("Always test in paper trading before using with real money.")
    print("="*80)
