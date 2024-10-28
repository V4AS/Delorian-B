import streamlit as st
import vectorbt as vbt
import talib
import pandas as pd
import datetime
from yfinance import download

# Set up Streamlit UI
st.title('Delorian A&B Backtest')

# User input for symbol, timeframe, and start/end dates
symbol = st.text_input('Enter asset symbol (e.g., BTC-USD):', 'BTC-USD')
timeframe = st.selectbox('Select timeframe:', ['1h', '1d'])
start_date = st.date_input('Start date (YYYY-MM-DD):', datetime.date(2022, 1, 1))
end_date = st.date_input('End date (YYYY-MM-DD):', datetime.date(2023, 1, 1))

# Fetch historical data based on user inputs
if st.button('Run Backtest'):
    # Get data
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    data = download(symbol, start=start_str, end=end_str, interval=timeframe)
    close = data['Close']

    # Calculate Wave Trend using TA-Lib
    def wave_trend(close, channel_length=9, avg_length=12, ma_length=3):
        esa = talib.EMA(close, timeperiod=channel_length)
        deviation = abs(close - esa)
        de = talib.EMA(deviation, timeperiod=channel_length)
        ci = (close - esa) / (0.015 * de)
        wt1 = talib.EMA(ci, timeperiod=avg_length)
        wt2 = talib.SMA(wt1, timeperiod=ma_length)
        return wt1, wt2

    # Calculate wave trend
    wt1, wt2 = wave_trend(close)

    # Generate buy and sell signals based on divergence and wave trend crossover
    buy_signal = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    sell_signal = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    # Find divergence conditions
    bullish_div = (wt1 < 0) & (close > close.shift(1))
    bearish_div = (wt1 > 0) & (close < close.shift(1))

    # Create entries for backtesting
    long_entries = (buy_signal & bullish_div).shift(1).fillna(False).astype(bool)
    short_entries = (sell_signal & bearish_div).shift(1).fillna(False).astype(bool)

    # Run the backtest and optimize take-profit and stop-loss levels
    tp_values = [1.5, 2.0, 2.5, 3.0]
    sl_values = [1.0, 1.5, 2.0, 2.5]
    best_return = -float('inf')
    best_pf = None
    best_tp = None
    best_sl = None

    for tp in tp_values:
        for sl in sl_values:
            # Make sure indices align properly by reindexing
            long_exit_prices = close[long_entries].reindex(close.index, method='ffill')
            short_exit_prices = close[short_entries].reindex(close.index, method='ffill')

            # Create exit signals based on tp and sl
            long_exits = ((close >= long_exit_prices * (1 + tp / 100)) | 
                          (close <= long_exit_prices * (1 - sl / 100))) & long_entries.shift(1, fill_value=False)
            short_exits = ((close <= short_exit_prices * (1 - tp / 100)) | 
                           (close >= short_exit_prices * (1 + sl / 100))) & short_entries.shift(1, fill_value=False)

            pf = vbt.Portfolio.from_signals(
                close,
                entries=long_entries,
                exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                freq=timeframe,
                init_cash=1000
            )
            total_return = pf.total_return()
            if total_return > best_return:
                best_return = total_return
                best_pf = pf
                best_tp = tp
                best_sl = sl

    # Display the best performance
    if best_pf is not None:
        st.subheader('Optimal Parameters and Stats')
        st.write(f"Optimal Take-Profit: {best_tp}")
        st.write(f"Optimal Stop-Loss: {best_sl}")
        st.write(best_pf.stats())

        # Plot results
        st.subheader('Equity Curve')
        fig = best_pf.plot()
        st.plotly_chart(fig)

        # Show trades stats and drawdowns
        st.subheader('Trades Stats')
        st.write(best_pf.trades.stats())
        st.subheader('Positions')
        st.write(best_pf.positions.records_readable)
        st.subheader('Drawdowns')
        drawdowns = best_pf.drawdowns.records_readable
        st.write(drawdowns)
        st.subheader('Drawdown Plot')
        drawdown_fig = best_pf.drawdowns.plot()
        st.plotly_chart(drawdown_fig)
