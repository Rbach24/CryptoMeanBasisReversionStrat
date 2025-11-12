# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# endregion

class MLPredictiveBasisTrader(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)

        # Binance Futures brokerage
        self.SetBrokerageModel(BrokerageName.BINANCE_FUTURES, AccountType.MARGIN)

        # Add assets
        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour).Symbol

        # Rolling data and cooldown
        self.last_prices = {}
        self.cooldown_hours = 6
        self.last_trade_time = None
        self.last_basis_trade_value = None

        # For plotting
        self.basis_chart = Chart("Basis")
        self.basis_chart.AddSeries(Series("BasisValue", SeriesType.LINE, 0))
        self.AddChart(self.basis_chart)

        # Train model once on historical data
        self.model = self.TrainMLModel()

        # Schedule trading logic every hour
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self.TradeBasis
        )

    def TrainMLModel(self):
        self.Debug("Fetching historical data for ML training...")

        # Fetch 90 days of hourly historical data
        history_spot = self.History(self.spot, 90, Resolution.Hour)
        history_fut = self.History(self.future, 90, Resolution.Hour)

        if history_spot.empty or history_fut.empty:
            self.Debug("Not enough historical data to train model.")
            return None

        # Convert to DataFrame and align timestamps
        spot_df = history_spot.close.unstack(level=0)
        fut_df = history_fut.close.unstack(level=0)
        df = pd.concat([spot_df[self.spot], fut_df[self.future]], axis=1, join="inner").dropna()
        df.columns = ["spot", "future"]

        # Build features
        df["basis"] = (df["future"] - df["spot"]) / df["spot"]
        df["basis_mean"] = df["basis"].rolling(24).mean()
        df["basis_std"] = df["basis"].rolling(24).std()
        df["momentum"] = df["basis"].diff()
        df["vol_spot"] = df["spot"].pct_change().rolling(24).std()
        df["target"] = (df["basis"].shift(-6) > df["basis"]).astype(int)
        df.dropna(inplace=True)

        # Prepare training data
        X = df[["basis", "basis_mean", "basis_std", "momentum", "vol_spot"]].values
        y = df["target"].values
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        self.Debug(f"ML model trained with accuracy: {acc:.2f}")

        return model

    def OnData(self, slice):
        # Store latest prices
        if self.spot in slice.Bars:
            self.last_prices[self.spot] = slice.Bars[self.spot].Close
        if self.future in slice.Bars:
            self.last_prices[self.future] = slice.Bars[self.future].Close

    def TradeBasis(self):
        if not self.model:
            return

        if self.spot not in self.last_prices or self.future not in self.last_prices:
            return

        spot_price = self.last_prices[self.spot]
        future_price = self.last_prices[self.future]
        basis = float((future_price - spot_price) / spot_price)
        self.Plot("Basis", "BasisValue", basis)

        # Fetch last 24 hours for rolling stats
        history_spot = self.History(self.spot, 24, Resolution.Hour)
        history_fut = self.History(self.future, 24, Resolution.Hour)
        if history_spot.empty or history_fut.empty:
            return

        spot_df = history_spot.close.unstack(level=0)
        fut_df = history_fut.close.unstack(level=0)
        aligned = pd.concat([spot_df[self.spot], fut_df[self.future]], axis=1, join="inner").dropna()
        aligned.columns = ["spot", "future"]

        if len(aligned) < 2:
            return  # Not enough data

        spot_prices = aligned["spot"].values
        fut_prices = aligned["future"].values
        arr_basis = (fut_prices - spot_prices) / spot_prices

        # Feature vector
        mean = np.mean(arr_basis)
        std = np.std(arr_basis)
        momentum = arr_basis[-1] - arr_basis[-2]
        vol_spot = np.std(np.diff(spot_prices) / spot_prices[:-1])
        x = np.array([[basis, mean, std, momentum, vol_spot]])

        # Predict basis direction
        pred = self.model.predict(x)[0]

        # Cooldown check
        if self.last_trade_time and (self.Time - self.last_trade_time).total_seconds() < self.cooldown_hours * 3600:
            return

        # Position sizing
        portfolio_value = self.Portfolio.TotalPortfolioValue
        exposure = portfolio_value * 0.25
        future_qty = exposure / future_price
        spot_qty = exposure / spot_price

        # Predicted up → short future / long spot
        if pred == 1 and not self.Portfolio[self.future].IsShort:
            self.Debug("ML predicts basis rising → short future / long spot")
            self.Liquidate()
            self.MarketOrder(self.future, -future_qty)
            self.MarketOrder(self.spot, spot_qty)
            self.last_trade_time = self.Time
            self.last_basis_trade_value = portfolio_value

        # Predicted down → long future / short spot
        elif pred == 0 and not self.Portfolio[self.future].IsLong:
            self.Debug("ML predicts basis falling → long future / short spot")
            self.Liquidate()
            self.MarketOrder(self.future, future_qty)
            self.MarketOrder(self.spot, -spot_qty)
            self.last_trade_time = self.Time
            self.last_basis_trade_value = portfolio_value

        # Close when basis is near neutral
        elif abs(basis) < 0.005 and (self.Portfolio[self.future].Invested or self.Portfolio[self.spot].Invested):
            self.Liquidate()
            if self.last_basis_trade_value:
                trade_profit = portfolio_value - self.last_basis_trade_value
                self.Debug(f"Closed positions | Profit: {trade_profit:.2f} USDT")
            self.last_trade_time = self.Time
            self.last_basis_trade_value = None

