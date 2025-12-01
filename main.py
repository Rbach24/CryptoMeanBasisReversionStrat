from AlgorithmImports import *
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

class TreeModelTrader(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021,1,1)
        self.SetEndDate(2021,12,31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BINANCE_FUTURES, AccountType.MARGIN)

        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour).Symbol

        self.cooldown_hours = 6
        self.last_trade_time = None
        self.horizon = 6
        self.model = None

        chart = Chart("Basis")
        chart.AddSeries(Series("BasisValue", SeriesType.Line, 0))
        chart.AddSeries(Series("PredProb", SeriesType.Line, 0))
        self.AddChart(chart)

        self.model = self.TrainRF()
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.TradeBasis)

    def TrainRF(self):
        if not SKLEARN_OK:
            self.Debug("sklearn unavailable; skipping RF.")
            return None

        history_spot = self.History(self.spot, 120, Resolution.Hour)
        history_fut = self.History(self.future, 120, Resolution.Hour)
        if history_spot.empty or history_fut.empty:
            self.Debug("Insufficient history for RF.")
            return None

        spot = history_spot.close.unstack(level=0)[self.spot]
        fut = history_fut.close.unstack(level=0)[self.future]
        df = pd.concat([spot,fut],axis=1,join='inner').dropna()
        df.columns = ["spot","future"]

        df["basis"] = (df.future - df.spot) / df.spot
        df["basis_mean24"] = df.basis.rolling(24).mean()
        df["basis_std24"] = df.basis.rolling(24).std()
        df["momentum"] = df.basis.diff()
        df["vol_spot"] = df.spot.pct_change().rolling(24).std()
        # target: whether basis returns closer to mean within horizon
        df["future_basis"] = df.basis.shift(-self.horizon)
        df["target_prob"] = (abs(df.future_basis - df.basis.mean()) < abs(df.basis - df.basis.mean())).astype(int)
        df.dropna(inplace=True)

        X = df[["basis","basis_mean24","basis_std24","momentum","vol_spot"]].values
        y = df["target_prob"].values

        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        self.Debug(f"Random forest trained; test accuracy: {score:.3f}")
        return model

    def TradeBasis(self):
        if not self.model:
            return
        history_spot = self.History(self.spot, 24, Resolution.Hour)
        history_fut = self.History(self.future, 24, Resolution.Hour)
        if history_spot.empty or history_fut.empty:
            return
        spot = history_spot.close.unstack(level=0)[self.spot]
        fut = history_fut.close.unstack(level=0)[self.future]
        aligned = pd.concat([spot,fut],axis=1,join='inner').dropna()
        aligned.columns = ["spot","future"]
        if len(aligned) < 2:
            return

        spot_price = float(aligned["spot"].iloc[-1])
        fut_price = float(aligned["future"].iloc[-1])
        basis = (fut_price - spot_price) / spot_price
        arr_basis = (aligned["future"].values - aligned["spot"].values) / aligned["spot"].values

        feat = np.array([[basis, np.mean(arr_basis), np.std(arr_basis), arr_basis[-1]-arr_basis[-2] if len(arr_basis)>1 else 0.0, np.std(np.diff(aligned["spot"].values)/aligned["spot"].values[:-1])]])
        prob = float(self.model.predict_proba(feat)[0][1])  # probability revert-to-mean
        self.Plot("Basis","BasisValue",basis)
        self.Plot("Basis","PredProb",prob)

        if self.last_trade_time and (self.Time - self.last_trade_time).total_seconds() < self.cooldown_hours*3600:
            return

        portfolio_value = self.Portfolio.TotalPortfolioValue
        exposure = portfolio_value * 0.25
        future_qty = exposure / fut_price
        spot_qty = exposure / spot_price

        # interpret prob > 0.6 as likely reversion: short side that is high relative to fair
        threshold = 0.6
        mean_basis = np.mean(arr_basis)
        if prob > threshold and basis > mean_basis and not self.Portfolio[self.future].IsShort:
            self.Debug(f"RF prob {prob:.2f} -> short future / long spot")
            self.Liquidate()
            self.MarketOrder(self.future, -future_qty)
            self.MarketOrder(self.spot, spot_qty)
            self.last_trade_time = self.Time
        elif prob > threshold and basis < mean_basis and not self.Portfolio[self.future].IsLong:
            self.Debug(f"RF prob {prob:.2f} -> long future / short spot (below mean)")
            self.Liquidate()
            self.MarketOrder(self.future, future_qty)
            self.MarketOrder(self.spot, -spot_qty)
            self.last_trade_time = self.Time
        elif prob < 0.5 and (self.Portfolio[self.future].Invested or self.Portfolio[self.spot].Invested):
            # low confidence -> close
            self.Debug("RF low confidence -> close positions")
            self.Liquidate()
            self.last_trade_time = self.Time
