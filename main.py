# Combines a Kalman filter fair-value basis estimator with:
#  - regime filter (volatility + trend neutrality)
#  - RSI confirmation
#  - cost-aware no-trade band (min move to cover fees)
#
# Trading idea: trade basis mean-reversions for BTCUSDT (spot vs perp/future).
# Conservative sizing and cooldown to keep trade count low.

from AlgorithmImports import *
import numpy as np
import pandas as pd
import math

class KalmanEnsembleTrader(QCAlgorithm):

    def Initialize(self):
        # === basic config ===
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)

        # Binance futures model (margin)
        self.SetBrokerageModel(BrokerageName.BINANCE_FUTURES, AccountType.MARGIN)

        # instruments
        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour).Symbol

        # scheduling: run logic every hour (matching data resolution)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.RunStrategy)

        # === Kalman state (1D) ===
        # state x = fair basis (future - spot) / spot
        self.x = 0.0       # state estimate
        self.P = 1.0       # estimate covariance
        self.Q = 1e-5      # process noise (tune)
        self.R = 5e-4      # measurement noise (tune)

        # === risk / sizing / cooldown ===
        self.cooldown_hours = 6               # cooldown after a trade
        self.last_trade_time = None
        self.exposure_frac = 0.20             # fraction of portfolio per leg (spot and future)
        self.max_exposure_frac = 0.4          # safety cap
        self.max_leverage = 3.0               # not strictly enforced—informational
        self.min_hours_between_retrain = 24   # placeholder if you add retrain

        # === thresholds & filters (conservative defaults) ===
        self.zscore_entry = 1.8              # base z-score threshold
        self.rsi_period = 14
        self.rsi_entry_allow = (30, 70)      # only enter when RSI not in extreme momentum
        self.vol_lookback = 24
        self.vol_min = 0.0005                # minimum sigma (low vol) — tune for BTC hourly
        self.vol_max = 0.02                  # maximum sigma (high vol) — avoid trading in very choppy times
        self.trend_fast = 12
        self.trend_slow = 48
        self.trend_max_gap = 0.005           # when fast/slow MA gap > this, consider trending and disable

        # === fee model (estimated) ===
        # Use an estimated round-trip fee percentage of notionally traded USDT.
        # For example: if maker/taker fees + slippage ~= 0.01% each way, round-trip ~0.0002 (0.02%).
        # You observed about 900 in fees on some run—use conservative fraction here and adjust.
        self.estimated_roundtrip_fee_frac = 0.0008  # 0.08% round-trip by default (tune for your experience)

        # === plotting ===
        c = Chart("Basis")
        c.AddSeries(Series("BasisValue", SeriesType.Line, 0))
        c.AddSeries(Series("KalmanFair", SeriesType.Line, 0))
        c.AddSeries(Series("KalmanZ", SeriesType.Line, 0))
        c.AddSeries(Series("RSI", SeriesType.Line, 0))
        self.AddChart(c)

        # === logging / bookkeeping ===
        self.trades_taken = 0
        self.total_fees_paid_est = 0.0
        self.total_profit_record = []

        # seed Kalman with recent average if available
        self.PrimeKalman()

    def PrimeKalman(self):
        # initialize x and P with recent history mean
        history_spot = self.History(self.spot, 48, Resolution.Hour)
        history_fut = self.History(self.future, 48, Resolution.Hour)
        if history_spot.empty or history_fut.empty:
            return
        s = history_spot.close.unstack(level=0)[self.spot]
        f = history_fut.close.unstack(level=0)[self.future]
        df = pd.concat([s, f], axis=1, join='inner').dropna()
        df.columns = ["spot", "future"]
        df["basis"] = (df.future - df.spot) / df.spot
        if len(df["basis"]) > 0:
            self.x = float(df["basis"].iloc[-1])
            self.P = max(1e-4, float(np.var(df["basis"].values)))
            self.Debug(f"Kalman primed: x={self.x:.6f}, P={self.P:.6e}")

    def OnData(self, slice):
        # We don't use OnData for decision making here; RunStrategy is scheduled hourly.
        pass

    def RunStrategy(self):
        # fetch sufficient history
        lookback = max(self.trend_slow, 48)
        history_spot = self.History(self.spot, lookback, Resolution.Hour)
        history_fut = self.History(self.future, lookback, Resolution.Hour)
        if history_spot.empty or history_fut.empty:
            return

        spot_s = history_spot.close.unstack(level=0)[self.spot]
        fut_s = history_fut.close.unstack(level=0)[self.future]
        df = pd.concat([spot_s, fut_s], axis=1, join='inner').dropna()
        df.columns = ["spot", "future"]
        if len(df) < 10:
            return

        # compute basis series
        df["basis"] = (df["future"] - df["spot"]) / df["spot"]

        # --- Kalman sequential update using recent measurements ---
        measurements = df["basis"].values
        # perform sequential update on last N (cap at 240 to keep it reasonable)
        for z in measurements[-240:]:
            # prediction step: x = x (identity), P = P + Q
            self.P = self.P + self.Q
            # update step:
            K = self.P / (self.P + self.R)
            self.x = self.x + K * (z - self.x)
            self.P = (1 - K) * self.P

        fair_basis = float(self.x)
        uncertainty = math.sqrt(self.P)  # estimate standard deviation of state
        current_spot = float(df["spot"].iloc[-1])
        current_fut = float(df["future"].iloc[-1])
        current_basis = (current_fut - current_spot) / current_spot

        # --- features for regime filter & confirmation ---
        # volatility (spot returns std)
        ret = np.diff(df["spot"].values) / df["spot"].values[:-1]
        vol = float(np.std(ret[-self.vol_lookback:])) if len(ret) >= self.vol_lookback else float(np.std(ret))
        # trend via moving averages on future/spot basis (converted to basis MA)
        fast_ma = df["basis"].rolling(self.trend_fast).mean().iloc[-1]
        slow_ma = df["basis"].rolling(self.trend_slow).mean().iloc[-1]
        trend_gap = 0.0 if (np.isnan(fast_ma) or np.isnan(slow_ma)) else abs(fast_ma - slow_ma)

        # RSI computed on spot price
        rsi = self.SafeComputeRSI(df["spot"].values, self.rsi_period)

        # --- cost-aware minimum move (in basis units) ---
        # estimate minimal basis move such that profit on exposure covers round-trip fees
        # approximate: profit_usdt = exposure * (basis_move)  => basis_move_min = roundtrip_fees_frac
        # since both legs use exposure, we are conservative and say basis must move by at least fee_frac
        fee_frac = self.estimated_roundtrip_fee_frac
        min_basis_move_from_fees = fee_frac  # conservative approximation
        # also require move to beat statistical noise: scale by uncertainty
        min_basis_move_stat = self.zscore_entry * uncertainty

        # effective threshold (directional): require abs deviation > max(min_basis_move_from_fees, min_basis_move_stat)
        effective_threshold = max(min_basis_move_from_fees, min_basis_move_stat)

        # also compute z-score for plotting and diagnostic: (basis - fair) / uncertainty
        zscore = (current_basis - fair_basis) / (uncertainty + 1e-12)

        # plotting
        self.Plot("Basis", "BasisValue", current_basis)
        self.Plot("Basis", "KalmanFair", fair_basis)
        self.Plot("Basis", "KalmanZ", zscore)
        self.Plot("Basis", "RSI", rsi if not math.isnan(rsi) else 50)

        # --- regime filter: only trade when vol within band and trend weak ---
        regime_ok = (vol >= self.vol_min) and (vol <= self.vol_max) and (trend_gap <= self.trend_max_gap)
        # debug reasons
        # self.Debug(f"vol={vol:.6f}, trend_gap={trend_gap:.6f}, regime_ok={regime_ok}")

        # --- cooldown / safety checks ---
        if self.last_trade_time and (self.Time - self.last_trade_time).total_seconds() < self.cooldown_hours * 3600:
            # self.Debug("Cooldown active, skipping")
            return

        # don't trade if already invested in a direction that's consistent (we want few trades)
        invested = self.Portfolio[self.future].Invested or self.Portfolio[self.spot].Invested

        # determine direction signals
        allow_by_rsi = (rsi >= self.rsi_entry_allow[0]) and (rsi <= self.rsi_entry_allow[1])

        # calculated signal based on zscore vs effective threshold
        # positive zscore: basis above fair (future expensive) -> expect reversion: short future / long spot
        want_short_future = (zscore > 0) and (abs(zscore) * uncertainty >= effective_threshold)
        want_long_future = (zscore < 0) and (abs(zscore) * uncertainty >= effective_threshold)

        # require both regime and RSI confirmation and not currently invested in same direction
        open_short_ok = regime_ok and allow_by_rsi and want_short_future and (not self.Portfolio[self.future].IsShort)
        open_long_ok = regime_ok and allow_by_rsi and want_long_future and (not self.Portfolio[self.future].IsLong)

        # safety: ensure exposure fraction isn't larger than allowed by portfolio health
        portfolio_value = self.Portfolio.TotalPortfolioValue
        exposure_frac = min(self.exposure_frac, self.max_exposure_frac)
        exposure = portfolio_value * exposure_frac
        future_qty = exposure / current_fut if current_fut > 0 else 0
        spot_qty = exposure / current_spot if current_spot > 0 else 0

        # Execute trades
        if open_short_ok:
            self.Debug(f"ENTER SHORT FUT / LONG SPOT | z={zscore:.2f} fair={fair_basis:.6f} basis={current_basis:.6f} vol={vol:.6f} rsi={rsi:.1f}")
            self.Liquidate()
            # short future, long spot
            if future_qty > 0:
                self.MarketOrder(self.future, -future_qty)
            if spot_qty > 0:
                self.MarketOrder(self.spot, spot_qty)
            self.last_trade_time = self.Time
            self.trades_taken += 1

        elif open_long_ok:
            self.Debug(f"ENTER LONG FUT / SHORT SPOT | z={zscore:.2f} fair={fair_basis:.6f} basis={current_basis:.6f} vol={vol:.6f} rsi={rsi:.1f}")
            self.Liquidate()
            # long future, short spot
            if future_qty > 0:
                self.MarketOrder(self.future, future_qty)
            if spot_qty > 0:
                self.MarketOrder(self.spot, -spot_qty)
            self.last_trade_time = self.Time
            self.trades_taken += 1

        else:
            # no new trade; consider exit if basis near-neutral
            neutral_close_band = max(min_basis_move_from_fees / 2.0, 0.0025)  # tolerance to close
            if abs(current_basis) < neutral_close_band and (self.Portfolio[self.future].Invested or self.Portfolio[self.spot].Invested):
                self.Debug("Closing: basis near neutral")
                self.Liquidate()
                self.last_trade_time = self.Time

        # record estimated fees roughly for diagnostics (this is an estimate, not from actual fills)
        # estimate fees only when we recently traded
        if invested and not (self.Portfolio[self.future].Invested or self.Portfolio[self.spot].Invested):
            # we closed positions; estimate one round-trip fee on prior exposure
            self.total_fees_paid_est += portfolio_value * fee_frac

    def SafeComputeRSI(self, prices, period):
        # compute basic RSI; return NaN if insufficient data
        prices = np.asarray(prices, dtype=float)
        if len(prices) < period + 1:
            return float("nan")
        deltas = np.diff(prices)
        ups = np.where(deltas > 0, deltas, 0.0)
        downs = np.where(deltas < 0, -deltas, 0.0)
        roll_up = np.mean(ups[-period:])
        roll_down = np.mean(downs[-period:])
        if roll_down == 0 and roll_up == 0:
            return 50.0
        rs = roll_up / (roll_down + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)
