from AlgorithmImports import *
from datetime import timedelta
from collections import deque,OrderedDict
import numpy as np
import math,pickle,json
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE=True
except ImportError:
    xgb=None
    XGBOOST_AVAILABLE=False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE=True
except ImportError:
    torch=None
    nn=None
    TORCH_AVAILABLE=False

class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size=64,num_layers=2,lstm_dropout=0.2,dropout_rate=0.3,fc_structure=None):
        super(LSTMModel,self).__init__()
        lstm_dropout=lstm_dropout if num_layers>1 else 0.0
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers=num_layers,batch_first=True,dropout=lstm_dropout)
        self.dropout=nn.Dropout(dropout_rate) if dropout_rate>0 else None
        self.activation=nn.ReLU()
        self.fc_layer_names=[]
        fc_structure=fc_structure if fc_structure is not None else [32,1]
        prev_dim=hidden_size
        if fc_structure:
            for idx,layer_dim in enumerate(fc_structure):
                layer_name=f"fc{idx+1}"
                layer=nn.Linear(prev_dim,layer_dim)
                setattr(self,layer_name,layer)
                self.fc_layer_names.append(layer_name)
                prev_dim=layer_dim
        else:
            self.fc=nn.Linear(hidden_size,1)

    def forward(self,x):
        lstm_out,_=self.lstm(x)
        output=lstm_out[:,-1,:]
        if self.dropout is not None:
            output=self.dropout(output)
        if self.fc_layer_names:
            for idx,name in enumerate(self.fc_layer_names):
                layer=getattr(self,name)
                output=layer(output)
                if idx<len(self.fc_layer_names)-1:
                    output=self.activation(output)
                    if self.dropout is not None:
                        output=self.dropout(output)
            return output
        return self.fc(output)

FEATURE_ORDER=['spot_return_1h','perp_return_1h','spot_return_24h','perp_return_24h','spot_vol_24h','perp_vol_24h','spot_momentum','perp_momentum','spot_volume_ratio','perp_volume_ratio','basis','basis_mean_48h','basis_std_48h','basis_zscore','basis_change_1h','basis_change_24h','basis_momentum','funding_rate','funding_rate_ma24h','funding_rate_std24h','funding_rate_change_1h','funding_pressure','eth_basis','ethbtc_ratio','hour_of_day','day_of_week']

class MLBasisEnsembleStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021,1,1)
        self.SetEndDate(2022,12,31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BINANCE_FUTURES,AccountType.MARGIN)
        self.use_pretrained_models=True
        self.pretrained_models_loaded=False
        self.btc_spot=self.AddCrypto("BTCUSDT",Resolution.Hour).Symbol
        self.btc_perp=self.AddCryptoFuture("BTCUSDT",Resolution.Hour).Symbol
        self.btc_future_contract=self.Securities[self.btc_perp]
        try:
            self.eth_spot=self.AddCrypto("ETHUSDT",Resolution.Hour).Symbol
            self.eth_perp=self.AddCryptoFuture("ETHUSDT",Resolution.Hour).Symbol
        except:
            self.eth_spot=None
            self.eth_perp=None
        self.lookback_hours=168
        self.sequence_length=24
        self.prediction_horizon=6
        self.price_data={'btc_spot':deque(maxlen=self.lookback_hours),'btc_perp':deque(maxlen=self.lookback_hours),'eth_spot':deque(maxlen=self.lookback_hours),'eth_perp':deque(maxlen=self.lookback_hours)}
        self.funding_rate_history=deque(maxlen=self.lookback_hours)
        self.funding_rate_available=False
        self.funding_rate_warning_logged=False
        self.feature_history=deque(maxlen=self.sequence_length)
        self.max_training_samples=2000
        self.training_data={'features':deque(maxlen=self.max_training_samples),'labels':deque(maxlen=self.max_training_samples),'timestamps':deque(maxlen=self.max_training_samples),'basis_raw':deque(maxlen=self.max_training_samples)}
        self.prediction_lookup=OrderedDict()
        self.realized_history=deque(maxlen=200)
        self.feature_means={}
        self.feature_stds={}
        self.scaler_initialized=False
        self.xgb_model=None
        self.lstm_model=None
        self.model_trained=False
        self.xgboost_available=XGBOOST_AVAILABLE
        self.lstm_available=TORCH_AVAILABLE
        if not self.xgboost_available:
            self.Debug("XGBoost not available")
        if not self.lstm_available:
            self.Debug("PyTorch not available")
        self.xgb_weight=0.5
        self.lstm_weight=0.5
        self.latest_xgb_prediction=0.0
        self.latest_lstm_prediction=0.0
        self.latest_ensemble_prediction=0.0
        perp_props=self.Securities[self.btc_perp].SymbolProperties
        spot_props=self.Securities[self.btc_spot].SymbolProperties
        self.perp_lot_size=float(perp_props.LotSize or 0.001)
        self.spot_lot_size=float(spot_props.LotSize or 0.001)
        self.effective_lot_size=max(self.perp_lot_size,self.spot_lot_size)
        min_order_size=getattr(perp_props,"MinimumOrderSize",None)
        self.perp_min_quantity=float(min_order_size) if min_order_size is not None else self.perp_lot_size
        self.exchange_min_notional=5.0
        self.current_position=None
        self.position_entry_time=None
        self.position_entry_basis=None
        self.position_entry_value=None
        self.last_conviction=0.0
        self.conviction_threshold=0.05 # 0.05 ≈ 0.5% predicted basis change after tanh scaling
        self.exit_conviction_threshold=self.conviction_threshold*0.5
        self.max_leverage=2.0
        self.volatility_cap=0.5
        self.position_max_hours=12
        self.retrain_days=1
        self.min_training_samples=168
        self.max_drawdown_per_trade=0.05
        self.maker_fee=0.0002
        self.taker_fee=0.0004
        self.slippage_bps=2
        self.plot_interval=timedelta(hours=6) # throttle plotting to stay within chart quota
        self.last_plot_time=None
        self.FEATURE_ORDER=FEATURE_ORDER
        self.SetWarmUp(self.lookback_hours,Resolution.Hour)
        self._setup_charts()
        self.Schedule.On(self.DateRules.EveryDay(),self.TimeRules.Every(timedelta(hours=1)),self.HourlyUpdate)
        self.Schedule.On(self.DateRules.EveryDay(),self.TimeRules.At(0,0),self.DailyRetrain)
        if self.use_pretrained_models:
            self._load_pretrained_models()
        self.Debug("ML Basis Ensemble Strategy Initialized")

    def _setup_charts(self):
        basis_chart=Chart("Basis")
        basis_chart.AddSeries(Series("Basis",SeriesType.Line,0))
        basis_chart.AddSeries(Series("Basis_MA",SeriesType.Line,0))
        self.AddChart(basis_chart)
        pred_chart=Chart("Predictions")
        pred_chart.AddSeries(Series("XGB_Pred",SeriesType.Line,0))
        pred_chart.AddSeries(Series("LSTM_Pred",SeriesType.Line,0))
        pred_chart.AddSeries(Series("Ensemble_Pred",SeriesType.Line,0))
        self.AddChart(pred_chart)
        conv_chart=Chart("Conviction")
        conv_chart.AddSeries(Series("Conviction",SeriesType.Line,0))
        conv_chart.AddSeries(Series("Threshold",SeriesType.Line,0))
        self.AddChart(conv_chart)
        pnl_chart=Chart("Trade_PnL")
        pnl_chart.AddSeries(Series("Unrealized_PnL",SeriesType.Line,0))
        self.AddChart(pnl_chart)

    def _coerce_to_bytes(self,data):
        if data is None:
            return None
        if isinstance(data,bytes):
            return data
        if isinstance(data,bytearray):
            return bytes(data)
        try:
            return bytes(bytearray(data))
        except Exception:
            try:
                return bytes(data)
            except Exception as e:
                self.Debug("Unable to convert data to bytes: "+str(e))
                return None

    def _load_pretrained_models(self):
        try:
            try:
                xgb_bytes=self._coerce_to_bytes(self.ObjectStore.ReadBytes("models/xgb_model.pkl"))
                if xgb_bytes is None:
                    raise ValueError("XGBoost model bytes unavailable")
                self.xgb_model=pickle.loads(xgb_bytes)
                self.Debug("Loaded XGBoost model")
            except Exception as e:
                self.Debug("Could not load XGBoost:"+str(e))
            try:
                if TORCH_AVAILABLE:
                    lstm_bytes=self._coerce_to_bytes(self.ObjectStore.ReadBytes("models/lstm_model.pth"))
                    config_bytes=self._coerce_to_bytes(self.ObjectStore.ReadBytes("models/model_config.json"))
                    if lstm_bytes is None or config_bytes is None:
                        raise ValueError("LSTM model bytes unavailable")
                    config=json.loads(config_bytes.decode('utf-8'))
                    input_size=config['input_size']
                    hidden_size=int(config.get('hidden_size',64))
                    state_dict=pickle.loads(lstm_bytes)
                    num_layers=config.get('num_layers',config.get('lstm_layers'))
                    if num_layers is None:
                        lstm_layer_keys=[key for key in state_dict.keys() if key.startswith('lstm.weight_ih_l')]
                        if lstm_layer_keys:
                            num_layers=max(int(key.split('_l')[-1]) for key in lstm_layer_keys)+1
                        else:
                            num_layers=2
                    num_layers=int(num_layers)
                    lstm_dropout=float(config.get('lstm_dropout',0.2 if num_layers>1 else 0.0))
                    dropout_rate=float(config.get('dropout_rate',config.get('fc_dropout',0.3)))
                    fc_layers_raw=config.get('fc_layers') or config.get('fc_structure') or config.get('fc_layer_sizes')
                    if isinstance(fc_layers_raw,dict):
                        fc_layers=[int(value) for value in fc_layers_raw.values()]
                    elif isinstance(fc_layers_raw,list):
                        fc_layers=[int(value) for value in fc_layers_raw]
                    else:
                        fc_layers=None
                    if fc_layers is None:
                        derived_layers=[]
                        idx=1
                        while True:
                            weight_key=f"fc{idx}.weight"
                            if weight_key in state_dict:
                                derived_layers.append(int(state_dict[weight_key].shape[0]))
                                idx+=1
                            else:
                                break
                        if derived_layers:
                            fc_layers=derived_layers
                        elif "fc.weight" in state_dict:
                            fc_layers=[]
                    self.lstm_model=LSTMModel(input_size,hidden_size,num_layers=num_layers,lstm_dropout=lstm_dropout,dropout_rate=dropout_rate,fc_structure=fc_layers)
                    self.lstm_model.load_state_dict(state_dict)
                    self.lstm_model.eval()
                    self.Debug("Loaded LSTM model")
            except Exception as e:
                self.Debug("Could not load LSTM:"+str(e))
            try:
                scaler_bytes=self._coerce_to_bytes(self.ObjectStore.ReadBytes("models/scaler_config.json"))
                if scaler_bytes is None:
                    raise ValueError("Scaler bytes unavailable")
                scaler_config=json.loads(scaler_bytes.decode('utf-8'))
                self.feature_means=scaler_config.get('means',{})
                self.feature_stds=scaler_config.get('stds',{})
                if self.feature_means and self.feature_stds:
                    self.scaler_initialized=True
                    self.Debug("Loaded scalers")
            except Exception as e:
                self.Debug("Could not load scaler:"+str(e))
            if self.xgb_model is not None or self.lstm_model is not None:
                self.model_trained=True
                self.pretrained_models_loaded=True
                self.Debug("Pre-trained models loaded!")
            else:
                self.Debug("No pre-trained models, will train online")
                self.use_pretrained_models=False
        except Exception as e:
            self.Debug("Error loading models:"+str(e))
            self.use_pretrained_models=False

    def OnData(self,slice):
        if self.btc_spot in slice.Bars:
            self.price_data['btc_spot'].append({'time':self.Time,'close':float(slice.Bars[self.btc_spot].Close),'volume':float(slice.Bars[self.btc_spot].Volume)})
        if self.btc_perp in slice.Bars:
            self.price_data['btc_perp'].append({'time':self.Time,'close':float(slice.Bars[self.btc_perp].Close),'volume':float(slice.Bars[self.btc_perp].Volume)})
            try:
                if hasattr(self.btc_future_contract,'FundingRate'):
                    funding_rate=float(self.btc_future_contract.FundingRate)
                    self.funding_rate_history.append({'time':self.Time,'rate':funding_rate})
                    self.funding_rate_available=True
            except:
                pass
        if self.eth_spot and self.eth_spot in slice.Bars:
            self.price_data['eth_spot'].append({'time':self.Time,'close':float(slice.Bars[self.eth_spot].Close),'volume':float(slice.Bars[self.eth_spot].Volume)})
        if self.eth_perp and self.eth_perp in slice.Bars:
            self.price_data['eth_perp'].append({'time':self.Time,'close':float(slice.Bars[self.eth_perp].Close),'volume':float(slice.Bars[self.eth_perp].Volume)})

    def HourlyUpdate(self):
        if self.IsWarmingUp or not self._has_sufficient_data():
            return
        features=self._compute_features()
        if features is None:
            return
        self.feature_history.append(features.copy())
        self._store_training_sample(features)
        if self.model_trained:
            predictions=self._generate_predictions(features)
            if predictions is not None:
                self._execute_trading_logic(predictions)
        self._check_exit_conditions()
        self._update_plots(features)

    def DailyRetrain(self):
        if self.use_pretrained_models:
            return
        if len(self.training_data['features'])<self.min_training_samples:
            self.Debug("Insufficient data:"+str(len(self.training_data['features'])))
            return
        if self.funding_rate_available and len(self.funding_rate_history)<24:
            self.Debug("WARNING: FR flag true but only "+str(len(self.funding_rate_history))+" samples")
        self.Debug("Retraining with "+str(len(self.training_data['features']))+" samples")
        self._train_xgboost()
        self._train_lstm()
        self._update_model_weights()
        self.model_trained=True
        self.Debug("Retraining complete")

    def _has_sufficient_data(self):
        return len(self.price_data['btc_spot'])>=48 and len(self.price_data['btc_perp'])>=48

    def _compute_features(self):
        try:
            features={}
            btc_spot_prices=np.array([p['close'] for p in self.price_data['btc_spot']])
            btc_perp_prices=np.array([p['close'] for p in self.price_data['btc_perp']])
            btc_spot_returns=np.diff(np.log(btc_spot_prices[-25:]))
            btc_perp_returns=np.diff(np.log(btc_perp_prices[-25:]))
            features['spot_return_1h']=btc_spot_returns[-1]
            features['perp_return_1h']=btc_perp_returns[-1]
            features['spot_return_24h']=np.sum(btc_spot_returns)
            features['perp_return_24h']=np.sum(btc_perp_returns)
            features['spot_vol_24h']=np.std(btc_spot_returns)*np.sqrt(24)
            features['perp_vol_24h']=np.std(btc_perp_returns)*np.sqrt(24)
            if len(btc_spot_returns)>=24:
                features['spot_momentum']=np.sum(btc_spot_returns[-12:])-np.sum(btc_spot_returns[-24:-12])
                features['perp_momentum']=np.sum(btc_perp_returns[-12:])-np.sum(btc_perp_returns[-24:-12])
            else:
                features['spot_momentum']=0.0
                features['perp_momentum']=0.0
            btc_spot_volumes=np.array([p['volume'] for p in self.price_data['btc_spot']])
            btc_perp_volumes=np.array([p['volume'] for p in self.price_data['btc_perp']])
            features['spot_volume_ratio']=btc_spot_volumes[-1]/(np.mean(btc_spot_volumes[-24:])+1e-8)
            features['perp_volume_ratio']=btc_perp_volumes[-1]/(np.mean(btc_perp_volumes[-24:])+1e-8)
            basis=(btc_perp_prices[-1]-btc_spot_prices[-1])/btc_spot_prices[-1]
            features['basis']=basis
            basis_series=(btc_perp_prices[-48:]-btc_spot_prices[-48:])/btc_spot_prices[-48:]
            features['basis_mean_48h']=np.mean(basis_series)
            features['basis_std_48h']=np.std(basis_series)
            features['basis_zscore']=(basis-np.mean(basis_series))/(np.std(basis_series)+1e-8)
            features['basis_change_1h']=basis_series[-1]-basis_series[-2]
            features['basis_change_24h']=basis_series[-1]-basis_series[-24]
            features['basis_momentum']=np.mean(basis_series[-6:])-np.mean(basis_series[-12:-6])
            if len(self.funding_rate_history)>=24:
                funding_rates=np.array([f['rate'] for f in self.funding_rate_history])
                features['funding_rate']=funding_rates[-1]
                features['funding_rate_ma24h']=np.mean(funding_rates[-24:])
                features['funding_rate_std24h']=np.std(funding_rates[-24:])
                features['funding_rate_change_1h']=funding_rates[-1]-funding_rates[-2]
                features['funding_pressure']=funding_rates[-1]/(np.mean(np.abs(funding_rates[-24:]))+1e-8)
            else:
                if not self.funding_rate_available and not self.funding_rate_warning_logged:
                    self.Debug("FR data unavailable")
                    self.funding_rate_warning_logged=True
                features['funding_rate']=0.0
                features['funding_rate_ma24h']=0.0
                features['funding_rate_std24h']=0.0
                features['funding_rate_change_1h']=0.0
                features['funding_pressure']=0.0
            if self.eth_spot and len(self.price_data['eth_spot'])>=24:
                eth_spot_prices=np.array([p['close'] for p in self.price_data['eth_spot']])
                eth_perp_prices=np.array([p['close'] for p in self.price_data['eth_perp']])
                eth_basis=(eth_perp_prices[-1]-eth_spot_prices[-1])/eth_spot_prices[-1]
                features['eth_basis']=eth_basis
                ethbtc=eth_spot_prices[-1]/btc_spot_prices[-1]
                ethbtc_mean=np.mean(eth_spot_prices[-24:]/btc_spot_prices[-24:])
                features['ethbtc_ratio']=ethbtc/ethbtc_mean
            else:
                features['eth_basis']=0.0
                features['ethbtc_ratio']=1.0
            features['hour_of_day']=self.Time.hour
            features['day_of_week']=self.Time.weekday()
            return self._normalize_features(features)
        except Exception as e:
            self.Debug("Feature error:"+str(e))
            return None

    def _normalize_features(self,features):
        if not self.scaler_initialized:
            for key,value in features.items():
                if key not in ['hour_of_day','day_of_week']:
                    self.feature_means[key]=value
                    self.feature_stds[key]=1.0
            self.scaler_initialized=True
            return features
        normalized={}
        for key,value in features.items():
            if key in ['hour_of_day','day_of_week']:
                normalized[key]=value
            else:
                if key in self.feature_means:
                    alpha=0.01
                    self.feature_means[key]=(1-alpha)*self.feature_means[key]+alpha*value
                    deviation=value-self.feature_means[key]
                    self.feature_stds[key]=np.sqrt((1-alpha)*self.feature_stds[key]**2+alpha*deviation**2)
                else:
                    self.feature_means[key]=value
                    self.feature_stds[key]=1.0
                normalized[key]=(value-self.feature_means[key])/(self.feature_stds[key]+1e-8)
        return normalized

    def _store_training_sample(self,features):
        if len(self.price_data['btc_spot'])==0 or len(self.price_data['btc_perp'])==0:
            return
        spot_price=self.price_data['btc_spot'][-1]['close']
        perp_price=self.price_data['btc_perp'][-1]['close']
        current_basis=(perp_price-spot_price)/(spot_price+1e-8)
        self.training_data['features'].append(features)
        self.training_data['timestamps'].append(self.Time)
        self.training_data['basis_raw'].append(current_basis)
        if len(self.training_data['basis_raw'])>self.prediction_horizon:
            idx=-(self.prediction_horizon+1)
            past_basis=self.training_data['basis_raw'][idx]
            label=current_basis-past_basis
            if len(self.training_data['labels'])==len(self.training_data['features'])-self.prediction_horizon-1:
                self.training_data['labels'].append(label)
                try:
                    prediction_time=self.training_data['timestamps'][idx]
                except IndexError:
                    prediction_time=None
                pred_record=self.prediction_lookup.pop(prediction_time,None) if prediction_time else None
                if pred_record and(pred_record.get('has_xgb')or pred_record.get('has_lstm')):
                    self.realized_history.append({'actual':label,'xgb':pred_record['xgb'] if pred_record.get('has_xgb')else None,'lstm':pred_record['lstm'] if pred_record.get('has_lstm')else None})

    def _train_xgboost(self):
        if not self.xgboost_available or xgb is None:
            self.Debug("Skipping XGBoost")
            return
        try:
            n_samples=min(len(self.training_data['features']),len(self.training_data['labels']))
            if n_samples<self.min_training_samples:
                return
            X,y=[],[]
            for i in range(n_samples):
                X.append(list(self.training_data['features'][i].values()))
                y.append(self.training_data['labels'][i])
            X=np.array(X)
            y=np.array(y)
            split_idx=int(0.8*n_samples)
            X_train,X_val=X[:split_idx],X[split_idx:]
            y_train,y_val=y[:split_idx],y[split_idx:]
            self.xgb_model=xgb.XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=42)
            self.xgb_model.fit(X_train,y_train)
            y_pred=self.xgb_model.predict(X_val)
            mse=np.mean((y_val-y_pred)**2)
            self.Debug("XGB MSE:"+str(mse))
        except Exception as e:
            self.Debug("XGB error:"+str(e))

    def _train_lstm(self):
        if not self.lstm_available or torch is None or nn is None:
            self.Debug("Skipping LSTM")
            return
        try:
            if len(self.training_data['features'])<self.sequence_length+self.min_training_samples:
                return
            X_sequences,y_sequences=[],[]
            for i in range(len(self.training_data['labels'])):
                if i>=self.sequence_length:
                    seq=[]
                    for j in range(i-self.sequence_length,i):
                        seq.append(list(self.training_data['features'][j].values()))
                    X_sequences.append(seq)
                    y_sequences.append(self.training_data['labels'][i])
            if len(X_sequences)<self.min_training_samples:
                return
            X=torch.FloatTensor(X_sequences)
            y=torch.FloatTensor(y_sequences).unsqueeze(1)
            split_idx=int(0.8*len(X))
            X_train,X_val=X[:split_idx],X[split_idx:]
            y_train,y_val=y[:split_idx],y[split_idx:]
            input_size=X.shape[2]
            hidden_size=64
            if self.lstm_model is None:
                self.lstm_model=LSTMModel(input_size,hidden_size)
                self.Debug("Init new LSTM")
            else:
                self.Debug("Warm-start LSTM")
            criterion=nn.MSELoss()
            optimizer=torch.optim.Adam(self.lstm_model.parameters(),lr=0.001)
            epochs=20
            batch_size=32
            for epoch in range(epochs):
                self.lstm_model.train()
                for i in range(0,len(X_train),batch_size):
                    batch_X=X_train[i:i+batch_size]
                    batch_y=y_train[i:i+batch_size]
                    optimizer.zero_grad()
                    outputs=self.lstm_model(batch_X)
                    loss=criterion(outputs,batch_y)
                    loss.backward()
                    optimizer.step()
            self.lstm_model.eval()
            with torch.no_grad():
                val_pred=self.lstm_model(X_val)
                val_mse=criterion(val_pred,y_val).item()
            self.Debug("LSTM MSE:"+str(val_mse))
        except Exception as e:
            self.Debug("LSTM error:"+str(e))
            self.lstm_model=None

    def _update_model_weights(self):
        if len(self.realized_history)<20:
            return
        recent_history=list(self.realized_history)[-20:]
        xgb_pairs=[(e['xgb'],e['actual'])for e in recent_history if e['xgb']is not None]
        lstm_pairs=[(e['lstm'],e['actual'])for e in recent_history if e['lstm']is not None]
        scores={}
        if len(xgb_pairs)>=5:
            xgb_preds,xgb_actuals=zip(*xgb_pairs)
            xgb_errors=np.abs(np.array(xgb_preds)-np.array(xgb_actuals))
            scores['xgb']=-np.mean(xgb_errors)/(np.std(xgb_errors)+1e-8)
        if len(lstm_pairs)>=5:
            lstm_preds,lstm_actuals=zip(*lstm_pairs)
            lstm_errors=np.abs(np.array(lstm_preds)-np.array(lstm_actuals))
            scores['lstm']=-np.mean(lstm_errors)/(np.std(lstm_errors)+1e-8)
        if not scores:
            return
        if len(scores)==1:
            if 'xgb' in scores:
                self.xgb_weight,self.lstm_weight=1.0,0.0
            else:
                self.xgb_weight,self.lstm_weight=0.0,1.0
            return
        exp_xgb=np.exp(scores.get('xgb',0.0))
        exp_lstm=np.exp(scores.get('lstm',0.0))
        total=exp_xgb+exp_lstm
        self.xgb_weight=exp_xgb/total
        self.lstm_weight=exp_lstm/total
        self.Debug("Weights XGB:"+str(self.xgb_weight)+" LSTM:"+str(self.lstm_weight))

    def _generate_predictions(self,features):
        try:
            predictions={}
            has_xgb_prediction=self.xgb_model is not None
            if has_xgb_prediction:
                feature_vector=np.array([[features[key]for key in self.FEATURE_ORDER]])
                xgb_pred=self.xgb_model.predict(feature_vector)[0]
                predictions['xgb']=float(xgb_pred)
            else:
                predictions['xgb']=0.0
            predictions['has_xgb']=has_xgb_prediction
            has_lstm_prediction=self.lstm_model is not None and self.lstm_available and torch is not None and len(self.feature_history)>=self.sequence_length
            if has_lstm_prediction:
                sequence=[]
                for feat_dict in list(self.feature_history)[-self.sequence_length:]:
                    sequence.append([feat_dict[key]for key in self.FEATURE_ORDER])
                seq_tensor=torch.FloatTensor([sequence])
                self.lstm_model.eval()
                with torch.no_grad():
                    lstm_pred=self.lstm_model(seq_tensor)
                predictions['lstm']=float(lstm_pred.item())
            else:
                predictions['lstm']=0.0
            predictions['has_lstm']=has_lstm_prediction
            ensemble_pred=self.xgb_weight*predictions['xgb']+self.lstm_weight*predictions['lstm']
            predictions['ensemble']=ensemble_pred
            conviction=np.tanh(ensemble_pred*10)
            predictions['conviction']=conviction
            self.latest_xgb_prediction=predictions['xgb']
            self.latest_lstm_prediction=predictions['lstm']
            self.latest_ensemble_prediction=ensemble_pred
            if predictions.get('has_xgb')or predictions.get('has_lstm'):
                self.prediction_lookup[self.Time]={'xgb':predictions['xgb'],'lstm':predictions['lstm'],'has_xgb':predictions.get('has_xgb',False),'has_lstm':predictions.get('has_lstm',False)}
                while len(self.prediction_lookup)>self.max_training_samples:
                    try:
                        self.prediction_lookup.popitem(last=False)
                    except KeyError:
                        break
            return predictions
        except Exception as e:
            self.Debug("Prediction error:"+str(e))
            return None

    def _execute_trading_logic(self,predictions):
        conviction=predictions['conviction']
        self.last_conviction=conviction
        if abs(conviction)<self.conviction_threshold:
            return
        position_size=self._compute_position_size(conviction,predictions['ensemble'])
        if position_size==0:
            return
        if conviction>self.conviction_threshold and self.current_position!='long_perp':
            self._enter_long_perp(position_size)
        elif conviction<-self.conviction_threshold and self.current_position!='short_perp':
            self._enter_short_perp(position_size)

    def _compute_position_size(self,conviction,expected_move):
        try:
            portfolio_value=self.Portfolio.TotalPortfolioValue
            btc_perp_prices=np.array([p['close']for p in self.price_data['btc_perp']])
            returns=np.diff(np.log(btc_perp_prices[-25:]))
            realized_vol=np.std(returns)*np.sqrt(24)
            if realized_vol==0:
                return 0
            leverage_factor=min(self.max_leverage,self.volatility_cap)
            move_vol_ratio=abs(expected_move)/realized_vol
            move_vol_ratio=np.clip(move_vol_ratio,0.0,10.0)
            size_factor=abs(conviction)*move_vol_ratio
            position_value=portfolio_value*leverage_factor*size_factor
            min_position=portfolio_value*0.05
            max_position=portfolio_value*self.volatility_cap
            position_value=np.clip(position_value,min_position,max_position)
            if position_value<self.exchange_min_notional:
                return 0
            current_price=btc_perp_prices[-1]
            quantity=position_value/current_price
            lot_size=self.effective_lot_size if self.effective_lot_size>0 else 0.001
            quantity=math.floor(quantity/lot_size)*lot_size
            min_quantity=self.perp_min_quantity if self.perp_min_quantity>0 else lot_size
            if quantity<min_quantity or quantity<=0:
                return 0
            if(quantity*current_price)<self.exchange_min_notional:
                return 0
            return quantity
        except Exception as e:
            self.Debug("Position sizing error:"+str(e))
            return 0

    def _enter_long_perp(self,quantity):
        self.Liquidate()
        self.MarketOrder(self.btc_perp,quantity)
        self.MarketOrder(self.btc_spot,-quantity)
        self.current_position='long_perp'
        self.position_entry_time=self.Time
        btc_spot_price=self.price_data['btc_spot'][-1]['close']
        btc_perp_price=self.price_data['btc_perp'][-1]['close']
        self.position_entry_basis=(btc_perp_price-btc_spot_price)/btc_spot_price
        trade_value=quantity*btc_perp_price
        estimated_cost=trade_value*(self.taker_fee+self.slippage_bps/10000)*2
        self.position_entry_value=self.Portfolio.TotalPortfolioValue
        self.Debug("LONG basis="+str(self.position_entry_basis)+" conv="+str(self.last_conviction)+" cost="+str(estimated_cost))

    def _enter_short_perp(self,quantity):
        self.Liquidate()
        self.MarketOrder(self.btc_perp,-quantity)
        self.MarketOrder(self.btc_spot,quantity)
        self.current_position='short_perp'
        self.position_entry_time=self.Time
        btc_spot_price=self.price_data['btc_spot'][-1]['close']
        btc_perp_price=self.price_data['btc_perp'][-1]['close']
        self.position_entry_basis=(btc_perp_price-btc_spot_price)/btc_spot_price
        trade_value=quantity*btc_perp_price
        estimated_cost=trade_value*(self.taker_fee+self.slippage_bps/10000)*2
        self.position_entry_value=self.Portfolio.TotalPortfolioValue
        self.Debug("SHORT basis="+str(self.position_entry_basis)+" conv="+str(self.last_conviction)+" cost="+str(estimated_cost))

    def _check_exit_conditions(self):
        if self.current_position is None:
            return
        if abs(self.last_conviction)<self.exit_conviction_threshold:
            self._exit_position("conviction_zero")
            return
        if(self.Time-self.position_entry_time).total_seconds()>self.position_max_hours*3600:
            self._exit_position("time_stop")
            return
        current_value=self.Portfolio.TotalPortfolioValue
        drawdown=(self.position_entry_value-current_value)/self.position_entry_value
        if drawdown>self.max_drawdown_per_trade:
            self._exit_position("drawdown_stop")

    def _exit_position(self,reason):
        self.Liquidate()
        pnl=self.Portfolio.TotalPortfolioValue-self.position_entry_value
        self.Debug("EXIT "+str(self.current_position)+" reason="+reason+" PnL="+str(pnl))
        self.current_position=None
        self.position_entry_time=None
        self.position_entry_basis=None
        self.position_entry_value=None

    def _update_plots(self,features):
        if features is None:
            return
        if self.last_plot_time is not None and(self.Time-self.last_plot_time)<self.plot_interval:
            return
        if len(self.price_data['btc_spot'])>=48 and len(self.price_data['btc_perp'])>=48:
            btc_spot_prices=np.array([p['close']for p in self.price_data['btc_spot']][-48:])
            btc_perp_prices=np.array([p['close']for p in self.price_data['btc_perp']][-48:])
            basis_series=(btc_perp_prices-btc_spot_prices)/(btc_spot_prices+1e-8)
            self.Plot("Basis","Basis",basis_series[-1])
            self.Plot("Basis","Basis_MA",np.mean(basis_series))
        if self.model_trained:
            self.Plot("Predictions","XGB_Pred",self.latest_xgb_prediction)
            self.Plot("Predictions","LSTM_Pred",self.latest_lstm_prediction)
            self.Plot("Predictions","Ensemble_Pred",self.latest_ensemble_prediction)
        self.Plot("Conviction","Conviction",self.last_conviction)
        self.Plot("Conviction","Threshold",self.conviction_threshold)
        if self.current_position is not None:
            unrealized_pnl=self.Portfolio.TotalPortfolioValue-self.position_entry_value
            self.Plot("Trade_PnL","Unrealized_PnL",unrealized_pnl)
        self.last_plot_time=self.Time
