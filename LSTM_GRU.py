#######################################
#       LSTM-GRU model                #
#######################################

# Prepare data for time series forecasting

data=r.BTC

lookback = 20  # Number of previous time steps to use for prediction
delay = 1      # Number of time steps ahead to predict

# Function to create sequences for time series
def create_sequences(data, lookback, delay):
    X, y = [], []
    for i in range(len(data) - lookback - delay + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback + delay - 1])
    return np.array(X), np.array(y)

# Create training and validation datasets
split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

X_train, y_train = create_sequences(train_data, lookback, delay)
X_val, y_val = create_sequences(val_data, lookback, delay)
```

# model bulding

def build_model(hp):
    model = Sequential()
    
    # Tune the number of LSTM units
    hp_lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=hp_lstm_units, return_sequences=True, input_shape=(lookback, 1)))
    
    # Tune the number of GRU units
    hp_gru_units = hp.Int('gru_units', min_value=32, max_value=128, step=32)
    model.add(GRU(units=hp_gru_units))
    
    # Tune the number of dense units
    hp_dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(Dense(units=hp_dense_units, activation='relu'))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='mse'
    )
    
    return model

#############################################
#  Random search for hyperparameter tuning  #
#############################################

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Number of hyperparameter combinations to try
    executions_per_trial=2,  # Number of models to train per trial
    directory='tuner_results',
    project_name='lstm_gru_tuning'
)

# Perform hyperparameter tuning
tuner.search(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

#   get best model

best_model=tuner.get_best_models(num_models=0)

# fit best model

history = best_model.fit( X_train, y_train,epochs=100,  # Train for more epochs with the best hyperparameters
     batch_size=32, validation_data=(X_val, y_val), verbose=0)

y_pred = best_model.predict(X_val)
