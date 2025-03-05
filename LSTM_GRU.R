library(keras3)
library(tensorflow)
library(tibble)
library(dplyr)
library(ggplot2)
library(purrr)

# Simulate time series data
generate_data <- function(n = 1000) {
  time <- 1:n
  value <- sin(time * 0.02) + rnorm(n, sd = 0.2)
  return(data.frame(time = time, value = value))
}

data <- generate_data()

# Create sequences for training
create_sequences <- function(data, seq_length) {
  X <- array(NA, dim = c(nrow(data) - seq_length, seq_length, 1))
  y <- array(NA, dim = c(nrow(data) - seq_length, 1))
  for (i in 1:(nrow(data) - seq_length)) {
    X[i,,1] <- data$value[i:(i + seq_length - 1)]
    y[i] <- data$value[i + seq_length]
  }
  return(list(X = X, y = y))
}

# Define model creation function
build_model <- function(num_layers, units, learning_rate, dropout_rate, seq_length) {
  model <- keras_model_sequential()
  
  # Add LSTM and GRU layers dynamically based on num_layers
  for (i in 1:num_layers) {
    if (i == 1) {
      model %>% layer_lstm(units = units, return_sequences = TRUE, input_shape = c(seq_length, 1))
    } else {
      model %>% layer_gru(units = units, return_sequences = (i != num_layers))
    }
    model %>% layer_dropout(rate = dropout_rate)
  }
  
  model %>% layer_dense(units = 1)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = 'mse'
  )
  return(model)
}

# Random search hyperparameter tuning
random_search <- function(n_iter = 10) {
  results <- list()
  
  for (i in 1:n_iter) {
    params <- list(
      num_layers = sample(1:4, 1),
      units = sample(c(16, 32, 64, 128, 256, 512, 1024), 1),
      learning_rate = runif(1, 0.0001, 0.01),
      batch_size = sample(c(16, 32, 64), 1),
      dropout_rate = runif(1, 0.1, 0.5),
      seq_length = sample(c(10, 20, 30, 40, 50, 60), 1)
    )
    
    # Create sequences
    sequences <- create_sequences(data, params$seq_length)
    X_train <- sequences$X
    y_train <- sequences$y
    
    # Build and train model
    model <- build_model(params$num_layers, params$units, params$learning_rate, params$dropout_rate, params$seq_length)
    history <- model %>% fit(X_train, y_train, epochs = 20, batch_size = params$batch_size, verbose = 0)
    
    # Store results
    results[[i]] <- list(params = params, loss = min(history$metrics$loss), model = model)
  }
  
  return(results)
}

# Run random search
tuning_results <- random_search(10)

# Select best model
best_result <- tuning_results %>% purrr::reduce(~ if (.x$loss < .y$loss) .x else .y)

# Evaluate and visualize final model
best_model <- best_result$model
sequences <- create_sequences(data, best_result$params$seq_length)
X_test <- sequences$X
y_test <- sequences$y
predictions <- best_model %>% predict(X_test)

ggplot(data.frame(time = (best_result$params$seq_length + 1):1000, actual = y_test, predicted = predictions),
       aes(x = time)) +
  geom_line(aes(y = actual), color = 'blue') +
  geom_line(aes(y = predicted), color = 'red', linetype = 'dashed') +
  ggtitle("Stacked LSTM-GRU Model Predictions vs Actual")
