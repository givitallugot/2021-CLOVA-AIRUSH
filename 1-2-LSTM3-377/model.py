import tensorflow as tf
import nsml

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each timestep is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta

class LSTM_Model(tf.keras.Model):
    def __init__(self, optimizer, loss, metrics, epochs, batch_size):
        super(LSTM_Model, self).__init__()

        # self.dense = ResidualWrapper(
        #     tf.keras.Sequential([
        #     tf.keras.layers.LSTM(32, return_sequences=True),
        #     tf.keras.layers.Dense(1,
        #         # The predicted deltas should start small
        #         # So initialize the output layer with zeros
        #         kernel_initializer=tf.initializers.zeros)
        # ]))

        # model, batch_input_shape=(batch_size, 1, 50), statefull=True 추가
        # self.dense = tf.keras.Sequential([
        #     tf.keras.layers.LSTM(units=32, return_sequences=True, activation='relu'),
        #     tf.keras.layers.Dense(units=1)
        # ])

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error',patience=2, mode='min')

        self.lstm_input = tf.keras.layers.Input(shape=(1,56)) # 4: 변수 개수, batch_shape=(self.batch_size, 1, 4)
        self.lstm_out = tf.keras.layers.LSTM(32, activation='relu')(self.lstm_input) #  stateful=True, return_sequences=True (이거 추가하면 TimeDistributed(Dense(300))), stateful=True,(batch_size 명시) 
        self.dense_output = tf.keras.layers.Dense(1)(self.lstm_out)

        self.dense = tf.keras.models.Model(inputs=self.lstm_input, outputs=self.dense_output)

        self.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        # training variables
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, train_data):
        # Group training_data into batches
        train_data_batch = train_data.batch(self.batch_size)
        self.fit(train_data_batch, epochs = self.epochs, callbacks = [self.early_stopping]) # shuffle false 넣어야함

        # Save model for nsml
        nsml.save(1)
    
    def call(self, inputs):
        return self.dense(inputs)