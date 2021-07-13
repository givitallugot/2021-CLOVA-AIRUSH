import tensorflow as tf
import nsml

class BaselineModel(tf.keras.Model):
    def __init__(self, optimizer, loss, metrics, epochs, batch_size):
        super(BaselineModel, self).__init__()

        # training variables
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error',patience=2, mode='min')

        # model
        self.mlp_input = tf.keras.layers.Input(shape=(37,)) # 32: 변수 개수, batch_shape=(self.batch_size, 34,)
        self.hidden1 = tf.keras.layers.Dense(64, activation='relu')(self.mlp_input)
        self.mlp_output = tf.keras.layers.Dense(64, activation='relu')(self.hidden1)
        self.mlp_model = tf.keras.models.Model(inputs=self.mlp_input, outputs=self.mlp_output)

        self.lstm_input = tf.keras.layers.Input(shape=(1,40)) # 4: 변수 개수, batch_shape=(self.batch_size, 1, 4)
        self.lstm_out = tf.keras.layers.LSTM(32, activation='relu')(self.lstm_input) #  stateful=True, return_sequences=True (이거 추가하면 TimeDistributed(Dense(300))), stateful=True,(batch_size 명시) 
        self.lstm_model = tf.keras.models.Model(inputs=self.lstm_input, outputs=self.lstm_out)

        self.concatenated = tf.keras.layers.concatenate([self.mlp_model.output, self.lstm_model.output])
        self.concatenated = tf.keras.layers.Dense(32, activation='relu')(self.concatenated)
        # self.concatenated = tf.keras.layers.BatchNormalization()(self.concatenated)
        self.concat_out = tf.keras.layers.Dense(1)(self.concatenated)

        self.model = tf.keras.models.Model([self.lstm_input, self.mlp_input], self.concat_out)

        self.compile(optimizer = optimizer, loss = loss, metrics = metrics) # callbacks=[self.early_stopping]

    def train(self, train_data):
        # Group training_data into batches
        train_data_batch = train_data.batch(self.batch_size, drop_remainder=True)
        self.fit(train_data_batch, epochs = self.epochs, callbacks = [self.early_stopping])

        # Save model for nsml
        nsml.save(1)
    
    def call(self, inputs):
        return self.model(inputs)
