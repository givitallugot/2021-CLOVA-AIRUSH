from sklearn.utils import validation
import tensorflow as tf
import nsml


class BaselineModel(tf.keras.Model):
    def __init__(self, optimizer, loss, metrics, epochs, batch_size):
        super(BaselineModel, self).__init__()
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',patience=2, mode='min')

        # model
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
            # 
        ])
        self.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        # training variables
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, train_data, validate_data):
        # Group training_data into batches
        train_data_batch = train_data.batch(self.batch_size)
        validate_data_batch = validate_data.batch(self.batch_size)
        self.fit(train_data_batch, epochs = self.epochs, validation_data=validate_data_batch, callbacks = [self.early_stopping])

        # Save model for nsml
        nsml.save(1)
    
    def call(self, inputs):
        return self.dense(inputs)
