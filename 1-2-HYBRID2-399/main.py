import argparse
import os
import tensorflow as tf
import pickle

from pandas.core.frame import DataFrame

from trainer import Trainer
from model import BaselineModel

import nsml

def bind_model(trainer: Trainer):
    def save(dirname, *args):
        trainer.model.save(os.path.join(dirname, 'model'))
        with open(os.path.join(dirname, "preprocessor.pckl"), "wb") as f:
            pickle.dump(trainer.preprocessor, f)

    def load(dirname, *args):
        trainer.model = tf.keras.models.load_model(os.path.join(dirname, 'model'))
        with open(os.path.join(dirname, "preprocessor.pckl"), "rb") as f:
            trainer.preprocessor = pickle.load(f)

    def infer(test_data : DataFrame):
        test_data = trainer.preprocessor.preprocess_test_data(test_data).batch(1)
        prediction = trainer.model.predict(test_data).flatten().tolist()

        # single prediction for last next duration
        return round(prediction[0]) 

    nsml.bind(save = save, load = load, infer = infer)

def main():
    args = argparse.ArgumentParser()

    # RESERVED FOR NSML
    args.add_argument('--mode', type=str, default='train', help='nsml submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # ADD MORE ARGS IF NECESSARY

    # NSML - Bind Model
    # Building model and preprocessor
    model = BaselineModel(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), 
                        loss = tf.losses.MeanSquaredError(), 
                        metrics = [tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()], 
                        epochs = 20, batch_size = 32) # 50

    trainer = Trainer(model)

    # nsml.bind() should be called before nsml.paused()
    bind_model(trainer)
    if config.pause:
        nsml.paused(scope=locals())

    # Train Model
    if config.mode == "train":
        print("Training model...")
        trainer.train()
    

if __name__ == '__main__':
    main()
