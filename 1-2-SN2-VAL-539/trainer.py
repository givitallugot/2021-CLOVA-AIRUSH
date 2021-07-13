from preprocessor import Preprocessor

class Trainer():
    def __init__(self, model):
        self.preprocessor = Preprocessor()
        self.model = model

    def train(self):
        train_dataset, validate_dataset = self.preprocessor.preprocess_train_dataset()
        self.model.train(train_dataset, validate_dataset) 

        # model summary
        self.model.summary()