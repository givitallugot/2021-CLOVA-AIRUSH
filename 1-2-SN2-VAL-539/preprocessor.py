import numpy as np
import pandas as pd
import tensorflow as tf
from haversine import haversine
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os

from nsml import DATASET_PATH

LABEL_COLUMNS = ["route_id", "plate_no", "operation_id", "station_seq", "next_duration"]

class Normalizer():
    def __init__(self):
        self.train_mean = None
        self.train_std = None
    
    def build(self, train_data):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()
    
    def normalize(self, df):
        return (df - self.train_mean) / self.train_std

class OHEncoder():
    def __init__(self):
        self.encoder = None

    def build(self, train_data):
        self.encoder = OneHotEncoder(handle_unknown = 'ignore').fit(train_data)

    def onehotencoding(self, df):
        return pd.DataFrame(self.encoder.transform(df).toarray())


# CREATE YOUR OWN PREPROCESSOR FOR YOUR MODEL
class Preprocessor():
    def __init__(self):
        self.normalizer = Normalizer()
        self.ohencoder = OHEncoder()
        self.train_data_path = os.path.join(DATASET_PATH, "train", "train_data", "data")
        self.train_label_path = os.path.join(DATASET_PATH, "train", "train_label")
        self.numcols = ["next_duration_pred", "prev_station_distance", "next_station_distance", "prev_station_lng", "prev_station_lat", "next_station_lng", "next_station_lat", "station_lng", "station_lat"] #  "prev_station_lng", "prev_station_lat", 
        self.catcols = ["hour", "dow", "next_curvature_bin", "prev_duration_bin"] # "hour", "dow", 

    def _load_train_dataset(self, train_data_path = None):
        train_data = pd.read_parquet(self.train_data_path) \
            .sort_values(by = ["route_id", "plate_no", "operation_id", "station_seq"], ignore_index = True)
        train_label = pd.read_csv(self.train_label_path, header = None, names = LABEL_COLUMNS) \
            .sort_values(by = ["route_id", "plate_no", "operation_id", "station_seq"], ignore_index = True)
        
        return train_data, train_label

    def _compute_direct_distance(self, next_lat, next_lng, prev_lat, prev_lng):
        return haversine((float(next_lat), float(next_lng)), (float(prev_lat), float(prev_lng)))

    def preprocess_train_dataset(self):
        train_data, train_label = self._load_train_dataset()
        train_label = train_label["next_duration"]

        train_data['next_direct_distance'] = train_data[['next_station_lat', 'next_station_lng', 'station_lat', 'station_lng']].apply(lambda x: self._compute_direct_distance(x[0], x[1], x[2], x[3]), axis=1)
        train_data['next_curvature'] = train_data['next_station_distance']/(train_data['next_direct_distance']*100)

        train_data["prev_velocity"] = train_data[['prev_station_distance', 'prev_duration']].apply(lambda x: np.nan if x[1] == 0 else x[0]/x[1], axis=1)
        train_data["prev_velocity"] = train_data["prev_velocity"].fillna(method='bfill')
        
        train_data['next_duration_pred'] = train_data['next_station_distance']*train_data['prev_velocity']
        train_data["prev_station_distance2"] = train_data["prev_station_distance"]*2
        train_data["next_station_distance2"] = train_data["next_station_distance"]*2
        
        # Make Binary
        train_data['dows'] = train_data['dow'].apply(lambda x: 'WD' if 6<=x<=7 else ('M' if x==1 else ('F' if x==5 else 'E')))
        train_data['hours'] = train_data['hour'].apply(lambda x: 1 if 7<=x<=9 else (2 if 10<=x<=14 else (3 if 15<=x<=17 else (4 if 18<=x<=20 else (5 if 21<=x<=23 else 6)))) )
        train_data['dowhour'] = train_data['dows']*train_data['hours']

        train_data["prev_duration_bin"] = train_data["prev_duration"].apply(lambda x: 1 if x > 330 else 0)
        # train_data["prev_station_distance_bin"] = train_data["prev_station_distance"].apply(lambda x: 1 if x > 1100 else 0)
        # train_data["next_station_distance_bin"] = train_data["next_station_distance"].apply(lambda x: 1 if x > 1200 else 0)
        # train_data["station_id_bin"] = train_data["station_id"].apply(lambda x: 1 if x > 125000 else 0)
        train_data["next_curvature_bin"] = train_data["next_curvature"].apply(lambda x: 1 if x > 40 else 0)

        # Seperate Scaler for Numeric and Categorical
        train_data_num = train_data[self.numcols]
        train_data_cat = train_data[self.catcols]

        self.normalizer.build(train_data_num)
        train_data_num = self.normalizer.normalize(train_data_num)

        self.ohencoder.build(train_data_cat)
        train_data_cat = self.ohencoder.onehotencoding(train_data_cat)

        print(train_data_num.shape, train_data_cat.shape)

        # Concate
        train_data = pd.concat([train_data_num, train_data_cat], axis=1)

        print(f"Loaded total data count: {len(train_data)}")

        X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_label, test_size=0.33)

        dataset_train = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values)).shuffle(len(X_train))
        dataset_validation = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values)).shuffle(len(X_train))

        # split = 3
        # dataset_train = dataset.window(split, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds)) #.flat_map(lambda ds: ds)
        # dataset_validation = dataset.skip(split).window(1, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds)) #.flat_map(lambda ds: ds)
        
        return dataset_train, dataset_validation
    

    def preprocess_test_data(self, test_data):

        test_data_ = test_data.tail(5).copy()
        
               
        test_data_['next_direct_distance'] = test_data[['next_station_lat', 'next_station_lng', 'station_lat', 'station_lng']].apply(lambda x: self._compute_direct_distance(x[0], x[1], x[2], x[3]), axis=1)
        test_data_['next_curvature'] = test_data['next_station_distance']/(test_data_['next_direct_distance']*100)
        
        test_data_["prev_velocity"] = test_data[['prev_station_distance', 'prev_duration']].apply(lambda x: np.nan if x[1] == 0 else x[0]/x[1], axis=1)
        test_data_["prev_velocity"] = test_data_["prev_velocity"].fillna(method='bfill')
        
        test_data_['next_duration_pred'] = test_data['next_station_distance']*test_data_['prev_velocity']
        test_data_["prev_station_distance2"] = test_data["prev_station_distance"]*2
        test_data_["next_station_distance2"] = test_data["next_station_distance"]*2
        
        # Make Binary
        test_data_['dows'] = test_data['dow'].apply(lambda x: 'WD' if 6<=x<=7 else ('M' if x==1 else ('F' if x==5 else 'E')))
        test_data_['hours'] = test_data['hour'].apply(lambda x: 1 if 7<=x<=9 else (2 if 10<=x<=14 else (3 if 15<=x<=17 else (4 if 18<=x<=20 else (5 if 21<=x<=23 else 6)))) )
        test_data_['dowhour'] = test_data_['dows']*test_data_['hours']

        test_data_["prev_duration_bin"] = test_data["prev_duration"].apply(lambda x: 1 if x > 330 else 0)
        # test_data_["prev_station_distance_bin"] = test_data["prev_station_distance"].apply(lambda x: 1 if x > 1100 else 0)
        # test_data_["next_station_distance_bin"] = test_data["next_station_distance"].apply(lambda x: 1 if x > 1200 else 0)
        # test_data_["station_id_bin"] = test_data["station_id"].apply(lambda x: 1 if x > 125000 else 0)
        test_data_["next_curvature_bin"] = test_data_["next_curvature"].apply(lambda x: 1 if x > 40 else 0)


        # Seperate Scaler for Numeric and Categorical
        test_data_num = test_data_[self.numcols]
        test_data_cat = test_data_[self.catcols]

        test_data_num = self.normalizer.normalize(test_data_num)

        test_data_cat = self.ohencoder.onehotencoding(test_data_cat)

        # Concate
        test_data_num = test_data_num.reset_index(drop=True)
        test_data_ = pd.concat([test_data_num, test_data_cat], axis=1).tail(1)

        dataset = tf.data.Dataset.from_tensor_slices(test_data_.values)
        return dataset
