import numpy as np
import pandas as pd
import tensorflow as tf
import os
from haversine import haversine

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

# CREATE YOUR OWN PREPROCESSOR FOR YOUR MODEL
class Preprocessor():
    def __init__(self):
        self.normalizer = Normalizer()
        self.train_data_path = os.path.join(DATASET_PATH, "train", "train_data", "data")
        self.train_label_path = os.path.join(DATASET_PATH, "train", "train_label")
        #self.numcols = ["prev_duration", "prev_station_distance", "next_station_distance",  "prev_station_lng", "prev_station_lat", "next_station_lng", "next_station_lat", "station_lng", "station_lat"]
        self.numcols = ["ts", "operation_id", "prev_velocity", "prev_duration", "next_direct_distance", "next_station_distance", "diff_station_id"]
        # self.catcols = ["hour", "dow", "prev_duration_bin", "prev_station_distance_bin", "next_station_distance_bin", "station_id_bin"]

    def _load_train_dataset(self, train_data_path = None):
        train_data = pd.read_parquet(self.train_data_path) \
            .sort_values(by = ["route_id", "plate_no", "operation_id", "station_seq"], ignore_index = True)
        train_label = pd.read_csv(self.train_label_path, header = None, names = LABEL_COLUMNS) \
            .sort_values(by = ["route_id", "plate_no", "operation_id", "station_seq"], ignore_index = True)
        
        return train_data, train_label


    def _compute_direct_distance(self, next_lat, next_lng, prev_lat, prev_lng):
        return haversine((float(next_lat), float(next_lng)), (float(prev_lat), float(prev_lng)))

    def _series_to_supervised_train(self, data, n_in=7, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg

    def _series_to_supervised_test(self, data, n_in=7, n_out=1, dropnan=True): # 일단 변수명 없움..
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        if df.shape[0] <= n_in:
            ilist = list(range(df.shape[0]-1, 0, -1)) + [0]*(n_in-df.shape[0]+1)
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(ilist[(n_in-i)])) 
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df)
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        else:
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg.tail(1)


    def preprocess_train_dataset(self):
        train_data, train_label = self._load_train_dataset()
        
        print(f"Loaded total data count: {len(train_data)}")

        train_data['next_direct_distance'] = train_data[['next_station_lat', 'next_station_lng', 'station_lat', 'station_lng']].apply(lambda x: self._compute_direct_distance(x[0], x[1], x[2], x[3]), axis=1)
        train_data["prev_velocity"] = train_data[['prev_station_distance', 'prev_duration']].apply(lambda x: np.nan if x[1] == 0 else x[0]/x[1], axis=1)
        train_data["prev_velocity"] = train_data["prev_velocity"].fillna(method='bfill')
        # train_data['next_curvature'] = train_data['next_station_distance']/(train_data['next_direct_distance']*100)
        # train_data['next_duration_pred'] = train_data['next_station_distance']*train_data['prev_velocity']
        train_data['diff_station_id'] = train_data['next_station_id'] - train_data['station_id']

        train_data = train_data[self.numcols]
        train_label = train_label[['next_duration']]

        # Normalize(minmaxscaler)
        self.normalizer.build(train_data)
        train_data = self.normalizer.normalize(train_data)
        train_data = self._series_to_supervised_train(train_data)
        train_label = train_label[7:] 

        print(train_data.shape, train_label.shape)

        train_data = np.array(train_data).astype('float32')
        train_data = np.reshape(train_data, ([train_data.shape[0], 1, train_data.shape[1]]))
        train_label = np.array(train_label).astype('float32')

        dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(len(train_data))
        return dataset

    def preprocess_test_data(self, test_data):
        
        test_data['next_direct_distance'] = test_data[['next_station_lat', 'next_station_lng', 'station_lat', 'station_lng']].apply(lambda x: self._compute_direct_distance(x[0], x[1], x[2], x[3]), axis=1)
        test_data["prev_velocity"] = test_data[['prev_station_distance', 'prev_duration']].apply(lambda x: np.nan if x[1] == 0 else x[0]/x[1], axis=1)
        test_data["prev_velocity"] = test_data["prev_velocity"].fillna(method='bfill')
        # train_data['next_curvature'] = train_data['next_station_distance']/(train_data['next_direct_distance']*100)
        # train_data['next_duration_pred'] = train_data['next_station_distance']*train_data['prev_velocity']
        test_data['diff_station_id'] = test_data['next_station_id'] - test_data['station_id']

        # Select Columns
        test_data = test_data[self.numcols]

        # Normalize
        test_data = self.normalizer.normalize(test_data)
        test_data = self._series_to_supervised_test(test_data)
        print(test_data)

        test_data = np.array(test_data).astype('float32')
        test_data = np.reshape(test_data, ([test_data.shape[0], 1, test_data.shape[1]]))

        dataset = tf.data.Dataset.from_tensor_slices(test_data)
        return dataset