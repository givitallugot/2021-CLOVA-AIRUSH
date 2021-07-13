import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from haversine import haversine
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
        self.numcols1 = ["next_curvature", "next_direct_distance", "prev_station_lng", "prev_station_lat", "station_lng", "station_lat", "next_station_lng", "next_station_lat"] # "next_duration_pred", "station_id"
        self.numcols2 = ["ts", "operation_id", "prev_velocity", "prev_duration", "prev_station_distance"] # , "prev_station_seq"
        self.catcols = ["hour", "dow"] # ,"prev_duration_bin", "prev_station_distance_bin", "next_station_distance_bin", "station_id_bin"

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

        return agg

    def preprocess_train_dataset(self):
        train_data, train_label = self._load_train_dataset()
        train_label = train_label["next_duration"]

        # Make Continuous
        train_data['next_direct_distance'] = train_data[['next_station_lat', 'next_station_lng', 'station_lat', 'station_lng']].apply(lambda x: self._compute_direct_distance(x[0], x[1], x[2], x[3]), axis=1)

        train_data["prev_velocity"] = train_data[['prev_station_distance', 'prev_duration']].apply(lambda x: np.nan if x[1] == 0 else x[0]/x[1], axis=1)
        train_data["prev_velocity"] = train_data["prev_velocity"].fillna(method='bfill')
        
        train_data['next_curvature'] = train_data['next_station_distance']/(train_data['next_direct_distance']*100)
        train_data['next_duration_pred'] = train_data['next_station_distance']*train_data['prev_velocity']
        
        # Make Binary
        # train_data["prev_duration_bin"] = train_data["prev_duration"].apply(lambda x: 1 if x > 330 else 0)
        # train_data["prev_station_distance_bin"] = train_data["prev_station_distance"].apply(lambda x: 1 if x > 1100 else 0)
        # train_data["next_station_distance_bin"] = train_data["next_station_distance"].apply(lambda x: 1 if x > 1200 else 0)
        # train_data["station_id_bin"] = train_data["station_id"].apply(lambda x: 1 if x > 125000 else 0)
        # train_data["hours"] = train_data['hour'].apply(lambda x: 1 if 7<=x<=9 else (2 if 10<=x<=17 else (3 if 18<=x<=20 else (4 if 20<=x<=23 else 5))) )
        # train_data["worw"] = train_data['dow'].apply(lambda x: 1 if 6<=x<=7 else 0) 

        # Seperate Scaler for Numeric and Categorical
        train_data_num = train_data[self.numcols1+self.numcols2]
        train_data_cat = train_data[self.catcols]

        self.normalizer.build(train_data_num)
        train_data_num = self.normalizer.normalize(train_data_num)

        self.ohencoder.build(train_data_cat)
        train_data_cat = self.ohencoder.onehotencoding(train_data_cat)

        # Concate & Seperate Two Inputs
        train_data1 = pd.concat([train_data_num[self.numcols1], train_data_cat], axis=1)
        train_data2 = train_data_num[self.numcols2]

        train_data2 = self._series_to_supervised_train(train_data2)
        train_data1 = train_data1[7:] 
        train_label = train_label[7:] 

        print(f"Loaded total data count: {len(train_data1)}")

        train_data1 = np.array(train_data1).astype('float32')
        train_data2 = np.array(train_data2).astype('float32')
        train_data2 = np.reshape(train_data2, ([train_data2.shape[0], 1, train_data2.shape[1]]))
        train_label = np.array(train_label).astype('float32')
        print(train_data1.shape, train_data2.shape)

        train_data = tf.data.Dataset.from_tensor_slices(({'input_1': train_data1, 'input_2': train_data2})) # .shuffle(len(train_data))
        train_label = tf.data.Dataset.from_tensor_slices(train_label)

        dataset = tf.data.Dataset.zip((train_data, train_label))
        return dataset
    
    def preprocess_test_data(self, test_data):

        # Make Continuous
        test_data['next_direct_distance'] = test_data[['next_station_lat', 'next_station_lng', 'station_lat', 'station_lng']].apply(lambda x: self._compute_direct_distance(x[0], x[1], x[2], x[3]), axis=1)

        test_data["prev_velocity"] = test_data[['prev_station_distance', 'prev_duration']].apply(lambda x: np.nan if x[1] == 0 else x[0]/x[1], axis=1)
        test_data["prev_velocity"] = test_data["prev_velocity"].fillna(method='bfill')
        
        test_data['next_curvature'] = test_data['next_station_distance']/(test_data['next_direct_distance']*100)
        test_data['next_duration_pred'] = test_data['next_station_distance']*test_data['prev_velocity']
        

        # Make Binary
        # test_data_["prev_duration_bin"] = test_data["prev_duration"].apply(lambda x: 1 if x > 330 else 0)
        # test_data_["prev_station_distance_bin"] = test_data["prev_station_distance"].apply(lambda x: 1 if x > 1100 else 0)
        # test_data_["next_station_distance_bin"] = test_data["next_station_distance"].apply(lambda x: 1 if x > 1200 else 0)
        # test_data_["station_id_bin"] = test_data["station_id"].apply(lambda x: 1 if x > 125000 else 0)
        # test_data["hours"] = test_data['hour'].apply(lambda x: 1 if 7<=x<=9 else (2 if 10<=x<=17 else (3 if 18<=x<=20 else (4 if 20<=x<=23 else 5))) )
        # test_data["worw"] = test_data['dow'].apply(lambda x: 1 if 6<=x<=7 else 0) 

        # Seperate Scaler for Numeric and Categorical
        test_data_num = test_data[self.numcols1+self.numcols2]
        test_data_cat = test_data[self.catcols]

        test_data_num = self.normalizer.normalize(test_data_num)

        test_data_cat = self.ohencoder.onehotencoding(test_data_cat)

        # Concate & Seperate Two Inputs
        test_data_num = test_data_num.reset_index(drop=True)
        test_data1 = pd.concat([test_data_num[self.numcols1], test_data_cat], axis=1).tail(1)
        
        test_data2 = test_data_num[self.numcols2]
        test_data2 = self._series_to_supervised_test(test_data2).tail(1)

        test_data1 = np.array(test_data1).astype('float32')
        test_data2 = np.array(test_data2).astype('float32')
        test_data2 = np.reshape(test_data2, ([test_data2.shape[0], 1, test_data2.shape[1]]))

        dataset = tf.data.Dataset.from_tensor_slices(({'input_1': test_data1, 'input_2': test_data2})) 
        
        return dataset
