import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data



def handle_missing_values(data):
    # First : selecting categorical and numerical columns
    numerical_columns = data.select_dtypes(include =['float64', 'int64'] ).columns
    categorical_cols = data.select_dtypes(include = ['object']).columns

    # Second : Imputing missing numerical values with the median 
    imputer = SimpleImputer(strategy='median')
    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    # also imputing the categorical columns with the most frequent value
    imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer.fit_transform(data[categorical_cols])

    return data
    


def encode_categorical_features(data):
    #selecting categorical columns
    categorical_cols = data.select_dtypes(include = ['object']).columns

    # OneHotEncoder for categorical variables
    encoder = OneHotEncoder(drop='first'  , sparse_output=False,dtype=np.int32)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]) )
    encoded_data.columns = encoder.get_feature_names_out(categorical_cols)

    # drop the original columns and join the encoded columns
    data = data.drop(categorical_cols , axis= 1)
    data= pd.concat([data, encoded_data] , axis=1)

    return data



def scale_numerical_features(data):
    num_cols = data.select_dtypes(include=['float64' , 'int64']).columns

    # Scaling numerical features by applying StandardScaler  method
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    return data

def split_data(data,target_column , test_size=0.2, random_state=42):
    x= data.drop(target_column , axis=1).to_numpy()
    y = data[target_column].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test




def preprocess_data(filepath , target_column):

    # 1- Load Data
    data = load_data(filepath)

    # 2- Handle Missing Values
    data = handle_missing_values(data)

    # 3- Scale numerical features
    data = scale_numerical_features(data)

    # 4- Encode categorical variables 
    data = encode_categorical_features(data)

    # 5- Split data
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    return X_train, X_test, y_train, y_test