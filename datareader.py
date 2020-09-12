from . import arffreader as arff
from . import discretizer as disc
# from Misc import globals

import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, zero_one_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Start by printing the welcome message
# globals.printWelcomeMessage()

# -------------------------------------------
# Global paths
# -------------------------------------------

#dataset_name = ['breastCancer.txt', 'car.txt', 'covtypetx.txt', 'adult.txt', 'adult.arff', 'covtype.arff',
#                'nursery.arff']

# dataset_name = ['car.arff', 'iris.arff', 'half.arff']
# d_index = 2
# path = globals.data_path +'{}'.format(dataset_name[d_index])

def read_uci(path, ALL_DATA_FOR_DISCRETIZE = False,
                   DISCRETIZE = True,
                   ONEHOTENCODE = False):
    '''
    read and preprocess .arff format uci data;
    maybe can also process .txt or .csv data.

    code by nayyar

    Param
    ---------------
    path: 
        string that indicate .arff file location.

    ALL_DATA_FOR_DISCRETIZE: 
        if True, copy all data to train as well as test; 
        if False, split train/test using a fixed random seed and split rate.

    DISCRETIZE: 
        if True, discretize numerical attributes.

    ONEHOTENCODE: 
        if True, encode all attribute into onehot. ONLY affect when DISCRETIZE is True.

    Return
    ---------------
    (X_train, X_test, y_train, y_test)

    '''
    if '.arff' in path:
    
        print(' ************************ ')
        print('    Arff detected    ')
        print(' ************************ ')
    
        data, dict1 = arff.read_arff(path)
    
        cat_features = [i for i in data.columns.values if i in dict1.keys()]
        num_features = [i for i in data.columns.values if i not in dict1.keys()]
    
    elif '.txt' in path:
    
        print(' ************************ ')
        print('    txt detected          ')
        print(' ************************ ')
    
        data, dict1 = arff.read_txt(path)
    
        cat_features = [i for i in data.columns.values if i in dict1.keys()]
        num_features = [i for i in data.columns.values if i not in dict1.keys()]
    # This segment can not work
    # elif '.csv' in path:
    # 
    #     print(' ************************************ ')
    #     print('    csv detected (All Numeric)        ')
    #     print(' ************************************ ')
    # 
    #     data = pd.read_csv(path, header=None, index_col=None)
    # 
    #     cat_features = []
    #     num_features = [i for i in data.columns.values]
    # fail if encounter other files
    else:
        return None        
    
    # --------------------------------------------------------------------------
    # N and n of the data
    # --------------------------------------------------------------------------

    N, n = data.shape
    print('Loaded Data of Size = {}, Dimensions = {}'.format(N, n))
    
    # -------------------------------------------------------------------------------------------------------------------
    # Sanity checking
    # 1) remove columns that have just one unique value
    # 2) delete columns that have as many values as the number of data points
    # -------------------------------------------------------------------------------------------------------------------
    
    feat_set_to_del = [i for i in data.columns.values if
                       (len(np.unique(data[i])) == 1 or len(np.unique(data[i])) == data.shape[0])]
    
    print('Dropping these features {} after loading:'.format(feat_set_to_del))
    
    data = data.drop(feat_set_to_del, axis=1)
    
    for i in feat_set_to_del:
        if i in cat_features:
            cat_features.remove(i)
            dict1.pop(i)
        if i in num_features:
            num_features.remove(i)
    
    print('After cleansing -- Num Features = {}, Cat Features = {}'.format(len(num_features), len(cat_features)))
    
    # Dividing data into training and testing:
    feats_sel = [data.columns.values[i] for i in range(0,len(data.columns.values) - 1)]
    class_col = data.columns.values[len(data.columns.values) - 1]

    data_all = data[feats_sel]
    labels_all = data[class_col]
    
    if ALL_DATA_FOR_DISCRETIZE:  
        data_train = data_all
        data_test = data_all
        labels_train = labels_all
        labels_test = labels_all
    else:    
        data_train, data_test, labels_train, labels_test = train_test_split(data_all, labels_all, test_size=0.2,
                                                                            random_state=51214)
         
    if DISCRETIZE:
        params_disc = {'num_bins': 10, 'method': 'EF', 'feats_sel': num_features, 'dict': dict1, 'labels': labels_train,
                       'treat_missing_distinct': True}
        data_train_disc, data_test_disc, dict, cutPoints = disc.discretize(data_train, data_test, **params_disc)
        
        if ONEHOTENCODE:        
            # -------------------------------------------------------------------------------------------------------------------
            # One-hot-Encoding
            # -------------------------------------------------------------------------------------------------------------------
            #drop_enc = OneHotEncoder(drop='first').fit(data_train_disc)
            data_train_disc_1he = OneHotEncoder(drop='first').fit_transform(data_train_disc)
            data_test_disc_1he = OneHotEncoder(drop='first').fit_transform(data_test_disc)
        
            # from scipy import sparse
            # ma = sparse.csc_matrix.todense(data_test_disc_1he)
            return data_train_disc_1he, data_test_disc_1he, labels_train, labels_test
        else:
            return data_train_disc, data_test_disc, labels_train, labels_test
    else:
        return data_train, data_test, labels_train, labels_test
