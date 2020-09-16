from scipy.io import arff
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------------------------
# Load Txt
#   Missing values in categorical data will be replaced by '?'
#
# --------------------------------------------------------------------------------------------------------------
def read_txt(path, verbose):

    data = pd.read_csv(path, header=None, skiprows=2)

    with open(path) as f:
        for line in f:
            if line.find('Header') != -1:
                temp, header = line.split(':')
                header = header.strip('\n').split(',')
                header = [x.strip(' ') for x in header]
                numDims = header.__len__()
                if verbose > 1:
                    print("No. of Attributes = " + str(numDims) + "\nHeader = " + str(header))
            elif line.find('Name') != -1:
                temp, datasetName = line.split(':')
                datasetName = datasetName.replace(' ','').replace('\n','')
                if verbose > 1:
                    print("DataSet name = " + datasetName)
            else:
                pass

    cat_features = [i for i in range(0,len(header)) if header[i] == 'D']
    num_features = [i for i in range(0,len(header)) if header[i] == 'N']

    #if data.isnull().values.any() or data.isna().values().any():
    #    print('Found Null/Nan -- Be careful, GET RID OF THEM')
    
    meta = {}
    for i in cat_features:
        d = tuple([j for j in np.unique(data[i]) if j != '?'])
        meta[i] = d

    # Enumerate all features
    metaDict = {}
    for i in cat_features:

        name = i
        if verbose > 1:
            print('\t Processing (discrete) feature {}'.format(name))

        if len(np.where(data[name].isin(['?']))[0]) > 0:
            if verbose > 1:
                print('\t\tFound missing value')
            t = tuple([i for i in range(0, len(meta[name]) + 1)])
            d = meta[name] + ('?',)

            data[name].fillna('missing')

        else:

            t = tuple([i for i in range(0, len(meta[name]))])
            d = meta[name]

        metaDict[name] = dict(zip(d, t))

    # Enumerate the dataset
    for i in cat_features:
        name = i
        #print('Processing {}'.format(name))
        d = list(metaDict[name].keys())

        for j in d:
            data[name].values[data[name] == j] = metaDict[name][j]

    return data, metaDict

# --------------------------------------------------------------------------------------------------------------
# Load Arff
#   Missing values in categorical data will be replaced by '?'
#
# --------------------------------------------------------------------------------------------------------------
def read_arff(path, verbose):

    data, meta = arff.loadarff(path)

    mDict = meta.__dict__['_attributes']
    if verbose > 1:
        print('(arffreader.py) Loaded data successfully from path: ' + path)

    attNames = meta.names()
    attTypes = meta.types()

    n = len(mDict) - 1
    classIndex = len(mDict) - 1

    nc = len(meta[attNames[classIndex]][1])

    discRows = [i for i in range(0, len(attTypes)) if attTypes[i] == 'nominal' and i != classIndex]
    numRows = [i for i in range(0, len(attTypes)) if attTypes[i] == 'numeric' and i != classIndex]

    pdata = pd.DataFrame(data)

    # Enumerate all features
    metaDict = {}
    for i in discRows:

        name = attNames[i]
        if verbose > 1:
            print('\t Processing (discrete) feature {}'.format(name))

        pdata[name] = pdata[name].str.decode("utf-8")

        if len(np.where(pdata[name].isin(['?']))[0]) > 0:
            if verbose > 1:
                print('\t\tFound missing value')
            t = tuple([i for i in range(0, len(meta[name][1]) + 1)])
            d = meta[name][1] + ('?',)

            pdata[name].fillna('missing')

        else:

            t = tuple([i for i in range(0, len(meta[name][1]))])
            d = meta[name][1]

        metaDict[name] = dict(zip(d, t))
        
    # Enumerate the class
    name = attNames[classIndex]
    pdata[name] = pdata[name].str.decode("utf-8")

    t = tuple([i for i in range(0, len(meta[name][1]))])
    d = meta[name][1]
    metaDict[name] = dict(zip(d, t))

    # Enumerate the dataset
    for i in discRows:
        name = attNames[i]
        #print('Processing {}'.format(name))
        d = list(metaDict[name].keys())

        for j in d:
            pdata[name].values[pdata[name] == j] = metaDict[name][j]
            
    name = attNames[classIndex]
    d = list(metaDict[name].keys())

    for j in d:
        pdata[name].values[pdata[name] == j] = metaDict[name][j]

    return pdata, metaDict

    #feats_sel = [i for i in pdata.columns.values if i != 'class']
    #data = pdata[feats_sel]
    #labels = pdata['class']

    #replace_missing_flag = True

    #if replace_missing_flag:
    #    min_per_feature = data.min()
    #    features_with_null = np.where(min_per_feature.isnull())[0]
    #    if len(features_with_null) == 0:
    #        data = data.fillna(data.min() - 1).reset_index(drop=True)
    #    else:
    #        print('Critical error: all values of following features is NaN')
    #        nan_features = data.columns.values[features_with_null]
    #        print(nan_features)
    #        data = data.drop(nan_features, axis=1)

    #       data = data.fillna(data.min() - 1).reset_index(drop=True)

    #if '.txt' in path:

    #    disc_dic = {}
    #    for i in discrete_att:
    #        temp_dic = {}
    #        counter = 0
    #        for j in np.unique(data[i]):
    #            temp_dic.update({j:counter})
    #            counter = counter + 1

    #        #disc_dic.append(temp_dic)
    #        disc_dic[i] = temp_dic

    #    for key in disc_dic:

    #        data['Disc_' + str(key)] = [disc_dic[key][data[key][j]] for j in range(0, len(data))]

    #    data = data.drop(columns = discrete_att)

    #    for key in disc_dic:
    #        data = data.rename(columns = {'Disc_' + str(key):key})

    #    cols = np.sort(data.columns.values)
    #    data = data[cols]


