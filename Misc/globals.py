import numpy as np

# data_path = "../../Data/nayyar/"
# m_MEsti = 0.1

def getDataMeta(data, labels):

    paramsPerAtt = [len(np.unique(data[i])) for i in data.columns.values]
    NumData, NumAttributes = data.shape
    NumClasses = len(np.unique(labels))
    ColNames = data.columns.values

    params_data = {'paramsPerAtt': paramsPerAtt, 'NumData': NumData, 'NumAttributes': NumAttributes,
                   'NumClasses': NumClasses, 'ColNames': ColNames}

    return params_data

def printWelcomeMessage():

    print('-------------------------------------------------------------------------')
    print('Welcome to Aquila Audax')
    print('Version: v0.3')

    print('Library for learning from extremely large quantities of data in minimal')
    print('number of passes through the data.')
    print('Salient features:')
    print('     1) Superior Feature Engineering Capability')
    print('     2) Fast Optimization')
    print('     3) Explainable Predictions ')
    print('     3) Confidence Predictions ')

    print('Type -help for information how to use the library')

    print('Copyrights Deakin Unviversity ')
    print('------------------------------------------------------------------------- ')