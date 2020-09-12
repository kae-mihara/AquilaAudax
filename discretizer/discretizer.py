import numpy as np
import pandas as pd

import math

#from AA.aautils import entropy, table_entropy

# --------------------------------------------------------------------------------------------------------------
#   Return cutpoints using EF on data
# --------------------------------------------------------------------------------------------------------------

def return_cutpoints(data, **params):

    m_NumBins = params.get('num_bins', None)
    discretization_type = params.get('method', None)
    feats_sel = params.get('feats_sel', None)
    labels = params.get('labels', None)
    dict = params.get('dict', None)
    treat_missing_distinct = params.get('treat_missing_distinct', None)
    verbosity = params.get('verbosity', None)

    cutPoints = {}
    for feat in feats_sel:
        if verbosity == True:
            print('(EF) Discretizing attribute {}'.format(feat))
        cutpoints = discretize_EF(data[feat].astype(dtype=np.float64), m_NumBins)
        cutPoints[feat] = cutpoints

    return cutPoints

# --------------------------------------------------------------------------------------------------------------
#   Apply Discretizer generic
# --------------------------------------------------------------------------------------------------------------

def applyDiscretizer(data, cutPoints, **params):

    m_NumBins = params.get('num_bins', None)
    discretization_type = params.get('method', None)
    feats_sel = params.get('feats_sel', None)

    if discretization_type == 'EF':
        dX_train = applyDiscretizer_EF_fast(data, cutPoints, m_NumBins, feats_sel)
    elif discretization_type == 'EF_NP':
        dX_train = applyDiscretizer_EF_fast(data, cutPoints, m_NumBins, feats_sel)
    elif discretization_type == 'MDL':
        dX_train = applyDiscretizer_MDL(data, cutPoints, feats_sel)

    return dX_train

# --------------------------------------------------------------------------------------------------------------
#   Main function to be called by any method requiring discretization
# --------------------------------------------------------------------------------------------------------------

def discretize(data_train, data_test, **params):

    m_NumBins = params.get('num_bins', None)
    discretization_type = params.get('method', None)
    feats_sel = params.get('feats_sel', None)
    labels = params.get('labels', None)
    dict = params.get('dict', None)
    treat_missing_distinct = params.get('treat_missing_distinct', None)
    verbosity = params.get('verbosity', None)

    if discretization_type == 'EF':

        # ------------------------------------------------------------------------------------------------------
        # our own weka-inspired Equal Frequency Discretization
        # ------------------------------------------------------------------------------------------------------

        if data_train.isnull().values.any() or data_test.isnull().values.any() or data_train.isna().values.any() \
                or data_test.isna().values.any():
            print('(discretize.py) Null or NaN values detected in either train or test.')
            print('(discretize.py) ---- NaN or Nulls will be assigned to the last bin ----')

        cutPoints = {}
        for feat in feats_sel:
            if verbosity == True:
                print('(EF) Discretizing attribute {}'.format(feat))
            cutpoints = discretize_EF(data_train[feat].astype(dtype=np.float64), m_NumBins)
            cutPoints[feat] = cutpoints

        dX_train = applyDiscretizer_EF_fast(data_train, cutPoints, m_NumBins, feats_sel)
        dX_test = applyDiscretizer_EF_fast(data_test, cutPoints, m_NumBins, feats_sel)

        strategy_I = False

        if strategy_I:

            for feat in feats_sel:
                unique_vals = np.unique(dX_train[feat])
                temp = {}
                j = 0
                for i in unique_vals:
                    temp[str(i)] = j
                    j = j + 1
                dict[feat] = temp

        else:

            dX_train, dX_test, dict = update_dictionary(data_train, data_test, feats_sel, dX_train, dX_test,
                                                        treat_missing_distinct, dict)

        return dX_train, dX_test, dict, cutPoints

    elif discretization_type == 'EF_NP':

        # ------------------------------------------------------------------------------------------------------
        # Equal Frequency Discretization using numpy and pandas
        # ------------------------------------------------------------------------------------------------------

        cutPoints = {}
        for feat in feats_sel:
            cutpoints = discretize_EF_NP(data_train[feat].astype(dtype=np.float64), m_NumBins)
            cutPoints[feat] = cutpoints

        dX_train = applyDiscretizer_EF_fast(data_train, cutPoints, m_NumBins, feats_sel)
        dX_test = applyDiscretizer_EF_fast(data_test, cutPoints, m_NumBins, feats_sel)

        return dX_train, dX_test, cutpoints

    elif discretization_type == 'MDL':

        # ------------------------------------------------------------------------------------------------------
        # MDL Discretization
        # ------------------------------------------------------------------------------------------------------

        if data_train.isnull().values.any() or data_test.isnull().values.any() or data_train.isna().values.any() \
                or data_test.isna().values.any():
            print('(discretize.py) Null or NaN values detected in either train or test.')
            print('NaN handling strategy: Will create a new value')

        verbosity_flag = verbosity

        cutPoints = {}
        i = 0
        for feat in feats_sel:
            if verbosity == True:
                print('(MDL) Discretizing attribute {}'.format(feat))
            cutpoints = discretize_MDL_flat(data_train[feat], labels, feat, verbosity_flag)
            print(cutpoints)
            cutPoints[feat] = cutpoints
            i = i + 1

        dX_train = applyDiscretizer_MDL(data_train, cutPoints, feats_sel)
        dX_test = applyDiscretizer_MDL(data_test, cutPoints, feats_sel)

        dX_train, dX_test, dict = update_dictionary(data_train, data_test, feats_sel, dX_train, dX_test,
                                                    treat_missing_distinct, dict)

        return dX_train, dX_test, dict, cutPoints


def update_dictionary(data_train, data_test, feats_sel, dX_train, dX_test, treat_missing_distinct, dict):

    for feat in feats_sel:

        if treat_missing_distinct:

            data_train_nan_indices = data_train[data_train[feat].isna()].index.values
            data_test_nan_indices = data_test[data_test[feat].isna()].index.values

            if data_train_nan_indices is not None or data_test_nan_indices is not None:

                unique_vals = np.unique(dX_train[feat])
                temp = {}
                j = 0
                for i in unique_vals:
                    temp[str(i)] = j
                    j = j + 1
                temp['?'] = j

                dict[feat] = temp

            if data_train_nan_indices is not None:
                dX_train[feat].loc[data_train_nan_indices] = j

            if data_test_nan_indices is not None:
                dX_test[feat].loc[data_test_nan_indices] = j

            else:

                unique_vals = np.unique(dX_train[feat])
                temp = {}
                j = 0
                for i in unique_vals:
                    temp[str(i)] = j
                    j = j + 1
                dict[feat] = temp

        else:

            unique_vals = np.unique(dX_train[feat])
            temp = {}
            j = 0
            for i in unique_vals:
                temp[str(i)] = j
                j = j + 1
            dict[feat] = temp

    return dX_train, dX_test, dict

def applyDiscretizer_MDL(data, cutPoints, feats_sel):

    discData = data.copy()
    for feat in feats_sel:
        if cutPoints[feat] is None:
            discData[feat] = -1
        else:
            discData[feat] = np.digitize(data[feat].values, cutPoints[feat].ravel(), right=True)

    return discData


# --------------------------------------------------------------------------------------------------------------
#   EF (Weka)
# --------------------------------------------------------------------------------------------------------------

def discretize_EF(fval, m_NumBins):

    IGNORE_NAN_Flag = False

    if IGNORE_NAN_Flag:

        fval = list(np.sort(fval))

    else:

        fval = np.sort(list(np.array(fval)[np.where(~np.isnan(fval))[0]]))

    sumOfWeights = len(fval)
    freq = sumOfWeights / m_NumBins
    cutPoints = np.inf * np.ones(m_NumBins - 1)

    counter = 0
    last = 0
    cpindex = 0
    lastIndex = -1

    for i in range(0, len(fval) - 1):
        counter = counter + 1
        sumOfWeights = sumOfWeights - 1

        # Do we have a potential cutpoint
        if (fval[i] < fval[i + 1]):
            # Have we passed an ideal size
            if (counter >= freq):

                # Is this break point worst than the last one
                if ((freq - last) < (counter - freq) and (lastIndex != -1)):
                    cutPoints[cpindex] = (fval[lastIndex] + fval[lastIndex + 1]) / 2
                    counter = counter - last
                    last = counter
                    lastIndex = i
                else:
                    cutPoints[cpindex] = (fval[i] + fval[i + 1]) / 2
                    counter = 0
                    last = 0
                    lastIndex = -1

                cpindex = cpindex + 1
                freq = (sumOfWeights + counter) / ((m_NumBins - cpindex))
            else:
                lastIndex = i
                last = counter

    if ((cpindex < len(cutPoints)) & (lastIndex != -1)):
        cutPoints[cpindex] = (fval[lastIndex] + fval[lastIndex + 1]) / 2
        cpindex = cpindex + 1

    return cutPoints


def applyDiscretizer_EF_fast(data, cutPoints, m_NumBins, feats_sel):

    discData = data.copy()

    for feat in feats_sel:
        if cutPoints[feat] is None:
            discData[feat] = -1
        else:
            discData[feat] = np.digitize(data[feat].values, cutPoints[feat].ravel(), right=True)

    return discData
    
def applyDiscretizer_EF(data, cutPoints, headerNoClass, m_NumBins):

    [numData, numDims] = data.shape
    
    discData = -1 * np.ones((numData, numDims))

    for i in range(numData):
        
        for d in range(0, numDims):
            cutpoints = cutPoints[d]
            val = data[headerNoClass[d]][i]
            
            if pd.isnull(val):
                discData[i][d] = m_NumBins + 1
            else:
                val_broken = False
                for j in range(0, len(cutpoints)):
                    if val <= cutpoints[j]:
                        val_broken = True
                        break

                if val_broken:
                    discData[i][d] = j
                else:
                    discData[i][d] = j + 1

    return discData

# --------------------------------------------------------------------------------------------------------------
#   EF (NP)
# --------------------------------------------------------------------------------------------------------------

def discretize_EF_NP(fval, m_NumBins):

    width = 100 / m_NumBins

    cutPoints = np.inf * np.ones(m_NumBins - 1)
    # print('width is {}'.format(width))

    end = 100
    index = 0
    for i in np.arange(width, end, width):
        cutPoints[index] = np.percentile(fval, i)
        index = index + 1


def applyDiscretizer_Ef_NP_fast(data, cutPoints, m_NumBins, feats_sel):
    cp = np.zeros([len(cutPoints), m_NumBins - 1])
    for i in range(0, len(cutPoints)):
        cp[i, :] = cutPoints[i].ravel()

    [numData, numDims] = data.shape
    discData = -1 * np.ones((numData, numDims))

    i = 0
    for feat in feats_sel:
        discData[:, i] = np.digitize(data[feat].values, cp[i, :], right=False)
        i = i + 1

    return discData


def applyDiscretizer_EF_NP(data, cutPoints, headerNoClass, m_NumBins):
    [numData, numDims] = data.shape

    discData = -1 * np.ones((numData, numDims))

    for i in range(numData):

        for d in range(0, numDims):
            cutpoints = cutPoints[d]
            val = data[headerNoClass[d]][i]

            if pd.isnull(val):
                discData[i][d] = m_NumBins + 1
            else:
                val_broken = False
                for j in range(0, len(cutpoints)):
                    if val <= cutpoints[j]:
                        val_broken = True
                        break

                if val_broken:
                    discData[i][d] = j
                else:
                    discData[i][d] = j + 1

    return discData

# --------------------------------------------------------------------------------------------------------------
#   MDL (Weka)
# --------------------------------------------------------------------------------------------------------------

class Node:

    def __init__(self, level, nid, XY, left_boundary, right_boundary, feature):

        self.left = None
        self.right = None
        self.level = level
        self.nid = nid

        self.XY = XY

        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        self.feature = feature

        self.pivot = -1

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def getLevel(self):
        return self.level

    def getNid(self):
        return self.nid

    def train(self):

        print('Left/Right Boundaries: {}/{}'.format(self.left_boundary, self.right_boundary))

        XY_mod = self.XY[int(self.left_boundary):int(self.right_boundary)].reset_index(drop=False)
        numInstances = len(XY_mod)

        # Compute fci -- Feature Change Index
        fci = np.where(np.abs(XY_mod[self.feature].diff().fillna(0)))[0]

        numCutPoints = (self.right_boundary - self.left_boundary) - 1

        X = XY_mod[self.feature]
        priorEntropy = entropy(list(XY_mod['class'].values))

        bestEntropy = priorEntropy

        for j in fci:

            pivot = (XY_mod[self.feature][j] + XY_mod[self.feature][j-1])/2

            X1 = XY_mod[XY_mod[self.feature] < pivot]
            X2 = XY_mod[XY_mod[self.feature] >= pivot]

            entropyX1 = entropy(list(X1['class'].values))
            entropyX2 = entropy(list(X2['class'].values))

            current_entropy = (len(X1) / len(X) * entropyX1 + len(X2) / len(X) * entropyX2)

            if current_entropy < bestEntropy:
                self.pivot = pivot
                bestEntropy = current_entropy
                numClassesLeft = len(np.unique(X1['class']))
                numClassesRight = len(np.unique(X2['class']))
                best_entropyleft = entropyX1
                best_entropyright = entropyX2
                best_index = XY_mod.loc[j]['index'] - 1

        print('Best pivot selected = {}'.format(self.pivot))

        gain =  priorEntropy - bestEntropy
        if gain <= 0:
            return

        numClassesTotal = len(np.unique(XY_mod['class']))

        delta = math.log2(math.pow(3, numClassesTotal) - 2) - ((numClassesTotal * priorEntropy) - (numClassesRight * best_entropyright) - (numClassesLeft * best_entropyleft))

        if gain > (math.log2(numCutPoints) + delta) / numInstances :

            nodeL = Node(self.level + 1, self.nid + "_" + str(self.level + 1) + "a_", self.XY, self.left_boundary, best_index + 1, self.feature)
            self.left = nodeL
            nodeL.train()

            nodeR = Node(self.level + 1, self.nid + "_" + str(self.level + 1) + "b_", self.XY, best_index + 1, self.right_boundary, self.feature)
            self.right = nodeR
            nodeR.train()

        else:

            return

def discretize_MDL_tree(fval, labels, feature):

    feat = fval.reset_index(drop=True)
    label = labels.reset_index(drop=True)

    XY = pd.concat([feat, label], axis=1)
    XY = XY.sort_values(by=[feature]).reset_index(drop=True)

    left_boundary = np.argmin(XY[feature])
    right_boundary = np.argmax(XY[feature]) + 1

    forest = Node(0, "0", XY, left_boundary, right_boundary, feature)
    forest.train()

    return 0

def discretize_MDL_flat(fval, labels, feature, verbosity_flag):

    feat = fval.reset_index(drop=True)
    label = labels.reset_index(drop=True)

    XY = pd.concat([feat, label], axis=1)
    data = XY.sort_values(by=[feature]).reset_index(drop=True)

    last_occurence = np.where(data[feature] == data[feature].max())[0]
    firstMissing = last_occurence[len(last_occurence) - 1] + 1

    m_CutPoints = cutPointsForSubset(data, 0, firstMissing, feature, verbosity_flag)

    return m_CutPoints

def cutPointsForSubset(data, left_boundary, right_boundary, feature, verbosity_flag):

    if (right_boundary  - left_boundary) < 2:
        return

    data = data.loc[int(left_boundary):int(right_boundary) - 1]

    numInstances = len(data)

    # Compute fci -- Feature Change Index
    vals = np.abs(data[feature].diff().fillna(0))
    fci = vals[vals != 0].index.values.astype(int)

    numCutPoints = (right_boundary - left_boundary) - 1

    X = data[feature]
    priorEntropy = entropy(list(data['class'].values))

    bestEntropy = priorEntropy

    for j in fci:

        pivot = (data[feature][j] + data[feature][j - 1]) / 2

        X1 = data[data[feature] < pivot]
        X2 = data[data[feature] >= pivot]

        entropyX1 = entropy(list(X1['class'].values))
        entropyX2 = entropy(list(X2['class'].values))

        current_entropy = (len(X1) / len(X) * entropyX1 + len(X2) / len(X) * entropyX2)

        if current_entropy < bestEntropy:
            bestCutPoint = pivot
            bestEntropy = current_entropy
            numClassesLeft = len(np.unique(X1['class']))
            numClassesRight = len(np.unique(X2['class']))
            best_entropyleft = entropyX1
            best_entropyright = entropyX2
            best_index = j - 1

    gain = priorEntropy - bestEntropy
    if gain <= 0:
        return

    numClassesTotal = len(np.unique(data['class']))

    delta = math.log2(math.pow(3, numClassesTotal) - 2) - (
                (numClassesTotal * priorEntropy) - (numClassesRight * best_entropyright) - (
                    numClassesLeft * best_entropyleft))

    if verbosity_flag:

        print('Left/Right Boundaries: {}/{} -- bestCutPoint = {}, best Entropy = {}, best_index = {}, delta = {}'.format(
            left_boundary, right_boundary, bestCutPoint, bestEntropy, best_index, delta))

    if gain > (math.log2(numCutPoints) + delta) / numInstances:

        left = cutPointsForSubset(data, left_boundary, best_index + 1, feature, verbosity_flag)
        right = cutPointsForSubset(data, best_index + 1, right_boundary, feature, verbosity_flag)

        if left is None and right is None:
            cutPoints =  np.array([bestCutPoint])
        elif right is None:
            cutPoints = np.zeros([len(left) + 1])
            cutPoints[0:len(left)] = left[0:len(left)]
            cutPoints[len(left)] = bestCutPoint
        elif left is None:
            cutPoints = np.zeros([1 + len(right)])
            cutPoints[0] = bestCutPoint
            cutPoints[1:len(right)+1] = right[0:len(right)]
        else:
            cutPoints = np.zeros(len(left) + len(right) + 1)
            cutPoints[0:len(left)] = left[0:len(left)]

            cutPoints[len(left)] = bestCutPoint
            cutPoints[len(left)+1:] = right[0:len(right)]

        return cutPoints

    else:

        return

