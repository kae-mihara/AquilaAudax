import numpy as np
from Misc import globals
from scipy import sparse
import pandas as pd

# from models.probDist import xyDist, xxyDist

def sortListBasedonProvidedIndex(list2, list1):

    zipped_lists = zip(list2, list1)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list1 = [element for _, element in sorted_zipped_lists]

    return sorted_list1

def ind(i, j):
    if i == j:
        return 1
    else:
        return 0


def MEsti(freq1, freq2, numValues):
    m = globals.m_MEsti
    return (freq1 + m / numValues) / (freq2 + m)


def getMutualInformation(xyDist_, params_data):
    N = params_data['NumData']
    n = params_data['NumAttributes']
    c = params_data['NumClasses']
    paramsperAtt = params_data['paramsPerAtt']

    mi = np.zeros(n)

    for u in range(0, n):
        m = 0
        for uval in range(0, paramsperAtt[u]):
            for y in range(0, c):
                avyCount = xyDist_.getCount(u, uval, y)
                if avyCount > 0:
                    m += (avyCount / N) * np.log(avyCount / (xyDist_.getCount_Marg(u, uval) / N * xyDist_.getClassCount(y))) / np.log(2)
                    #print(avyCount, xyDist_.getCount_Marg(u, uval), xyDist_.getClassCount(y), m)
        mi[u] = m

    return mi


def getCondMutualInf(xxyDist_, params_data):
    N = params_data['NumData']
    n = params_data['NumAttributes']
    c = params_data['NumClasses']
    paramsperAtt = params_data['paramsPerAtt']

    cmi = np.zeros((n, n))

    for u1 in range(1, n):
        for u2 in range(0, u1):

            mi = 0

            for u1val in range(0, paramsperAtt[u1]):
                for u2val in range(0, paramsperAtt[u2]):
                    for y in range(0, c):

                        avvyCount = xxyDist_.getCount(u1, u1val, u2, u2val, y)

                        if avvyCount > 0:
                            a = avvyCount
                            b = xxyDist_.xyDist_.getClassCount(y)
                            d = xxyDist_.xyDist_.getCount(u1, u1val, y)
                            e = xxyDist_.xyDist_.getCount(u2, u2val, y)

                            mitemp = (a / N) * np.log((a * b) / (d * e)) / np.log(2)
                            mi += mitemp

            cmi[u1][u2] = mi
            cmi[u2][u1] = mi

    return cmi


def pd_multiply_1_cols(pd1):
    df = pd.get_dummies(pd1, prefix=pd1.name)

    df = sparse.csr_matrix(df.values)
    return df

def pd_multiply_2_cols(pd1, pd2):
    df = pd.DataFrame()

    pd1_ohe = pd.get_dummies(pd1, prefix=pd1.name)
    pd2_ohe = pd.get_dummies(pd2, prefix=pd2.name)

    for u1 in range(0, len(pd1_ohe.columns.values)):

        u1_name = pd1_ohe.columns.values[u1]

        for u2 in range(0, len(pd2_ohe.columns.values)):
            u2_name = pd2_ohe.columns.values[u2]

            # df['{}-{}_{}.{}'.format(u1_name, u2_name, u1, u2)] = pd1_ohe[u1_name] * pd2_ohe[u2_name]
            df['{}x{}'.format(u1_name, u2_name)] = pd1_ohe[u1_name] * pd2_ohe[u2_name]

    df = sparse.csr_matrix(df.values)
    return df


def pd_multiply_3_cols(pd1, pd2, pd3):
    df = pd.DataFrame()

    pd1_ohe = pd.get_dummies(pd1, prefix=pd1.name)
    pd2_ohe = pd.get_dummies(pd2, prefix=pd2.name)
    pd3_ohe = pd.get_dummies(pd3, prefix=pd3.name)

    for u1 in range(0, len(pd1_ohe.columns.values)):

        u1_name = pd1_ohe.columns.values[u1]

        for u2 in range(0, len(pd2_ohe.columns.values)):

            u2_name = pd2_ohe.columns.values[u2]

            for u3 in range(0, len(pd3_ohe.columns.values)):
                u3_name = pd3_ohe.columns.values[u3]

                df['{}x{}x{}'.format(u1_name, u2_name, u3_name)] = pd1_ohe[u1_name] * pd2_ohe[u2_name] * pd3_ohe[
                    u3_name]

    df = sparse.csr_matrix(df.values)
    return df


def pd_multiply_4_cols(pd1, pd2, pd3, pd4):
    df = pd.DataFrame()

    pd1_ohe = pd.get_dummies(pd1, prefix=pd1.name)
    pd2_ohe = pd.get_dummies(pd2, prefix=pd2.name)
    pd3_ohe = pd.get_dummies(pd3, prefix=pd3.name)
    pd4_ohe = pd.get_dummies(pd4, prefix=pd3.name)

    for u1 in range(0, len(pd1_ohe.columns.values)):

        u1_name = pd1_ohe.columns.values[u1]

        for u2 in range(0, len(pd2_ohe.columns.values)):

            u2_name = pd2_ohe.columns.values[u2]

            for u3 in range(0, len(pd3_ohe.columns.values)):

                u3_name = pd3_ohe.columns.values[u3]

                for u4 in range(0, len(pd4_ohe.columns.values)):
                    u4_name = pd4_ohe.columns.values[u4]

                    df['{}x{}x{}x{}'.format(u1_name, u2_name, u3_name, u4_name)] = pd1_ohe[u1_name] * pd2_ohe[u2_name] * \
                                                                                   pd3_ohe[u3_name] * pd4_ohe[u4_name]

    df = sparse.csr_matrix(df.values)
    return df


def pd_multiply_5_cols(pd1, pd2, pd3, pd4, pd5):
    df = pd.DataFrame()
    
    pd1_ohe = pd.get_dummies(pd1, prefix=pd1.name)
    pd2_ohe = pd.get_dummies(pd2, prefix=pd2.name)
    pd3_ohe = pd.get_dummies(pd3, prefix=pd3.name)
    pd4_ohe = pd.get_dummies(pd4, prefix=pd3.name)
    pd5_ohe = pd.get_dummies(pd5, prefix=pd3.name)

    for u1 in range(0, len(pd1_ohe.columns.values)):

        u1_name = pd1_ohe.columns.values[u1]

        for u2 in range(0, len(pd2_ohe.columns.values)):

            u2_name = pd2_ohe.columns.values[u2]

            for u3 in range(0, len(pd3_ohe.columns.values)):

                u3_name = pd3_ohe.columns.values[u3]

                for u4 in range(0, len(pd4_ohe.columns.values)):

                    u4_name = pd4_ohe.columns.values[u4]

                    for u5 in range(0, len(pd5_ohe.columns.values)):
                        u5_name = pd5_ohe.columns.values[u5]

                        df['{}x{}x{}x{}x{}'.format(u1_name, u2_name, u3_name, u4_name, u5_name)] = pd1_ohe[u1_name] * \
                                                                                                   pd2_ohe[u2_name] * \
                                                                                                   pd3_ohe[u3_name] * \
                                                                                                   pd4_ohe[u4_name] * \
                                                                                                   pd5_ohe[u5_name]

    df = sparse.csr_matrix(df.values)
    return df