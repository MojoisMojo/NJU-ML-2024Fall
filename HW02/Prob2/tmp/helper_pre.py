import numpy as np
from math import log

equalNums = lambda x, y: 0 if x is None else x[x == y].size


def singleEntropy(x):
    # 转换为 numpy 矩阵
    x = np.asarray(x)
    # 取所有不同值
    xValues = set(x)
    # 计算熵值
    entropy = 0
    for xValue in xValues:
        p = equalNums(x, xValue) / x.size
        entropy -= p * log(p, 2)
    return entropy


def conditionnalEntropy(feature, y):
    """计算 某特征feature 条件下y的信息熵"""
    # 转换为numpy
    feature = np.asarray(feature)
    y = np.asarray(y)
    # 取特征的不同值
    featureValues = set(feature)
    # 计算熵值
    entropy = 0
    for feat in featureValues:
        # 解释：feature == feat 是得到取feature中所有元素值等于feat的元素的索引（类似这样理解）
        #       y[feature == feat] 是取y中 feature元素值等于feat的元素索引的 y的元素的子集
        p = equalNums(feature, feat) / feature.size
        entropy += p * singleEntropy(y[feature == feat])
    return entropy


def infoGain(feature, y):
    return singleEntropy(y) - conditionnalEntropy(feature, y)


def infoGainRatio(feature, y):
    return (
        0
        if singleEntropy(feature) == 0
        else infoGain(feature, y) / singleEntropy(feature)
    )


def voteLabel(labels):
    uniqLabels = list(set(labels))
    labels = np.asarray(labels)
    finalLabel = 0
    labelNum = []
    for label in uniqLabels:
        # 统计每个标签值得数量
        labelNum.append(equalNums(labels, label))
    # 返回数量最大的标签
    return uniqLabels[labelNum.index(max(labelNum))]


def bestFeature(dataSet, labels, method="infoGain"):
    assert method in ["infoGain", "gainRatio"], "method 须为id3或gainRatio"
    dataSet = np.asarray(dataSet)
    labels = np.asarray(labels)

    # 根据输入的method选取 评估特征的方法：id3 -> 信息增益; gainRatio -> 信息增益率
    def calcEnt(feature, labels):
        if method == "infoGain":
            return infoGain(feature, labels)
        elif method == "gainRatio":
            return infoGainRatio(feature, labels)

    # 特征数量  即 data 的列数量
    featureNum = dataSet.shape[1]
    # 计算最佳特征
    bestEnt = 0
    bestFeat = -1
    for feature in range(featureNum):
        ent = calcEnt(dataSet[:, feature], labels)
        if ent >= bestEnt:
            bestEnt = ent
            bestFeat = feature
        # print("feature " + str(feature + 1) + " ent: " + str(ent)+ "\t bestEnt: " + str(bestEnt))
    return bestFeat, bestEnt


def splitFeatureData(data, labels, feature):
    """feature 为特征列的索引"""
    # 取特征列
    print(np.asarray(data).shape)
    print(feature)
    features = np.asarray(data)[:, feature]
    # 数据集中删除特征列
    data = np.delete(np.asarray(data), feature, axis=1)
    # 标签
    labels = np.asarray(labels)

    uniqFeatures = set(features)
    dataSet = {}
    labelSet = {}
    for feat in uniqFeatures:
        dataSet[feat] = data[features == feat]
        labelSet[feat] = labels[features == feat]
    return dataSet, labelSet


# 精确度与上面介绍的有所不同，为计算简单，分母改成了当前验证集样本树，而不是所有。
def createTreePrePruning(
    dataTrain, labelTrain, dataValid, labelValid, feat_name, method="infoGain"
):
    dataTrain = np.asarray(dataTrain)
    labelTrain = np.asarray(labelTrain)
    dataValid = np.asarray(dataValid)
    labelValid = np.asarray(labelValid)
    feat_name = np.asarray(feat_name)
    # 如果结果为单一结果
    if len(set(labelTrain)) == 1:
        return labelTrain[0]
        # 如果没有待分类特征
    elif dataTrain.size == 0:
        return voteLabel(labelTrain)
    # 其他情况则选取特征
    bestFeat, bestEnt = bestFeature(dataTrain, labelTrain, method=method)
    # 取特征名称
    bestFeatName = feat_name[bestFeat]
    # 从特征名称列表删除已取得特征名称
    feat_name = np.delete(feat_name, [bestFeat])
    # 根据最优特征进行分割
    dataTrainSet, labelTrainSet = splitFeatureData(dataTrain, labelTrain, bestFeat)
    # 预剪枝评估
    # 划分前的分类标签
    labelTrainLabelPre = voteLabel(labelTrain)
    labelTrainRatioPre = equalNums(labelTrain, labelTrainLabelPre) / labelTrain.size
    # 划分后的精度计算
    if dataValid is not None:
        dataValidSet, labelValidSet = splitFeatureData(dataValid, labelValid, bestFeat)
        # 划分前的验证标签正确比例
        labelValidRatioPre = equalNums(labelValid, labelTrainLabelPre) / labelValid.size
        # 划分后 每个特征值的分类标签正确的数量
        labelTrainEqNumPost = 0
        for val in labelTrainSet.keys():
            labelTrainEqNumPost += (
                equalNums(labelValidSet.get(val), voteLabel(labelTrainSet.get(val)))
                + 0.0
            )
        # 划分后 正确的比例
        labelValidRatioPost = labelTrainEqNumPost / labelValid.size
    # 如果没有评估数据 但划分前的精度等于最小值0.5 则继续划分，这一步不是很理解
    if dataValid is None and labelTrainRatioPre == 0.5:
        decisionTree = {bestFeatName: {}}
        for featValue in dataTrainSet.keys():
            decisionTree[bestFeatName][featValue] = createTreePrePruning(
                dataTrainSet.get(featValue),
                labelTrainSet.get(featValue),
                None,
                None,
                feat_name,
                method,
            )
    elif dataValid is None:
        return labelTrainLabelPre
    # 如果划分后的精度相比划分前的精度下降, 则直接作为叶子节点返回
    elif labelValidRatioPost < labelValidRatioPre:
        return labelTrainLabelPre
    else:
        # 根据选取的特征名称创建树节点
        decisionTree = {bestFeatName: {}}
        # 对最优特征的每个特征值所分的数据子集进行计算
        for featValue in dataTrainSet.keys():
            decisionTree[bestFeatName][featValue] = createTreePrePruning(
                dataTrainSet.get(featValue),
                labelTrainSet.get(featValue),
                dataValidSet.get(featValue),
                labelValidSet.get(featValue),
                feat_name,
                method,
            )
    return decisionTree

from sklearn.datasets import load_iris

if __name__ == "__main__":
    iris = load_iris()
    data = iris.data
    target = iris.target
    feat_name = iris.feature_names
    decisionTree = createTreePrePruning(data, target, None, None, feat_name)
    print(decisionTree)