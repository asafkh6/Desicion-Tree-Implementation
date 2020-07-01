import random
import numpy as np
import pandas as pd
from scipy.stats import chi2
import math
import copy
import sys


def createData():
    file = open("OnlineNewsPopularity.data.txt")
    data1 = file.readlines()
    file.close()
    newData = []
    for example in data1:
        e = []
        a = ""  # a for attribute
        for x in example:
            if x != ',':
                a = a + x
            else:
                num = float(a)
                e.append(num)
                a = ""
        num = float(a)
        e.append(num)
        newData.append(e)

    return newData


def attributeReset():
    attributes_id = []
    for i in range(58):
        attributes_id.append(i)
    return attributes_id


def calcError(treeN, examples):  # returns the ratio of errors from group of examples
    numOfErrors = 0
    for example in examples:
        if isThisViral(example, treeN) != example[58]:
            numOfErrors = numOfErrors + 1
    return numOfErrors / len(examples)


def startBuildTree(trainSet, names):
    root = getNode(trainSet, names)
    split(root, trainSet)
    root = pruneVertices(root)
    return root


def split(node, parentExamples):  # the recursive function that decides how tree will be split
    childRight, childLeft = node['groups']
    spiltNode(node, node['attributesLeft'], parentExamples, childLeft, True)  # check if to split or prune
    spiltNode(node, node['attributesLeft'], parentExamples, childRight, False)  # check if to split or prune


def spiltNode(node, attributes, parentExamples, examples=None, isLeft=True):
    if len(examples) == 0:  # if examples are empty
        if isLeft:
            node['leftChild'] = pluralityValue(parentExamples)
        else:
            node['rightChild'] = pluralityValue(parentExamples)
        return
    plus, minus = getViralOrNot(examples)
    ratio = plus / len(examples)
    if ratio == 0 or ratio == 1:  # if all examples have the same classification
        if isLeft:
            node['leftChild'] = ratio
        else:
            node['rightChild'] = ratio
        return
    if len(attributes) == 0:  # if Attributes is empty
        if isLeft:
            node['leftChild'] = pluralityValue(examples)
        else:
            node['rightChild'] = pluralityValue(examples)
        return

    newNode = getNode(examples, attributes)
    if isLeft:
        node['leftChild'] = newNode
    else:
        node['rightChild'] = newNode

    split(newNode, examples)


def getNode(examples, attNames):
    threshold = 0
    attribute = "a"
    minEntropy = 1
    newAtt = copy.deepcopy(attNames)
    for att in newAtt:
        threshD = getThreshold(att, examples)
        if calcEntropy(att, examples, threshD) < minEntropy:
            attribute = att
            threshold = threshD
            minEntropy = calcEntropy(att, examples, threshD)
    if minEntropy < 1:
        newAtt.remove(attribute)
    if getDivision(examples, attribute, threshold) is None:
        groups = [pluralityValue(examples), pluralityValue(examples)]
    else:
        groups = getDivision(examples, attribute, threshold)
    return {'attribute': attribute, 'threshold': threshold, 'groups': groups,
            'attributesLeft': newAtt}
    # return n


def calcEntropy(attribute, examples, threshold):
    minus, plus = getDivision(examples, attribute, threshold)
    return len(plus) / len(examples) * entropy(plus) + len(minus) / len(examples) * entropy(minus)


def entropy(examples):
    if len(examples) == 0:
        return 0
    plus, minus = getViralOrNot(examples)
    plus = plus / len(examples)
    minus = minus / len(examples)
    if plus == 0 or minus == 0:
        return 0
    return - plus * math.log2(plus) - minus * math.log2(minus)


def getThreshold(attribute, examples):
    sumAll = 0
    for row in range(len(examples)):
        sumAll = sumAll + examples[row][attribute]
    return sumAll / len(examples)


def pluralityValue(examples):
    plus, minus = getViralOrNot(examples)
    if plus >= minus:
        return 1
    else:
        return 0


def pruneVertices(node):
    kStt = 0
    if (node['leftChild'] == 0 or node['leftChild'] == 1) and (
            node['rightChild'] == 0 or node['rightChild'] == 1):  # if both leafs
        largerE = node['groups'][0]
        smallerE = node['groups'][1]
        largerViral, largerNotViral = getViralOrNot(largerE)  # larger viral, larger not viral
        smallerViral, smallerNotViral = getViralOrNot(smallerE)  # smaller viral, smaller not viral
        pLarger = (largerViral + largerNotViral) / (len(largerE) + len(smallerE))  # probability larger
        pSmaller = (smallerViral + smallerNotViral) / (len(largerE) + len(smallerE))  # probability smaller
        plv = pLarger * (largerViral + smallerViral)
        psv = pSmaller * (largerViral + smallerViral)
        plnv = pLarger * (largerNotViral + smallerNotViral)
        psnv = pSmaller * (largerNotViral + smallerNotViral)
        if plv != 0:
            kStt = kStt + ((plv - largerViral) ** 2) / plv
        if plnv != 0:
            kStt = kStt + ((plnv - largerNotViral) ** 2) / plnv
        if psv != 0:
            kStt = kStt + ((psv - smallerViral) ** 2) / psv
        if psnv != 0:
            kStt = kStt + ((psnv - smallerNotViral) ** 2) / psnv
        kCrt = chi2.ppf(0.95, len(largerE) + len(smallerE) - 1)
        if kStt < kCrt:
            if largerViral + smallerViral > largerNotViral + smallerNotViral:
                return 1
            else:
                return 0
        else:
            return node
    # recursive calling until we reach a leaf
    elif node['rightChild'] == 0 or node['rightChild'] == 1:
        node['leftChild'] = pruneVertices(node['leftChild'])
    elif node['leftChild'] == 0 or node['leftChild'] == 1:
        node['rightChild'] = pruneVertices(node['rightChild'])
    else:
        node['leftChild'] = pruneVertices(node['leftChild'])
        node['rightChild'] = pruneVertices(node['rightChild'])
    return node


def getViralOrNot(examples):
    yes = 0
    no = 0
    for b in range(len(examples)):
        if examples[b][58] == 1:
            yes = yes + 1
        else:
            no = no + 1
    return yes, no


def countDivision(examples, attribute, threshold):
    bigger = 0
    smaller = 0
    for x in range(len(examples)):
        if examples[x][attribute] <= threshold:
            smaller = smaller + 1
        else:
            bigger = bigger + 1

    return smaller, bigger


def getDivision(examples, attribute, threshold):
    bigger = []
    smaller = []
    if attribute == "a":
        return
    for i in range(len(examples)):
        if examples[i][attribute] <= threshold:
            smaller.append(examples[i])
        else:
            bigger.append(examples[i])
    return smaller, bigger


def arrangeData(originData): # set the y
    for p in range(len(originData)):
        if originData[p][58] >= 2000:
            originData[p][58] = 1
        else:
            originData[p][58] = 0
    return originData


def divideData(k, thisData, isTrainingData):
    thisSplit = round(k * len(thisData))
    if isTrainingData:
        newData = thisData[:thisSplit]
    else:
        newData = thisData[thisSplit:]
    return newData


def divideDataCrossValidation(k, crossData):
    random.shuffle(crossData)
    kGroups = [[] for i in range(k)]
    groupSize = int(len(crossData) / k)
    counter = 0
    for i in range(k):
        kGroups = kGroups + []
        for j in range(groupSize):
            kGroups[i].append(crossData[counter])
            counter = counter + 1
    return kGroups


def deleteGroups(node):  # deleting groups of Examples for printing
    if 'attributesLeft' in node:
        del (node['attributesLeft'])
    if isinstance(node['leftChild'], dict):
        if 'groups' in node:
            del (node['groups'])
        deleteGroups(node['leftChild'])
    else:
        if 'groups' in node:
            del (node['groups'])
    if isinstance(node['rightChild'], dict):
        if 'groups' in node:
            del (node['groups'])
        deleteGroups(node['rightChild'])
    else:
        if 'groups' in node:
            del (node['groups'])
    return


def buildTree(data2, k, attributes_list):  # base function for building tree
    random.shuffle(data2)
    trainSet = divideData(k, data2, True)
    rootNode = startBuildTree(trainSet, attributes_list)
    validationSet = divideData(k, data2, False)
    deleteGroups(rootNode)
    print("Tree:")
    print(rootNode)
    print("Tree Error:", calcError(rootNode, validationSet))
    return rootNode


def treeError(data, k):
    kGroups = divideDataCrossValidation(k, data)
    totalError = 0
    for i in range(k):
        testSet = kGroups[i]
        trainSet = []
        for j in range(k):
            if j != i:
                for p in range(len(kGroups[j])):
                    trainSet = trainSet + [kGroups[j][p]]
        kTree = startBuildTree(trainSet, attributeReset())
        totalError = totalError + calcError(kTree, testSet)
    print("Cross Validation error: ", (totalError / k))


def isThisViral(example, tree1):  # predicting: 1 if viral zero if not
    if example[tree1['attribute']] > tree1['threshold']:
        if isinstance(tree1['leftChild'], dict):
            return isThisViral(example, tree1['leftChild'])
        else:
            return tree1['leftChild']
    else:
        if isinstance(tree1['rightChild'], dict):
            return isThisViral(example, tree1['rightChild'])
        else:
            return tree1['rightChild']


df = createData()
df = arrangeData(df)

buildTree(df, 0.8, attributeReset())
treeError(df, 4)
