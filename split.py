


def splitData(df):

    trainSet = df.sample(frac=0.7)

    testSet = df.loc[~df.index.isin(trainSet.index)]

    return{'trainSet':trainSet,'testSet':testSet}
