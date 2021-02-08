import re

removeNL = lambda x : x.replace("\n"," ")
'''replaces newline character with spaces'''

removeSC = lambda x : re.sub('[^A-Za-z0-9 ]+', ' ', x)
'''replaces special characters with spaces'''

def assignInt4Class(x):
    '''assigns Integer value to stances'''
    if(x=="unrelated"):
        return 0
    elif(x=="disagree"):
        return 1
    elif(x=="discuss"):
        return 2
    elif(x=="agree"):
        return 3


def assignIntBinary(x):
    '''assigns Integer value to stances'''
    if(x=="unrelated"):
        return 0
    else:
        return 1


def preprocessDF(df_merged, binaryclass = False):
    '''removes newline, special characters from articlebody, headline and assigns Integer to stance'''
    df_merged["articleBody"] = df_merged["articleBody"].apply(removeNL)
    df_merged["articleBody"] = df_merged["articleBody"].apply(removeSC)

    df_merged["Headline"] = df_merged["Headline"].apply(removeNL)
    df_merged["Headline"] = df_merged["Headline"].apply(removeSC)

    if(binaryclass):
        df_merged["Stance"] = df_merged["Stance"].apply(assignIntBinary)
    else:
        df_merged["Stance"] = df_merged["Stance"].apply(assignInt4Class)

    return df_merged


def getBalancedData(ppdf,binaryclass=False):
    '''oversampling approximation for classes'''

    vc = ppdf["Stance"].value_counts()
    if(binaryclass):
        counts = [0,0]
    else:
        counts = [0,0,0,0]
    
    for index,value in vc.iteritems():
        counts[index] = value
    maxval = max(counts)
    counts = [int(maxval/x)+1 for x in counts]
    retcount = lambda x : counts[x]
    repeatSer = ppdf["Stance"].apply(retcount)
    ppdfoversampled = ppdf.reindex(ppdf.index.repeat(repeatSer))
    return ppdfoversampled
