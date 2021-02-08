import os
import pandas as pd

## Methods for loading both csvs into dataframes and merge them into a single Dataframe.

def readDataset(path):
    stances = pd.read_csv(os.path.join(path,'train_stances.csv'))
    bodies = pd.read_csv(os.path.join(path,'train_bodies.csv'))
    return stances.merge(bodies,on='Body ID')

def readTestDataset(path):
    stances = pd.read_csv(os.path.join(path,'competition_test_stances.csv'))
    bodies = pd.read_csv(os.path.join(path,'competition_test_bodies.csv'))
    return stances.merge(bodies,on='Body ID')