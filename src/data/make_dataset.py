# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
#def main(input_filepath, output_filepath):
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # NEXT STEPS
    # fix age invalid val

    # LOAD DATA
    print("\nLOADING THE DATA...\n")
    train_data = pd.read_csv('data/raw/allhyperdata.csv')
    test_data = pd.read_csv('data/raw/allhypertest.csv')

    # DISPLAYING DATA INFO
    print("\n----- DISPLAYING DATA INFO -----\n")
    print(train_data.info())
    print(test_data.info())

    # DISPLAY HEAD
    print("\n----- DISPLAYING THE HEAD OF THE DATA -----\n")
    print(train_data.head())
    print(test_data.head())

    # CHECK SHAPE
    print("\n----- CHECKING THE SHAPE OF THE DATA -----\n")
    print("train_data: ", train_data.shape)
    print("test_data: ", test_data.shape)

    # ENSURE COLUMN NAMES MATCH
    print("\n----- CHECKING IF ALL COLUMN NAMES MATCH -----\n")
    print(train_data.columns)
    print(train_data.columns == test_data.columns)

    # USING VALUE COUNTS TO LOOK FOR MISSING OR INVALID DATA FOR EACH COLUMN
    print("\n--- LOOKING FOR MISSING OR INVALID DATA ---\n")
    for column in train_data.columns:
        print("\n", train_data[column].value_counts(), "\n")

    # DROP REDUNDANT COLUMNS
    print("\n----- DROPPING REDUNDANT COLUMNS -----\n")
    print("TBG was never measured, can drop 'TBG Measured' and 'TBG' columns")
    print("referral source also is a irrelevant column which can be dropped")
    train_data = train_data.drop(columns=['TBG Measured', 'TBG', 'Referral Source'])
    test_data = test_data.drop(columns=['TBG Measured', 'TBG', 'Referral Source'])
    print("the updated train data shape is: ", train_data.shape)
    print("the updated test data shape is: ", test_data.shape, "\n")
    print("it looks like all measured columns can be dropped as when they are false, they are '?' in the next column, so redundant")
    print("lets double check with some code...")
    print(train_data.loc[train_data['TSH Measured'] == 'f'].equals(train_data.loc[train_data['TSH'] == '?']))
    print(train_data.loc[train_data['T3 Measured'] == 'f'].equals(train_data.loc[train_data['T3'] == '?']))
    print(train_data.loc[train_data['TT4 Measured'] == 'f'].equals(train_data.loc[train_data['TT4'] == '?']))
    print(train_data.loc[train_data['T4U Measured'] == 'f'].equals(train_data.loc[train_data['T4U'] == '?']))
    print(train_data.loc[train_data['FTI Measured'] == 'f'].equals(train_data.loc[train_data['FTI'] == '?']))
    print("confirmed: all measured columns can be dropped as when they are false, they are '?' in the next column, so redundant")
    print('dropping all measured columns...')
    train_data = train_data.drop(columns=['TSH Measured', 'T3 Measured', 'TT4 Measured', 'T4U Measured', 'FTI Measured'])
    test_data = test_data.drop(columns=['TSH Measured', 'T3 Measured', 'TT4 Measured', 'T4U Measured', 'FTI Measured'])
    print("the updated train data shape is: ", train_data.shape)
    print("the updated test data shape is: ", test_data.shape, "\n")

    # CHANGE '?' TO NAN
    print("\n----- CHANING '?' TO NAN -----\n")
    print("'?' is used for unknown values in integer or float type columns")
    print("lets count the occurences in each column, to see which columns '?' appears in...")
    print("TRAIN DATA")
    for column in train_data.columns:
        print(column, ": ", (train_data[column] == '?').sum())
    print("\nTEST DATA")
    for column in train_data.columns:
        print(column, ": ", (test_data[column] == '?').sum())
    print("'?' only appears in columns with integer or float type data (opposed to the t/f columns) with the one exception being the 'Sex' column")
    print("we will change '?' to NaN in all integer and float columns, but leave it as ?, meaning unknown in the sex column...")
    train_data['Age'] = train_data['Age'].replace('?', np.nan) # ? does not appear in age in test data
    columns = train_data.columns[16:21] # TSH, T3 TT4, T4U, FTI
    for column in columns:
        train_data[column] = train_data[column].replace('?', np.nan)
        test_data[column] = test_data[column].replace('?', np.nan)
    print("the updated counts should all be zero, except for in 'Sex'")
    for column in train_data.columns:
        print(column, ": ", (train_data[column] == '?').sum())
        print(column, ": ", (test_data[column] == '?').sum())
    print("success\n")

    # CHANGING APPROPRIATE COLUMNS TO TYPE FLOAT
    print("\n----- CHANING APPROPRIATE COLUMNS TO TYPE FLOAT -----\n")
    train_data['Age'] = train_data['Age'].astype('float') # test data already has type int since it didnt have '?'
    columns = train_data.columns[16:21] # TSH, T3 TT4, T4U, FTI
    for column in columns:
        train_data[column] = train_data[column].astype('float')
        test_data[column] = test_data[column].astype('float')
    print(train_data.info())
    print(test_data.info())

    # CHECK MIN, MAX, AND MEDIAN VALUES TO CHECK FOR INVALID VALUES
    print("\n----- CHECK MIN AND MAX VALUES TO CHECK FOR INVALID VALUES -----\n")
    print("TRAIN DATA")
    print("Age", ": ", train_data["Age"].min(), " ", train_data["Age"].max(), " ", train_data["Age"].median())
    columns = train_data.columns[16:21]
    for column in columns:
        print(column, ": ", train_data[column].min(), " ", train_data[column].max(), " ", train_data[column].median())
    print("TEST DATA")
    print("Age", ": ", test_data["Age"].min(), " ", test_data["Age"].max(), " ", test_data["Age"].median())
    for column in columns:
        print(column, ": ", test_data[column].min(), " ", test_data[column].max(), " ", test_data[column].median())

    #print("checking how many occurences exist below and above a certain value")
    #print((train_data['Age'] < 1).sum())
    #print((train_data['Age'] < 1).sum())
    #print((train_data['Age'] < 1).sum())
    #print((train_data['Age'] < 1).sum())
    #print((train_data['Age'] < 1).sum())
    #print((train_data['Age'] < 1).sum())
    #print("according to research we can only conclude with absolute certainity the only value that is invalid is that age = 455, which will be replaced with NAN")

    #train_data['Age'].where([train_data['Age'] > 122], 999)
    #train_data['Age'] = np.where(train_data['Age'] > 122,np.nan)
    #print("Age", ": ", train_data["Age"].min(), " ", train_data["Age"].max(), " ", train_data["Age"].median())

    # CHANGE NAN VALUES TO THE MEAN OF THE COLUMN
    print("\n----- CHANGE NAN VALUES TO THE MEDIAN OF THE COLUMN -----\n")
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Age'] = train_data['Age'].astype('int64') # change Age back into int (Age was never suppose to be float, but it had to be temporarly since it had nan)
    columns = train_data.columns[16:21]
    for column in columns:
        print(column, ": ", train_data[column].isnull().sum())
        print(column, ": ", test_data[column].isnull().sum())
        train_data[column].fillna(train_data[column].median(), inplace=True)
        test_data[column].fillna(test_data[column].median(), inplace=True)
        print(column, ": ", train_data[column].isnull().sum())
        print(column, ": ", test_data[column].isnull().sum())
    print("all NAN values were replaced with the median value of the column")

    # SLICING IRRELEVANT VALUES ON THE END OF RESULT
    print("\n----- SLICING IRRELEVANT VALUES ON THE END OF RESULT -----\n")
    print("the 'Result' column has irrelevant data following the '.' that can be sliced")
    print("the updated result column value counts is...")
    def aux(val):
        val = val[0:val.index('.')]
        return val
    train_data["Result"] = train_data["Result"].apply(aux)
    test_data["Result"] = test_data["Result"].apply(aux)
    print(train_data['Result'].value_counts(), "\n")
    print(test_data['Result'].value_counts(), "\n")

    # USE GET_DUMMIES() FOR ENCODING OF CLASSIFICATION VALUES
    print("\n----- USE GET_DUMMIES() FOR ENCODING OF CLASSIFICATION VALUES -----\n")
    df = pd.DataFrame()
    columns = train_data.columns[1:16]
    for column in columns:
        df2 = pd.get_dummies(train_data[column], prefix=column)
        train_data = train_data.drop(columns=[column])
        df = pd.concat([df, df2], axis=1)
    age_col = train_data['Age']
    train_data = train_data.drop(columns=['Age'])
    train_data = pd.concat([age_col, df, train_data], axis=1)
    df2 = pd.get_dummies(train_data['Result'], prefix='Result')
    train_data = train_data.drop(columns=['Result'])
    train_data = pd.concat([train_data, df2], axis=1)
    train_data['Result_secondary toxic'] = 0 # train data does not have a value here so dummies will not create a column for it
    train_data['Result_secondary toxic'] = train_data['Result_secondary toxic'].astype('uint8')

    df = pd.DataFrame()
    columns = test_data.columns[1:16]
    for column in columns:
        df2 = pd.get_dummies(test_data[column], prefix=column)
        test_data = test_data.drop(columns=[column])
        df = pd.concat([df, df2], axis=1)
        if(column == 'Hypopituitary'): # test data does not have a t value here so dummies will not create a column for it
            df['Hypopituitary_t'] = 0
            df['Hypopituitary_t'] = df['Hypopituitary_t'].astype('uint8')
    age_col = test_data['Age']
    test_data = test_data.drop(columns=['Age'])
    test_data = pd.concat([age_col, df, test_data], axis=1)
    df2 = pd.get_dummies(test_data['Result'], prefix='Result')
    test_data = test_data.drop(columns=['Result'])
    test_data = pd.concat([test_data, df2], axis=1)   

    # FINAL OVERLOOK OF THE DATAFRAMES
    print("\n----- FINAL OVERLOOK OF THE DATAFRAMES -----\n")
    print(train_data.info())
    print(test_data.info())

    # SAVE PROCESSED DATA IN PROCESSED FOLDER
    train_data.to_csv('data/processed/train_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)

    return

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
