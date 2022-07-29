import sqlite3
import pandas as pd

dbpath = "./Data/base.db"
csvpath = "./Data/IMDB-Dataset.csv"


def feeddb(path: str):
    """Feed the database with a CSV file

    Args:
        path (str): Path to the CSV file
    """
    con = sqlite3.connect(dbpath)
    cur = con.cursor()
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS reviews (review text, sentiment text)''')
    df = pd.read_csv(path)
    df.to_sql('reviews', con, if_exists="replace", index=False)
    con.close()


def feeddbPreprocessed(df: pd.DataFrame):
    """Feed the database with a preprocessed DataFrame

    Args:
        df (pd.DataFrame): Preprocessed DataFrame
    """
    con = sqlite3.connect(dbpath)
    cur = con.cursor()
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS reviewspreprocessed (review text, sentiment number)''')

    df.to_sql('reviewspreprocessed', con, if_exists="replace", index=False)
    con.close()


def getData() -> pd.DataFrame:
    """Get data from the database

    Returns:
        pd.DataFrame: Data as a pandas DataFrame
    """
    con = sqlite3.connect(dbpath)
    df = pd.read_sql_query("SELECT * FROM reviews", con)
    con.close()
    return df


def getPreprocessedData() -> pd.DataFrame:
    """Get preprocessed data from the database

    Returns:
        pd.DataFrame: Data as a pandas DataFrame
    """
    con = sqlite3.connect(dbpath)
    df = pd.read_sql_query("SELECT * FROM reviewspreprocessed", con)
    con.close()
    return df


# View data stored in database file
# print(getData().head())
# print(getPreprocessedData().head())
