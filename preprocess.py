from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

from stringProcessor import processAll
from databaseHandler import getData
from databaseHandler import feeddbPreprocessed


df = getData()
preprocessing = Pipeline([
    ('encoder', OrdinalEncoder(dtype=int))
])

objectcols = ["sentiment"]

preprocessing.fit(df[objectcols])
df[objectcols] = preprocessing.transform(df[objectcols])
df['review'] = processAll(df['review'])

feeddbPreprocessed(df)
