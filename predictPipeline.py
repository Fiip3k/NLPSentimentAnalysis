import pickle
import numpy as np
from stringProcessor import processAllstr
from nltk.tokenize import word_tokenize


def pipePredict(text: str):
    """Simple pipeline for straight text-to-prediction conversion

    Args:
        text (str): Text for model input

    Returns:
        List of a dictionary: Object ready for json communication
    """
    # load model and vocab data from files created by model.py
    model = pickle.load(open('model1.pkl', 'rb'))
    vocab = pickle.load(open('bow.pkl', 'rb'))

    # ready text for model input
    processedtext = processAllstr(text[:])
    bow = []
    words = word_tokenize(processedtext)
    for word in words:
        bow.append(words.count(word))
    X = []
    for i in vocab:
        X.append(processedtext.count(i[0]))
    X = np.array(X).reshape(1, 1000)

    # get predictions from model
    predictions = model.predict(X)

    # return data ready for json
    return [{'text': text, 'pred': int(predictions),
            'label': 'Positive' if int(predictions) == 1 else 'Negative'}]
