import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re


def cleanHTML(text: str) -> str:
    """Cleans text from HTML tags

    Args:
        text (str): Text with HTML tags

    Returns:
        str: Text without HTML tags
    """
    cleaned = re.compile(r'<.*?>')  # regex expression for html tags
    return re.sub(cleaned, '', text)


def cleanSpecial(text: str) -> str:
    """Cleans text from special characters

    Args:
        text (str): Text with special characters

    Returns:
        str: Text without special characters
    """
    newtext = ''
    for c in text:
        if c.isalnum():
            # if current character is alphanumeric - pass
            newtext = newtext + c

        else:
            # otherwise replace with a space (so it doesn't create weird combined words)
            newtext = newtext + ' '
    return newtext


def cleanUpper(text: str) -> str:
    """Replace uppercase letters with lowercase

    Args:
        text (str): Text with uppercase letters

    Returns:
        str: Text without uppercase letters
    """
    return text.lower()


def cleanStopwords(text: str) -> str:
    """Remove stopwords from text

    Args:
        text (str): Text with stopwords

    Returns:
        str: Text without stopwords
    """
    stop_words = set(stopwords.words('english')
                     )  # use nltk for english stopwords
    words = word_tokenize(text)
    # return word if it's not in stopwords base
    return [w for w in words if w not in stop_words]


def stemmer(text: str) -> str:
    """Stem the text

    Args:
        text (str): Text to stem

    Returns:
        str: Stemmed text
    """
    ss = SnowballStemmer('english')  # use nltk stemmer for english words
    # return stemmed words for each word in text
    return " ".join([ss.stem(w) for w in text])


def processAll(srs: pd.Series) -> pd.Series:
    """Apply all the above methods shortcut

    Args:
        srs (pd.Series): Series to apply the methods

    Returns:
        pd.Series: Series after the methods were applied
    """
    srs = srs.apply(cleanHTML)
    srs = srs.apply(cleanSpecial)
    srs = srs.apply(cleanUpper)
    srs = srs.apply(cleanStopwords)
    srs = srs.apply(stemmer)
    return srs


def processAllstr(text: str) -> str:
    """Apply all the above methods shortcut

    Args:
        text (str): String to apply the methods

    Returns:
        str: String after the methods were applied
    """
    text = cleanHTML(text)
    text = cleanSpecial(text)
    text = cleanUpper(text)
    text = cleanStopwords(text)
    text = stemmer(text)
    return text
