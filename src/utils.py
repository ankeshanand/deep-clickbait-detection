from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize

def tokenize_tweet( tweet ):
    """
    Return a list of cleaned word tokens from the raw review

    """
    #Remove any HTML tags and convert to lower case
    text = BeautifulSoup(tweet).get_text()

    # Remove links
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    # Remove RTs

    if text.startswith('RT'):
        text = text[2:]

    text = re.sub('[\W_]+', ' ', text)
    text = text.strip()
    words = [w for w in word_tokenize(text.lower()) if w is not 's']
    return words


def clean_tweet( tweet ):
    """
    Return a list of cleaned word tokens from the raw review

    """
    #Remove any HTML tags and convert to lower case
    text = BeautifulSoup(tweet).get_text()

    # Remove links
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    # Remove RTs

    if text.startswith('RT'):
        text = text[2:]

    text = re.sub('[\W_]+', ' ', text)
    text = text.strip().lower()
    return text