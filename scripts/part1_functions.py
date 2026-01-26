from scripts.pg_clean_text import strip_headers
from scripts.utils import read_file
from glob import iglob
import os
from typing import List
import re

def read_and_concat_books(books_repo: str) -> str:
    """
    This function will read the content of all the books in the given path and return a single string with all the content concatenated
    on top of each other.
    Example: 
    - Book1: "Hello World"
    - Book2: "Python is cool"
    - Output: "Hello World\n
              Python is cool"
    """
    # Read all books saved in .txt files
    books = list(iglob(os.path.join(books_repo, "*.txt")))
    books_content = list(map(read_file, books))
    # Remove the GP headers
    cleaned_contents = list(map(lambda x: strip_headers(''.join(x)), books_content))
    # Stack the content of all books on top of each other for form one document
    concat_contents = "\n".join(cleaned_contents)

    return concat_contents


def tokenize(text: str):
    """ 
    This function will tokenize the input text into individual words.
    - input: a string of text
    - output: a list of tokens (words)
    """
    # step1: handle spaces arround contractions
    text = re.sub(r"(n't|'ll|'re|'ve|'s|'d|'m)", r" \1", text)
    # step2: edge case ellipsis(protecting)
    text = text.replace('...', ' ELLIPSIS ')
    # step3: spaces around punctuation
    text = re.sub(r'([,!?;:{}()\[\]<>])', r' \1 ', text)
    # step4: period handling (space around periods that are not between digits)
    text = re.sub(r'(?<!\d)\.(?!\d)', r' . ', text)
    # step5: separate arithmetical operators
    text = re.sub(r'([+\-*/=])', r' \1 ', text)
    # step6: Split on whitespace
    tokens = text.split()
    # step7: restoring elipsis
    tokens = [token if token != 'ELLIPSIS' else '...' for token in tokens]

    return tokens

def remove_special_characters(tokens: List) -> List[str]:
    """
    This function will remove special characters. Keeping only alphabetic, numeric, or alphanumeric tokens
    - input: a list of tokens
    - output: a list of tokens without special characters
    """
    def token_is_valid(token):
        #step 1: checking alphanumeric
        if token.isalnum():
            return True
        #step 2: checking if it's a valid number (includes decimals)
        try:
            float(token)
            return True
        except ValueError:
            return False
    filtered_tokens = [token for token in tokens if token_is_valid(token)]
    return filtered_tokens

def remove_stopwords(tokens: List, stopwords: List = []) -> List[str]:
    """
    This function will remove common stopwords from the previous list of tokens to focus on meaningful words.
    - input: a list of tokens
    - output: a list of tokens without stopwords
    """
    # loading stopwords from the data file (data/stopwords.txt)
    if not stopwords:
        with open('data/stopwords.txt', 'r') as f:
            stopwords = [line.strip() for line in f.readlines()]
            
    # filtering out the stopwords 
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    return filtered_tokens

def preprocess(text: str) -> List[str]:
    """
    This function will preprocess the text by running a pipline consisting of:
    - Tokenization of text into units (i.e. tokens).
    - Removing special characters and numbers.
    - Removing stopwords.
    """
    functions = [tokenize, remove_stopwords, remove_special_characters]
    tokens = text
    for function in functions:
        tokens = function(tokens)

    return tokens
