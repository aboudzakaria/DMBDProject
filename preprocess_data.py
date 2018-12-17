""" preprocess xml files to txt. """

import os, re, string, pickle
from lxml import etree
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from nltk.tokenize import TweetTokenizer

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
XML_DIR = os.path.join(CURRENT_DIR, "xml")
PKL_DIR = os.path.join(CURRENT_DIR, "pkl")

if not os.path.exists(XML_DIR):
    print("Error: xml directory not found!")
    exit(-1)

if not os.path.exists(PKL_DIR):
    os.mkdir(PKL_DIR)

i, n = 0, -1
FORCE_PROCESS = True
BREAK_ON_ERROR = True

# list of xml files that has no text
no_text = []
FORCE_NO_TEXT = True

# NLTK stuff
HAS_LETTERS = ".*[a-zA-Z]+"
STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = set(string.punctuation)
porter = PorterStemmer()

for filename in sorted(os.listdir(XML_DIR)):
    if filename.lower().endswith(".xml"):
        if n < 0 or i < n:
            i += 1
        else:
            break
        pklfilename = filename.replace(".xml", ".pkl")
        if os.path.isfile(os.path.join(PKL_DIR, pklfilename)) and not FORCE_PROCESS:
            continue

        try:
            # remove xml tags
            tree = etree.parse(os.path.join(os.path.join(XML_DIR, filename)))
            raw_text = etree.tostring(tree, encoding='utf8', method='text').decode('ascii')

            # check if there's no text
            if not raw_text:
                no_text.append(filename)
                if not FORCE_NO_TEXT:
                    print("SKIPPING: " + filename)
                    continue

            # nltk tokenize words
            tokens = word_tokenize(raw_text)

            # process tokens
            filtered_tokens = []
            for w in tokens:
                # filter punctuations
                if w in PUNCTUATIONS:
                    continue
                # filter stopwords
                if w in STOPWORDS:
                    continue
                # replace numbers into the word 'number'
                if not bool(re.match(HAS_LETTERS, w)):
                    w = "number"
                # stem the word
                w = porter.stem(w)
                # finally insert the word
                filtered_tokens.append(w.lower())

            # pickle data
            with open(os.path.join(PKL_DIR, pklfilename), 'wb') as pklfile:
                pickle.dump(filtered_tokens, pklfile)

        except Exception as err:
            print("Exception: {0}".format(err))
            if BREAK_ON_ERROR:
                break

pickle.dump(no_text, open("no_text.pkl", 'wb'))
