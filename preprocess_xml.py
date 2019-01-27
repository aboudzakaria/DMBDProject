""" preprocess xml files to txt. """

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize
from lxml import etree
import string
import re
import os
import sys
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--no-stem', help='do not stem words', action='store_true')
argparser.add_argument(
    '--no-wordnet', help='stop using wordnet synonyms lookup', action='store_true')
argparser.add_argument(
    '--len-min', help='minimum token word lenghth (default 3)', type=int)
argparser.add_argument(
    '--df-min', help='token document frequency lower bound (default 7)', type=int)
argparser.add_argument(
    '--df-max', help='token document frequency upper bound (default 3500)', type=int)
args = argparser.parse_args()

# extract command arguments
USE_STEM = not args.no_stem
USE_WORDNET = not args.no_wordnet
TOKEN_LEN_MIN = args.len_min or 3
DF_MIN = args.df_min or 7
DF_MAX = args.df_max or 3500

# delete this file yrham jiddak
if os.path.exists('X2.out'):
    os.remove('X2.out')
    
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
XML_DIR = os.path.join(CURRENT_DIR, "xml")
TKN_DIR = os.path.join(CURRENT_DIR, "tkn")
SEQ_DIR = os.path.join(CURRENT_DIR, 'seq')

if not os.path.exists(XML_DIR):
    print("Error: xml directory not found!")
    exit(-1)

if not os.path.exists(TKN_DIR):
    os.mkdir(TKN_DIR)

if not os.path.exists(SEQ_DIR):
    os.mkdir(SEQ_DIR)


i, n = 0, -1
FORCE_PROCESS = True
BREAK_ON_ERROR = True

# list of xml files that has no text
no_text = []
FORCE_NO_TEXT = False

# NLTK stuff
HAS_NUMBERS = ".*[0-9]+"
STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = set(string.punctuation)
porter = PorterStemmer()

#
# begin processing
#

token_df = {}

for filename in sorted(os.listdir(XML_DIR)):
    # skip not .xml files
    if not filename.lower().endswith(".xml"):
        continue

    # process limits
    if n < 0 or i < n:
        i += 1
    else:
        break

    tknfilename = filename.replace(".xml", ".tkn")
    seqfilename = filename.replace(".xml", ".seq")

    # already processed?
    if os.path.isfile(os.path.join(TKN_DIR, tknfilename)) and not FORCE_PROCESS:
        continue

    try:
        # remove xml tags
        tree = etree.parse(os.path.join(os.path.join(XML_DIR, filename)))
        raw_text = etree.tostring(
            tree, encoding='utf8', method='text').decode('ascii')

        # skip documents with no texts
        if not raw_text:
            no_text.append(filename)
            if not FORCE_NO_TEXT:
                continue

        # manual fixes
        #raw_text = raw_text.replace('-', ' ') # dashes!
        #raw_text = raw_text.replace('/', ' ') # slashes!
        #raw_text = raw_text.replace('+', ' ') # pluses!
        raw_text = raw_text.lower()

        # nltk tokenize words
        tokens = word_tokenize(raw_text)

        # process tokens
        doc_tf = {}
        doc_sequence = []
        for w in tokens:
            # filter punctuations with extra check!
            for p in PUNCTUATIONS:
                w = w.replace(p, '')
            # filter stopwords
            if w in STOPWORDS:
                continue
            # replace numbers into the word 'digit'
            if bool(re.match(HAS_NUMBERS, w)):
                w = "digit"
            # try finding a synonym which is already in the vector!
            if USE_WORDNET:
                try:
                    synonyms = [porter.stem(s.lower()) if USE_STEM else s.lower()
                                for s in wordnet.synsets(w)[0].lemma_names()]
                    if USE_STEM:
                        w = porter.stem(w)
                    for s in synonyms:
                        if s in token_df and w != s:
                            #print(w + ' -> ' + s)
                            w = s
                            break
                except:
                    pass
            # keep words with length > 2
            if len(w) < TOKEN_LEN_MIN:
                continue
            # stem the word
            if USE_STEM:
                w = porter.stem(w)
            # insert the word, count frequency
            doc_sequence.append(w)
            if w in token_df:
                if w not in doc_tf.keys():
                    doc_tf[w] = 1
                    token_df[w] += 1
                else:
                    doc_tf[w] += 1

            else:
                doc_tf[w] = 1
                token_df[w] = 1

        # write data
        with open(os.path.join(TKN_DIR, tknfilename), 'w') as tknfile:
            tknfile.write('\n'.join(
                sorted([','.join([key, str(value)]) for key, value in doc_tf.items()])))

        with open(os.path.join(SEQ_DIR, seqfilename), 'w') as seqfile:
            seqfile.write(' '.join(doc_sequence))

    except Exception as err:
        print("Exception: {0}".format(err))
        if BREAK_ON_ERROR:
            break

# keep tokens that exists in more than 'df_min' documents and less than 'df_max'
copy = token_df.copy()
for key, value in copy.items():
    if value < DF_MIN:
        del token_df[key]
    if value > DF_MAX:
        #print(key)
        del token_df[key]

with open('no_text.csv', 'w') as no_textfile:
    no_textfile.write('\n'.join(sorted(no_text)))

with open('token_df.tkn', 'w') as vecfile:
    vecfile.write('\n'.join(
        sorted([','.join([key, str(value)]) for key, value in token_df.items()])))

print("token vector size: " + str(len(token_df.keys())))
