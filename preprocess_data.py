""" preprocess xml files to txt. """

import os, re, string
from lxml import etree
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
#from nltk.tokenize import TweetTokenizer

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
XML_DIR = os.path.join(CURRENT_DIR, "xml")
TKN_DIR = os.path.join(CURRENT_DIR, "tkn")

if not os.path.exists(XML_DIR):
    print("Error: xml directory not found!")
    exit(-1)

if not os.path.exists(TKN_DIR):
    os.mkdir(TKN_DIR)

i, n = 0, -1
FORCE_PROCESS = True
BREAK_ON_ERROR = True

# list of xml files that has no text
no_text = []
FORCE_NO_TEXT = True

# NLTK stuff
HAS_NUMBERS = ".*[0-9]+"
STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = set(string.punctuation)
porter = PorterStemmer()

#
# begin processing
#

token_vector = set([])

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

    # already processed?
    if os.path.isfile(os.path.join(TKN_DIR, tknfilename)) and not FORCE_PROCESS:
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

        # manual fixes
        raw_text = raw_text.replace('-', ' ') # dashes!
        raw_text = raw_text.replace('/', ' ') # slashes!
        raw_text = raw_text.replace('+', ' ') # pluses!
        raw_text = raw_text.lower()

        # nltk tokenize words
        tokens = word_tokenize(raw_text)

        # process tokens
        clean_tokens = []
        for w in tokens:
            # filter punctuations with extra check!
            for p in PUNCTUATIONS:
                w = w.replace(p, '')
            # filter stopwords
            if w in STOPWORDS:
                continue
            # replace numbers into the word 'numero'
            if bool(re.match(HAS_NUMBERS, w)):
                w = "numero"
            # try finding a synonym which is already in the vector!
            try:
                synonyms = [porter.stem(s.lower()) for s in wordnet.synsets(w)[0].lemma_names()]
                for s in synonyms:
                    if s in token_vector and w != s:
                        print(w + ' -> ' + s)
                        w = s
            except:
                pass
            # keep words with length > 2
            if len(w) < 3:
                continue
            # stem the word
            w = porter.stem(w)
            # finally insert the word
            clean_tokens.append(w)
            # add once to the vector set
            token_vector.add(w)

        # write data
        with open(os.path.join(TKN_DIR, tknfilename), 'w') as tknfile:
            tknfile.write('\n'.join(clean_tokens))

    except Exception as err:
        print("Exception: {0}".format(err))
        if BREAK_ON_ERROR:
            break


with open('no_text.csv', 'w') as no_textfile:
    no_textfile.write('\n'.join(sorted(no_text)))

with open('token_vector.tkn', 'w') as vecfile:
    vecfile.write('\n'.join(sorted(token_vector)))

print("token vector size: " + str(len(token_vector)))
