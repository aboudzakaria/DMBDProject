""" preprocess xml files to txt. """

import os, re, string
from lxml import etree
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer

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

token_vector = {}

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
        raw_text = etree.tostring(tree, encoding='utf8', method='text').decode('ascii')

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
        doc_tokens = {}
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
            try:
                synonyms = [porter.stem(s.lower()) for s in wordnet.synsets(w)[0].lemma_names()]
                w = porter.stem(w)
                for s in synonyms:
                    if s in token_vector and w != s:
                        #print(w + ' -> ' + s)
                        w = s
                        break
            except:
                pass
            # keep words with length > 2
            if len(w) < 3:
                continue
            # stem the word
            w = porter.stem(w)
            # insert the word, count frequency
            doc_sequence.append(w)
            if w in token_vector:
                if w not in doc_tokens.keys():
                    doc_tokens[w] = 1 
                    token_vector[w] += 1 
                else:
                    doc_tokens[w] += 1
                    
            else:
                doc_tokens[w] = 1
                token_vector[w] = 1


        # write data
        with open(os.path.join(TKN_DIR, tknfilename), 'w') as tknfile:
            tknfile.write('\n'.join(sorted([','.join([key, str(value)]) for key, value in doc_tokens.items()])))

        with open(os.path.join(SEQ_DIR, seqfilename), 'w') as seqfile:
            seqfile.write(' '.join(doc_sequence))
            
    except Exception as err:
        print("Exception: {0}".format(err))
        if BREAK_ON_ERROR:
            break

# keep tokens that exists in more than 'idf_min' documents and less than 'idf_max'
idf_min = 7
idf_max = float('inf')
copy = token_vector.copy()
for key, value in copy.items():
    if value < idf_min or value > idf_max:
        del token_vector[key]

with open('no_text.csv', 'w') as no_textfile:
    no_textfile.write('\n'.join(sorted(no_text)))

with open('token_vector.tkn', 'w') as vecfile:
    vecfile.write('\n'.join(sorted([','.join([key, str(value)]) for key, value in token_vector.items()])))

print("token vector size: " + str(len(token_vector.keys())))
