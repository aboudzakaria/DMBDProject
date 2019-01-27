"generate boolean, tf, tfidf models"
import os
import math

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TKN_DIR = os.path.join(CURRENT_DIR, "tkn")
VEC_DIR = os.path.join(CURRENT_DIR, "vec")
N_DOCS = len([name for name in os.listdir('xml')])

if not os.path.exists(TKN_DIR):
    print("Error: tkn directory not found!")
    exit(-1)

if not os.path.exists(VEC_DIR):
    os.mkdir(VEC_DIR)

i, n = 0, -1
FORCE_PROCESS = True
BREAK_ON_ERROR = True

# read the token vector (ignore doc frequency)
tokenvec = []
token_df = []
with open('token_df_X2.tkn', 'r') as vecfile:
    veclines = vecfile.readlines()
    tokenvec = [l.split(',')[0] for l in veclines]
    token_df = [int(l.split(',')[1]) for l in veclines]

# file with doc,token,tfidf indices of the sparse matrix
sparsefile = open('data_sparse.csv', 'w')

for filename in sorted(os.listdir(TKN_DIR)):
    # skip not .xml files
    if not filename.lower().endswith(".tkn"):
        continue

    docid = int(filename.replace('.tkn', ''))

    # process limits
    if n < 0 or i < n:
        i += 1
    else:
        break

    # read document tokens and frequencies
    doc_tf = {}
    with open(os.path.join(TKN_DIR, filename), 'r') as tknfile:
        doc_tf = {l.split(',')[0]: l.split(',')[1].replace(
            '\n', '') for l in tknfile.readlines()}

    #print(doc_tf)

    # write token vector for each document
    vecfilename = filename.replace(".tkn", ".vec")
    with open(os.path.join(VEC_DIR, vecfilename), 'w') as vecfile:
        for tokid in range(len(tokenvec)):
            token = tokenvec[tokid]
            
            tf = int(doc_tf[token]) if token in doc_tf.keys() else 0
            idf = math.log10(N_DOCS/token_df[tokid])
            tfidf = tf * idf
            boolean = 1 if tf > 0 else 0

            value = '{0},{1},{2}'.format(str(boolean), str(tf), str(tfidf))
            vecfile.write(value + '\n')
            if tf > 0:
                sparsefile.write('{0},{1},{2}\n'.format(docid, tokid, value))
