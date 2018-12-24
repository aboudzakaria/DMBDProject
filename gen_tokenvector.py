
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TKN_DIR = os.path.join(CURRENT_DIR, "tkn")
VEC_DIR = os.path.join(CURRENT_DIR, "vec")

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
with open('token_vector.tkn', 'r') as vecfile:
    tokenvec = [l.split(',')[0] for l in vecfile.readlines()]


for filename in sorted(os.listdir(TKN_DIR)):
    # skip not .xml files
    if not filename.lower().endswith(".tkn"):
        continue

    # process limits
    if n < 0 or i < n:
        i += 1
    else:
        break

    # read document tokens and frequencies
    doc_tokens = {}
    with open(os.path.join(TKN_DIR, filename), 'r') as tknfile:
        doc_tokens = {l.split(',')[0]: l.split(',')[1].replace('\n', '') for l in tknfile.readlines()}

    #print(doc_tokens)

    # write token vector for each document
    vecfilename = filename.replace(".tkn", ".vec")
    with open(os.path.join(VEC_DIR, vecfilename), 'w') as vecfile:
        for token in tokenvec:
            freq = int(doc_tokens[token]) if token in doc_tokens.keys() else 0
            value = '1' if freq > 0 else '0' # str(freq) ?
            vecfile.write(value + '\n')

