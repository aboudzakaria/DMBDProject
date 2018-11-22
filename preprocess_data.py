""" preprocess xml files to txt. """

import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
XML_DIR = os.path.join(CURRENT_DIR, "data")
TXT_DIR = os.path.join(CURRENT_DIR, "text")

if not os.path.exists(TXT_DIR):
    os.mkdir(TXT_DIR)

i, n = 0, -1
FORCE_PROCESS = True
BREAK_ON_ERROR = True

for filename in sorted(os.listdir(XML_DIR)):
    if filename.lower().endswith(".xml"):
        if n < 0 or i < n:
            i += 1
        else:
            break
        txtfilename = filename.replace(".xml", ".txt")
        if os.path.isfile(os.path.join(TXT_DIR), txtfilename) and not FORCE_PROCESS:
            continue

        #try: process!
        # to be continued
