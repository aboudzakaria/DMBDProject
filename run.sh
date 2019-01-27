#!/bin/sh
##########################################################################
# preprocess xml files
# parameters:
#  --no-stem          do not stem words
#  --no-wordnet       stop using wordnet synonyms lookup
#  --len-min LEN_MIN  minimum token word lenghth (default 3)
#  --df-min DF_MIN    token document frequency lower bound (default 7)
#  --df-max DF_MAX    token document frequency upper bound (default 3500)
##########################################################################

python3 preprocess_xml.py

##########################################################################
# X2 statistic feature selection
# parameters:
#  --threshold THRESHOLD threshold percentage (default 50)
##########################################################################

python3 X2.py

##########################################################################
# generate data
##########################################################################

python3 gen_data.py

