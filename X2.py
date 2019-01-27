import sys
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--threshold', help='threshold percentage (default 50)', type=int)
args = argparser.parse_args()

threshold_percentage = args.threshold or 50
token_filename = "token_df_X2.tkn"

words = []
word_df = [] 
with open('token_df.tkn', 'r') as file:
    lines = file.readlines()
    for l in lines:
        arr = l.replace('\n',' ').split(',')
        words.append(arr[0])
        word_df.append(int(arr[1]))
#print(words)

if not os.path.isfile('X2.out'):

    docs_words = []
    zeros = [0 for _ in words]
    labels = []

    with open('train.csv') as f:
        ignore = True
        for line in f:
            if ignore:
                ignore = False
                continue 
            id, name, label = line.split(',')
            labels.append(int(label))
            #print(id,name,label)
            doc_words = []
            if not os.path.isfile(os.path.join('tkn',str(id)+'.tkn')):
                docs_words.append(zeros)
                continue
            with open(os.path.join('tkn',str(id)+'.tkn') ) as tkn_file:
                sparse_doc_words = [ str(tkn_line.split(',')[0]) for tkn_line in tkn_file]
                doc_words = [ int(w in sparse_doc_words) for w in words ]
            docs_words.append(doc_words)

    N = len(docs_words)

    #print(labels)
    #print(len(docs_words),len(docs_words[0]))
    #print(sum(len(s) for s in docs_words))

    def X2_value(word_idx, class_value):
        A = sum( [ int(class_value == labels[i] and docs_words[i][word_idx] != 0 ) for i in range(len(docs_words))]  )
        B = sum( [ int(class_value != labels[i] and docs_words[i][word_idx] != 0 ) for i in range(len(docs_words))]  )
        C = sum( [ int(class_value == labels[i] and docs_words[i][word_idx] == 0 ) for i in range(len(docs_words))]  )
        D = sum( [ int(class_value != labels[i] and docs_words[i][word_idx] == 0 ) for i in range(len(docs_words))]  )
        return ( N*(A*D - C*B)*(A*D - C*B) ) / ( (A+C) * (B+D) * (A+B) * (C+D) )
        
    X2_vals = []

    for i in range(len(words)):    
        if i%100 == 0:
            print(i)
        sys.stdout.flush()
        val = max(X2_value(i,0),X2_value(i,1))
        X2_vals.append(val)
    with open('X2.out', 'w') as f:
        for i in range(len(X2_vals)):
            if i == 0:
                f.write("{0}".format(X2_vals[i]))
            else:
                f.write(", {0}".format(X2_vals[i]))
                
else:
    with open( 'X2.out' ) as ff:
        line = ff.readline().split(',')
        X2_vals = [ float(l) for l in line]
            
sorted_X2_val = X2_vals
sorted_X2_val.sort(reverse=True)

threshold_idx = int( (threshold_percentage/100) * len(words) )
threshold = sorted_X2_val[threshold_idx]
#print(threshold)

with open(token_filename, 'w') as f:
    for i in range(len(words)):
        if X2_vals[i] > threshold:
            f.write("{0},{1}\n".format(words[i],word_df[i]))
