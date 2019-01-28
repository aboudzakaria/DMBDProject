import pandas as pd

train_data = pd.DataFrame()
test_data  = pd.DataFrame()
train_txt  = pd.read_csv('train.csv',sep=',')
test_txt  = pd.read_csv('test.csv',sep=',')
data_drop = pd.read_csv('no_text.csv',sep='.',header=None)
data_drop.columns = ['id','xml']

train_data['id']    = train_txt.id
train_data['label'] = train_txt.label

test_data['id']    = test_txt.id

list_drop = train_data[train_data.id.isin(data_drop.id)].index
train_data.drop(list_drop,inplace=True)

list_drop = test_data[test_data.id.isin(data_drop.id)].index
test_data.drop(list_drop,inplace=True)


train_data['txt'] = ['']*len(train_data.id)
test_data['txt']  = ['']*len(test_data.id)

for i in range(len(test_data.id)):
	f = open("seq/"+str(test_data.id.iloc[i])+".seq",'r')
	test_data.txt.iloc[i] = f.read()

test_data.to_csv('test_data.csv',sep=',',index=False)
print(test_data.head(5))

for i in range(len(train_data.id)):
	f = open("seq/"+str(train_data.id.iloc[i])+".seq",'r')
	train_data.txt.iloc[i] = f.read()

train_data.to_csv('train_data.csv',sep=',',index=False)
print(train_data.head(5))