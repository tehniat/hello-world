# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:19:13 2019

@author: hp z book
"""
import functools# library for reduce function
from functools import reduce
import nltk
nltk.download('punkt')
import re
import numpy as np
import tarfile
import numpy as np
from functools import reduce
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
try:
    #tar.gz data-set get saved on "~/.keras/datasets/" path
   path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
 #   print('Error downloading dataset, please download it manually:\n'
  #        '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
   #       '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
 #reading a tar.gz file
#path="E:/babi_tasks_1-20_v1-2.tar"
tar = tarfile.open(path)
# There are 20 tasks in data . Each task represent diffferent type of questions
#Here  we are solving task1 which has following type of format 
#ID text
#ID text
#ID text
#ID question[tab]answer[tab]supporting_fact ID. where supporting fact ID represent the line number of the sentence where answer can be deducted
#Each text represent a user story. so we have to first separate out sentences of stories.
#and have to convert them in the given format e.g in line 27----30.
# for the above purpose we make a function parse_stories(string). where string will be the all the lines in data file.
# so we first read all the lines in datafiles and parse it in a particlar format.
#our data file is represented by Tarfile obect tar which represent archive data stored in block.
#so we first extract the tar file blocks of tarin file as well as test file and then pass it to a function which parse it in the format(story sentences, question, answer)
# extractfile(member) Extract a member from the archive as a file object.
#train_file=tar.extractfile('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt')
#test_file=tar.extractfile('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt')
#we have now separated train and test story files from archive.  
#lines=train_file.readlines()
#now we need a  function parsestories()which takes lines of file and return tuple of (sent, qes, anser)
def parse_stories(lines):
     # list of stories=story[]
     #list of data=list of tupleof format story,q,ans)
 story=[]
 data=[]
     #for each line in lines decode string in utf-8 format scheme and strip off
    #leading and trailing whitespace characters.
 for line in lines:
        line= line.decode('utf-8').strip()
        print(line)
        #split the line onlyon the bases of one space it means it separate id from sentencei.e ['1 a b c'] became ['1' , 'a b c'] and store 1 in nid and remaing string in line
        nid, line=line.split(' ', 1)
        # now convert string '1' into intger
        nid=int(nid)
        # now check what is nid if ID is 1 it means it is start of a new story 
        #in data.so reset the story
        if nid==1:
            story=[]
        # check if there are any tabs in lines . If yes this means this is the line 
        #where question and answer  and the line number of the sentence from which answer is deducted is written.
        # this sentence is called supporting sentence.  
        if '\t' in line:
            # split sentence ith respect to tab to separate out story  question and answer
          question,answer,supportingID=line.split('\t')
        #now we have separated out question, answer and supportingID 
        #Now separate out words of questions. i.e tokenize the question 
        #or this purpose we can use tokenizer function of nltk library.
          from nltk.tokenize import word_tokenize  
          q=word_tokenize(question)
          #now separte out sentences other than questions i.e substories
          # i.e if there are no tab pick the lines and store it in substory list
          # initially it is empty list 
          substory = [x for x in story if x]
          #Now we have to store the tuple in list data as [ [], [qw1 qw2..] , answer ]
          data.append((substory,q, answer))
          #append space that behave as a separarator between two stories. 
        else:
            
            #these are  sentences of story
            # therefore decode each sentence line b line and tokenize the sentences one by one and append in story
            sentenceWords=nltk.word_tokenize(line)
            story.append(sentenceWords)
 return data 
#Now make function get_stories() that takes datafilename as input call function parse stories to 
    #convert data in the list of type[[], ['qw1', 'qw2'   ..], 'answer'] or [['sw1'....],['qw1' 'qw2'...],'answer']
    # and separate out the list of story words and make them column vector so the one hot vector can 
    #be made for the story words
def get_stories(fileName):
#read the file and parse 10k stories
  data=parse_stories(fileName.readlines())
#now flatten the list using lamda and reduce function. lamda is a funtion 
#which takes variable as arguments and an expression. It returns the result of 
#evaluated expression. e.g lamda x,y: x+y. if x=1, y=1 then it will return 2.
#lamda function is used with reduce() . Reduce function takes two arguments a function
# and a list. It will do like this reduce(lamda x,y : x+y , [10 12,13])=10+12=22
#22+13=33
#so to flatten list we use lamda and reduce function in this way
 
  flatten=lambda data: reduce(lambda x, y: x+y, data)
  #so flatten will become a function which flatten the list
 # the above line pick two elements of data list . concatenate them. store the result 
#in temp variable. Then take third element concatenate it with temp and store the result
#and so on then the right most lamda function store the result in data .return it 
#in flatten and immediately wipe out.(lamda is an anonymous function without name) 
 #but since we do'nt have library to have reduce function we are doing in this way
 #storyWordsInColumn= zip(*data[0])
 ## now update data list in this format[['w1,'w2'..], ['qw1','qw2',...], answer]   
 #i.e create a list name data having above type of elements
  data=[(flatten(story), q, answer) for story, q, answer in data]
  return data
#now use above function to get train and test stories
train_stories=get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt'))
test_stories=get_stories(tar.extractfile('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt')) 
print(train_stories[0])
# now start vectorization for sentences, questions and answers
#Find how many unique words in train and test and in questions. 
# answer is already present in questions
# set function is used to find distinct values
#vocab=set()
#for every story, question and answer present in train and test set pick 
#distinct word and store in vocab
unk=['unk']
total_unique_vocab=set()
words=[]
#swords=[]
answer_words=[]
for story, q, answer in train_stories+test_stories:
    answer_words.append(answer) 
    total_unique_vocab=(set(story+q +answer_words+unk))
total_unique_vocab=sorted(total_unique_vocab)
total_vocab_size=len(total_unique_vocab) #length of total vocab in story and question+1 for 'unkown'
#find the length of the longest  story in train or test stories
#max function takes two parameter a list and a key which tells take maximum by which 
#operation i.e max(list, key=len) will return the maximum element by length
#+operator is overloaded to take union
# for each stor in train or test stories pick story find length .Map function map the
#length function to each iteratablei.e find length of each element in list an find 
#maximum out of it.
len_longest_story=max(map(len, (x for x,_,_ in train_stories+test_stories)))
 #similarly find longest question length
len_longest_question=max(map(len, (x for _ , x, _ in train_stories+test_stories)))
len_longest_answer=max(map(len,(x for _,_, x in train_stories+test_stories)))
SEQLEN=max(len_longest_story,len_longest_question,len_longest_answer)
#now create dictionary i.e key /value pair for making word 2 index and index to
#word
#Create word to index dictionary . First create tuple (integer, word) usig enumerate 
#function it will return the list .Then initialize a dictionary with dict() function
#story_word_idx=dict((c, i+1) for i, c in enumerate(vocab_story))
#simiarly create index to word dictionary
#story_idx_word=dict((i+1, c) for i, c in enumerate(vocab_story))     
#make function vectorize to make one hot vector of the data
#question_word_idx=dict((c, i+1) for i, c in enumerate(vocab_question))
#question_idx_word=dict((i+1, c) for i, c in enumerate(vocab_question))
#answer_word_idx=dict((c, i+1) for i, c in enumerate(vocab_answer))
#simiarly create index to word dictionary
#answer_idx_word=dict((i+1, c) for i, c in enumerate(vocab_answer))  

word_idx=dict((c, i+1) for i, c in enumerate(total_unique_vocab))
idx_word=dict((i+1, c) for i, c in enumerate(total_unique_vocab))
#word_idx[0]='unk'
#idx_word['unk']=0
#separate out story sentences and store in list
train_stories_sentences=[]
for x, _, _  in train_stories:
  train_stories_sentences.append(x)
#def vectorize(any_text,SEQLEN,total_vocab_size ):
    #initialize vector of story, question and answer
    #X is any vector fill with zeros vector
 #   import numpy as np
  #  X= np.zeros((len(any_text), SEQLEN, total_vocab_size), dtype=np.bool)
    
   # print(np.array(X).shape)
    #for i, every_example in enumerate(any_text):
     #   print("i= ", i)
      #  for j,eachWord in enumerate(every_example):
       #    print("j=", j)
        #   X[i, j, word_idx[eachWord] ]=1
           
    #return  X
#X_storyVec= vectorize(train_stories_sentences,SEQLEN,total_vocab_size)
X=[]#matrix of all stories
Xq=[]#matrix of all questions     
x = [] #vector of each word in each story
xq=[]
ya=[]
Ya=[]   
 #pick each story ,question and answer from train_stories
x=np.zeros((total_vocab_size),dtype=np.bool)
xq=np.zeros((total_vocab_size), dtype=np.bool)
ya=np.zeros((total_vocab_size), dtype=np.bool)
for story,question, answer in train_stories:
        # pick words of story pick its index from dictionary and store the index 
        # in the list named x
        #vector of lenght 26 for intital 'unknown'
        
        for w in story:
            x[word_idx[w]]=True
        X.append(x)    
        #print(X) 
        #similarly pick words of question and make list of inices of its words
        for w in question:
            xq[word_idx[w]]=True
        Xq.append(xq)
        # since there are only one wword in anserwer so  make 
        #list of its index. So directly make one hot vector of it. 
        #fill y vector of length word_idx with 0(i.e total vocab)+1 as index 0 is reserved
        #y=np.zeros(len(word_idx)+1)
        #now store 1 at the index of the answer word
       # y[word_idx[answer]]=1
        # append the vector x in list
        for w in answer:
            ya[word_idx[w]]=True
        Ya.append(ya)
        
         
        
    return (pad_sequences(X, maxlen=len_longest_story), pad_sequences(Xq,maxlen=len_longest_question),np.array(Y))  
  #now call vectorize function to vectorize train story words, question words and 
#answer word
story_train,question_train,answer_train=vectorize(train_stories, word_idx, len_longest_story,len_longest_question)
story_test,question_test,answer_test=vectorize(test_stories, word_idx, len_longest_story,len_longest_question) 
       
    