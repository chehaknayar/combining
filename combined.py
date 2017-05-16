#do something to remove full stops from sent and content.correct other functions that take listfreq in their definition
import nltk
from nltk.corpus import stopwords 			
from nltk.tokenize import word_tokenize 
from collections import Counter
import re
import sqlite3
import math
import string
from itertools import chain
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from pywsd.utils import lemmatize, porter, lemmatize_sentence, synset_properties
import string



#EN_STOPWORD
EN_STOPWORDS = stopwords.words('english')

def compare_overlaps(context, synsets_signatures, \
                     nbest=False, keepscore=False, normalizescore=False):
    """
    Calculates overlaps between the context sentence and the synset_signture
    and returns a ranked list of synsets from highest overlap to lowest.
    """
    overlaplen_synsets = [] # a tuple of (len(overlap), synset).
    for ss in synsets_signatures:
        overlaps = set(synsets_signatures[ss]).intersection(context)
        overlaplen_synsets.append((len(overlaps), ss))

    # Rank synsets from highest to lowest overlap.
    ranked_synsets = sorted(overlaplen_synsets, reverse=True)

    # Normalize scores such that it's between 0 to 1.
    if normalizescore:
        total = float(sum(i[0] for i in ranked_synsets))
        ranked_synsets = [(i/total,j) for i,j in ranked_synsets]

    if not keepscore: # Returns a list of ranked synsets without scores
        ranked_synsets = [i[1] for i in sorted(overlaplen_synsets, \
                                               reverse=True)]

    if nbest: # Returns a ranked list of synsets.
        return ranked_synsets
    else: # Returns only the best sense.
        return ranked_synsets[0]
def simple_signature(ambiguous_word, pos=None, lemma=True, stem=False, \
                     hyperhypo=True, stop=True):
    """
    Returns a synsets_signatures dictionary that includes signature words of a
    sense from its:
    (i)   definition
    (ii)  example sentences
    (iii) hypernyms and hyponyms
    """
    synsets_signatures = {}
    for ss in wn.synsets(ambiguous_word):
        try: # If POS is specified.
            if pos and str(ss.pos()) != pos:
                continue
        except:
            if pos and str(ss.pos) != pos:
                continue
        signature = []
        # Includes definition.
        ss_definition = synset_properties(ss, 'definition')
        signature+=ss_definition
        # Includes examples
        ss_examples = synset_properties(ss, 'examples')
        signature+=list(chain(*[i.split() for i in ss_examples]))
        # Includes lemma_names.
        ss_lemma_names = synset_properties(ss, 'lemma_names')
        signature+= ss_lemma_names

        # Optional: includes lemma_names of hypernyms and hyponyms.
        if hyperhypo == True:
            ss_hyponyms = synset_properties(ss, 'hyponyms')
            ss_hypernyms = synset_properties(ss, 'hypernyms')
            ss_hypohypernyms = ss_hypernyms+ss_hyponyms
            signature+= list(chain(*[i.lemma_names() for i in ss_hypohypernyms]))

        # Optional: removes stopwords.
        if stop == True:
            signature = [i for i in signature if i not in EN_STOPWORDS]
        # Lemmatized context is preferred over stemmed context.
        if lemma == True:
            signature = [lemmatize(i) for i in signature]
        # Matching exact words may cause sparsity, so optional matching for stems.
        if stem == True:
            signature = [porter.stem(i) for i in signature]
        synsets_signatures[ss] = signature

    return synsets_signatures

def simple_lesk(context_sentence, ambiguous_word, \
                pos=None, lemma=True, stem=False, hyperhypo=True, \
                stop=True, context_is_lemmatized=False, \
                nbest=False, keepscore=False, normalizescore=False):
 
    # Ensure that ambiguous word is a lemma.
    ambiguous_word = lemmatize(ambiguous_word)
    # If ambiguous word not in WordNet return None
    if not wn.synsets(ambiguous_word):
        return None
    # Get the signatures for each synset.
    ss_sign = simple_signature(ambiguous_word, pos, lemma, stem, hyperhypo)
    # Disambiguate the sense in context.
    if context_is_lemmatized:
        context_sentence = context_sentence.split()
    else:
        context_sentence = lemmatize_sentence(context_sentence)
    best_sense = compare_overlaps(context_sentence, ss_sign, \
                                    nbest=nbest, keepscore=keepscore, \
                                    normalizescore=normalizescore)
    return best_sense

def semanticSimilarity(word,pos):
#	print word,pos

	if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS':
		pos = 'n'
	elif pos == 'JJ' or pos == 'JJR' or pos == 'JJS':
		pos = 'a'
	elif pos == 'VB' or pos ==  'VBD' or pos == 'VBG' or pos == 'VBN' or pos ==  'VBP' or pos ==   'VBZ':
		pos = 'v'
	elif pos == 'RB'or pos ==  'RBR' or pos ==  'RBS':
		pos = 'r'
	#print "#TESTING simple_lesk() with nbest results and scores"
#	print word,pos
	#related_words = []
	answer = simple_lesk(content,word,pos, True, \
                     nbest=True, keepscore=True)
	if answer == [] or answer == None:
		return '-'
	else:
		#print "Senses ranked by #overlaps:", answer
		best_sense = answer[0][1].name()
		
		#best_sense = best_sense.name()	
		
		#print best_sense
		related_words = (wn.synset(str(best_sense)).lemma_names())
		
		#print related_words
		#list(related_words.decode("utf-8"))
		#definition = best_sense.definition()
		#except: definition = best_sense.definition
		#print "Definition:", definition
#		print
		return related_words

sent = open('answers.txt', 'rb').readlines()#creates a list in which each element is one line of the file
#print 'sent'
#print sent

replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

for i in xrange(len(sent)):
	sent[i]=sent[i].lower()
	sent[i] = sent[i].translate(replace_punctuation)

def freq(word, doc):#finds freq of a word in a sentence
    
    return doc.count(word)

def word_count(doc):#finds the number of words in a sentence
    return len(doc)
    
def tf(listt, doc):#finds the normalised freq of a word in a sentence 
    return (listfreq(listt, doc) / float(word_count(doc)))
    
def num_sentences_containing(listt,sent):#finds the number of sentences that conatain a particular word
    count = 0
    for i in sent:
        if listfreq(listt, i) > 0:
        	count += 1
#		print 'number of sentences in database containing: ',
#		print listt
#  		print count
    return count
    
def idf(listt, sent):
	#print listt
	#print num_sentences_containing(listt,sent)
	return math.log(len(sent)/float(num_sentences_containing(listt,sent)))
    
            
def tf_idf(listt, doc, sent):#find tfidf value of a word in a sentence
    return (tf(listt, doc) * idf(listt, sent))	

def cosSimilarity(listt,content,sent):
	dot=0;
 	sentt=0;
 	docc=0;
 	senttroot=0;
 	doccroot=0;
 	for w in sent:
 		dot=dot+(tf_idf(listt, content, sent)* idf(listt, sent));
 		sentt=sentt+(tf_idf(listt, content, sent))**2;
 		docc=docc+(idf(listt, sent))**2;
 	senttroot=(sentt)**(1/2);
 	doccroot=(docc)**(1/2);
 	return dot/(senttroot*doccroot);

def listfreq(listt,doc):
#	print 'list sent to function'
#	print listt
#	print 'sentence sent to function'
#	print doc
	summ=0
	count=0
	for word in listt:
		count=doc.count(word)
#		print 'freq of'+word+'is:'
#		print count
		summ=summ+count;
#	print 'sum'
#	print summ
	return summ
	
con = sqlite3.connect('table_new1.db') #connection to connect to database
con.text_factory = str
cur = con.cursor() #variable to execute statements 
cur.execute('DROP TABLE IF EXISTS OriginalAnswers;') #Deletes any other existing table of the same name to avoid overwriting of tables

cur.execute('CREATE TABLE OriginalAnswers(Words TEXT , Q_Num INT , Ans_Num INT UNSIGNED , POS TEXT DEFAULT NULL, FreqTf INT ,Dft FLOAT,NormalisedTf FLOAT,IDft FLOAT,TfIdf FLOAT, Similar STRING, cosSim FLOAT);') #creates table with given coloumns

i,x,count = 1,0,1
with open('answers.txt') as f:	#opens the file in ''
	content = f.readline()		#reading the first line of the text file
	#content = conten.translate(conten.maketrans(""," "), string.punctuation)
	
	while content:	#reads every line in the file 
		#print content
		content=content.lower()
		
		content = content.translate(replace_punctuation) #removing all punctuations from a sentence 
		#content = re.sub('[.]+', ' ', content)
		#content = re.sub('[,]+', ' ', content)
		#content = re.sub('[?]+', ' ', content)
		#content = re.sub('[/]+', ' ', content)
		#content = re.sub('[:]+', ' ', content)
		#content = re.sub('[;]+', ' ', content)
		#content = re.sub('[\']+', ' ', content)
		#content = re.sub('[\"]+', ' ', content)
		
		stop_words = set(stopwords.words("english")) #setting stop_words set as ENGLISH LANGUAGE.

		words = word_tokenize(content)	#tokenizing the words in the example
		
		filtered_sentence = []
		
		k = dict(Counter(words))
		#print k
		a = nltk.pos_tag(words)
		#print a
		x=0
		similar=[]
		#print content
		for w in words:
			if w in filtered_sentence:
				count +=1
			#	continue
			if w in stop_words:
				x += 1
			#	continue	
			elif w not in stop_words:	#removing stop words
				filtered_sentence.append(w)	#tokenized stop word free list
				c=tff=idff=t=cs=0.0
		#		print w ,a[x][1]
				similar = semanticSimilarity(w,a[x][1])
				similarr= ', '.join(similar)

# ---------> HERE COMES A LIST OF SIMILAR WORDS (similar) UPDATE THE TABLE WITH ALL THE ELEMENTS OF THE LIST	----------------------------------------
#																																					   !
				
				cur.execute('INSERT INTO OriginalAnswers(Words , Q_Num , Ans_Num , POS , FreqTf , Dft, NormalisedTf, IDft, TfIdf, Similar, cosSim) VALUES (?,?,?,?,?,?,?,?,?,?,?);',(w,1,i,a[x][1],k[w],c,tff,idff,t,similarr,cs))
					
				cur.execute('select Similar from OriginalAnswers where Words=?;',(w,))
				synonyms=cur.fetchall()
				
				listt=synonyms[0][0].split(', ')
				for z in xrange(len(listt)):		#putting all words in list of the form [' word1 ',' word2 ']
					listt[z]=" "+listt[z]+" "	
				wor=" "+w+" "
				if wor not in listt:
					listt.append(wor)
#				print 'listt'
#				print listt
				
				fre=listfreq(listt,content)
#				print 'freq of list words in sentence:'
#				print fre
				
#				print 'listt after goin into function'
#				print listt
#				print '************************************************'
				
				tff=tf(listt, content)
				c=num_sentences_containing(listt,sent)
				idff=idf(listt, sent)
				t=tf_idf(listt, content, sent)
				#cs=cosSimilarity(listt,content,sent)
				
				cur.execute('UPDATE OriginalAnswers SET FreqTf=? , Dft=?, NormalisedTf=?, IDft=?, TfIdf=?, cosSim=? where Words=?;',(fre,c,tff,idff,t,cs,w))
				
				x +=1 
				count = 1
				con.commit()				
		
		
		i += 1	
		count = 1
				
		content = f.readline() #increments to next line in text file+------
		
		con.commit()  

data = cur.fetchall()
con.close()
