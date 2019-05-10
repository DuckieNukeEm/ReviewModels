import pandas as pd
import re
<<<<<<< HEAD
import gzip
import numpy as np
from multiprocessing import cpu_count, Pool #,parallel

from nltk import  pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from datetime import datetime

def parallelize(data, func, use_cores = None):
	"""appplies a function (func) across the a data frame (data), but parralizing
	it across the number of cores use_cores"""
	if use_cores is None:
		use_cores = cpu_count()

	data_split = np.array_split(data, use_cores)
	pool = Pool(use_cores)
	data = pd.concat(pool.map(func, data_split))
	pool.close()
	pool.join()
	return(data)

def parse(PATH: str):
	"""returns an itterable for each element in the raw gz.json data file"""
	g = gzip.open(PATH, 'r')
	for l in g:
		yield eval(l)

def getDF(PATH: str, SAVE_PATH: str, chunk_size: int = 100000, headers:list = [],fix_count: bool = False, fix_cat:bool = False):
	"""reads in chunks of the json file and flattens them out into a usable format"""
	# TODO Add ReviewID Field
	i = 0
	big_count = 0
	first_pass = True
	df = {}

	for d in parse(PATH):
		df[i] = d
		i += 1
		if i % chunk_size == 0:
			big_count += i
			print('Just finished record %i' % big_count)
			df_out = pd.DataFrame.from_dict(df,  orient = 'index',columns = headers )[headers]
			if fix_count:
				df_out[['good_count','bad_count']] = pd.DataFrame( df_out.helpful.astype(str).apply(lambda x: re.sub('\[|\]|\ ','',x)).str.split(',').values.tolist()) 
				df_out.drop(columns = ['helpful'], inplace = True)
#			if fix_cat:
#				True
#				df_out['categories'] = df.categories.apply(lambda x: re.sub('\[\[|\]\]','',x)
			if first_pass:
				df_out.to_csv(SAVE_PATH, header = True , sep = '|', index = False)
				first_pass = False
			else:
				with open(SAVE_PATH, 'a') as f:
					df_out.to_csv(f, header = False, sep = '|', index = False)
			i = 0
			df = {}
=======
import Basic_NLP_Func as bnf
import ReadWrite as RW

>>>>>>> d29e2fd45f0c7271d4218ae06bca35a8559a6e33

def simple_tagger(tag):
	""" takes a tag and simpleifies it down to four tags (or five) tags
	https://simonhessner.de/lemmatize-whole-sentences-with-python-and-nltks-wordnetlemmatizer/"""
	if tag[0] == 'J': #adjective
		return('a')
	elif tag[0] == 'V': # Verb
		return('v')
	elif tag[0] == 'N': #Noun
		return('n')
	elif tag[0] == 'R': #Ad Verb
		return('r')
	else:
		return(None)

def split_parse_join(sent: str, lemma):
	"""takes a string, splits it, lemmatizies it, and then joins it back again"""
	nltk_tagged = pos_tag(word_tokenize(sent.lower()))
	simple_tagged = map(lambda x: (x[0], simple_tagger(x[1])), nltk_tagged)

	out_list = [word if tag is None else lemma.lemmatize(word,tag) for  word, tag in simple_tagged]
	return(" ".join(out_list))

def chunk_spj(load_path, save_path, chunksize = 10000, multi_core = False, itter_steps = 4):
	"""takes a pandas itterable and then itterates through it, applies split_parse_join to each itteratable, and then saves it down to the disk"""
	df_itter = pd.read_csv(load_path, sep = '|', chunksize = chunksize)

	i = 0
	for chunk in df_itter:
		print('Working on Chunk %i' % i)
		if multi_core:
			df_chunk = parallelize(chunk, df_substep)
		else:
			df_chunk = df_substep(df)

		if i == 0:
			df_chunk.to_csv(save_path, index = False)
		else:
			with open(save_path, 'a') as f:
				df_chunk.to_csv(f, index = False, header = False)
		i += 1
		if itter_steps <= i:
			break


def df_substep(df):

	lemma = WordNetLemmatizer()

	df = df.reviewText.copy()
	df = df.astype(str).apply(lambda x: split_parse_join(x, lemma))
	df = df.apply(lambda x: re.sub('\|','',x))
	return(df)

Start = datetime.now()

if False: #chagnge to True to unzip the user_dedup.json.gz or metadata.json.gz

	if True:
		PATH = '~/R/Data/ARD/user_dedup.json.gz'
		SAVE_PATH = '~/R/Data/ARD/amazon_review_data.txt'
		HEADERS = ['reviewerID', 'asin', 'reviewerName', 'unixReviewTime', 'reviewText','overall', 'reviewTime', 'summary','helpful']
		CHUNKSIZE = 5000000
		FIX_COUNT = True
		FIX_CAT = False
	else:
		PATH =      '~/R/Data/ARD/metadata.json.gz'
		SAVE_PATH = '~/R/Data/ARD/product_data.txt'
		HEADERS = ['asin','categories','salesRank','title','imUrl','price','description','brand','related']
		CHUNKSIZE = 9400000
		FIX_COUNT = False
		FIX_CAT = True

	getDF(PATH, SAVE_PATH, chunk_size = CHUNKSIZE, headers = HEADERS, fix_count = FIX_COUNT, fix_cat = FIX_CAT)

if True: # change to True if you want to lowercase, split, lemmatize, then join sentences

<<<<<<< HEAD
	PATH = '~/R/Data/ARD/amazon_review_data.txt'
	SAVE_PATH = '~/R/Data/ARD/ARD_clean_reviews.txt'
	CHUNKSIZE = 1000000
	ITTER_STEPS = 20
=======
def chunk_tfidf(df: df.series, tf_dict: dict = {}, idf_dict: dict = {}) -> dict:
	"""calculates tf_idf on the series"""
	# TODO figure out how 
>>>>>>> d29e2fd45f0c7271d4218ae06bca35a8559a6e33

	chunk_spj(PATH, SAVE_PATH, CHUNKSIZE, multi_core = True, itter_steps = ITTER_STEPS)

End = datetime.now()

Delta = End - Start

<<<<<<< HEAD
print(Delta)
=======
#RW.getDF(PATH, SAVE_PATH, chunk_size = 1000000)
>>>>>>> d29e2fd45f0c7271d4218ae06bca35a8559a6e33
