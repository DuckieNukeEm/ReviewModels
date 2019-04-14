import pandas as pd
import re
import gzip
import Basic_NLP_Func as bnf

def parse(PATH: str):
	"""returns an itterable for each element in the raw data file"""
	g = gzip.open(PATH, 'r')
	for l in g:
		yield eval(l)

def getDF(PATH: str, SAVE_PATH: str, chunk_size: int = 100000):
	"""reads in chunks of the json file and flattens them out into a usable format"""
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
			df_out = pd.DataFrame.from_dict(df,  orient = 'index')
			df_out[['good_count','bad_count']] = pd.DataFrame( df_out.helpful.astype(str).apply(lambda x: re.sub('\[|\]|\ ','',x)).str.split(',').values.tolist()) 
			df_out.drop(columns = ['helpful'], inplace = True)
			if first_pass:
				df_out.to_csv(SAVE_PATH, header = True , sep = '|', index = False)
				first_pass = False
			else:
				with open(SAVE_PATH, 'a') as f:
					df_out.to_csv(f, header = False, sep = '|', index = False)
			i = 0
			df = {}


def parse_n_split(df: df.series,
		remove_punc: bool = True,
		lowercase: bool = True,
		expand_contractions: bool = True):
	"""takes a string and splits  it up, lower cases it, and expandes contraction"""
	# TODO get the expand contraction part owrking

	col_name = df.name
	df = bnf.split_into_sentences(df)

	df[col_name] = df[col_name].astype(str).apply(lambda x: bnf.tokenize(x))

	if remove_punc:
		punc_str =  bnf.get_punctuation_string()
		df[col_name] = df[col_name].apply(lambda x: [re.sub(punc,'', y) for y in x])
	if lowercase 
	

def chunk_tfidf(df: df.series, tf_dict: dict = {}, idf_dict: dict = {}) -> dict:
	"""calculates tf_idf on the series"""



PATH = '/home/beltain/R/Data/ARD/user_dedup.json.gz'
SAVE_PATH = '/home/beltain/R/Data/ARD/amazon_review_data.txt'

getDF(PATH, SAVE_PATH, chunk_size = 1000000)
