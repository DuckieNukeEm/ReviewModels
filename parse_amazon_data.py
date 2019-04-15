import pandas as pd
import re
import Basic_NLP_Func as bnf
import ReadWrite as RW



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
	# TODO figure out how 



PATH = '/home/beltain/R/Data/ARD/user_dedup.json.gz'
SAVE_PATH = '/home/beltain/R/Data/ARD/amazon_review_data.txt'

#RW.getDF(PATH, SAVE_PATH, chunk_size = 1000000)
