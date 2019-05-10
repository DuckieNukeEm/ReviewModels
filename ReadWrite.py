import gzip
import pandas as pd

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

def load_data(PATH: str, nrow: int = 100000, chunk_size: int = None ) -> pd.DataFrame:
	"""reads in a chunk of pandas data frame"""
	# TODO get chunk_Size to work
	
	df = pd.read_csv(PATH, sep = '|', nrows = nrow)
	return(df)
	
def save_model():
	"""saves the data saved model"""

