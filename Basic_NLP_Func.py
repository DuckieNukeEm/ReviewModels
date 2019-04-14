import re
import pandas as pd


def get_punctuation_string():
	return("""\!|\#|\$|\%|\&|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\]|\^|\_|\`|\{|\||\}|\~""")

def ranks_stopwords() -> list:
    	"""returns a list of common stop words
    	https://www.ranks.nl/stopwords"""
	#todo have it read in a list of stop words
	slist = ['a','A','about','above','after','again','against','all','am','an','and','any','are',"aren't","as","at",'be','because','been','before','being','below','between','both','but','by',"can't",
'cannot','could',"couldn't",'did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll',
'he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','I','i','i\'d','i\'ll','i\'m','i\'ve','I\'d','I\'ll','I\'m','I\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let\'s','me','more','most','mustn\'t',
'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','oursourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than',
'that','that\'s','the','their','theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we',
'we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve',
'your','yours','yourself','yourselves']

	return(slist)


def pivot_lists(df: pd.Series, col_to_pivot: str = None) -> pd.DataFrame:
        """a function that takes a series of lists, and then pivots it out so each element of the list is in
        it's own row. """
        if col_to_pivot == None:
                col_to_pivot = df.name
        df = df.to_frame(name = df.name).reset_index().query("index != 'index'").copy()

        #https://stackoverflow.com/questions/42012152/unstack-a-pandas-column-containing-lists-into-multiple-rows
        df = pd.DataFrame({
                col:np.repeat(df[col].values, df[col_to_pivot].str.len())
                for col in df.columns.difference([col_to_pivot])
                }).assign(**{col_to_pivot:np.concatenate(df[col_to_pivot].values)})[df.columns.tolist()]

        df['Sentence_Number'] = df.groupby(['index']).cumcount()
        return(df)

def split_into_sentences(df: pd.Series) -> pd.DataFrame:
	col_name = df.name
	df = df.astype(str).apply(lambda x: regex_sentences(x))
	df = pivot_lists(df)
	df[col_name] = df[col_name].astype(str)
	return(df)


def get_decontractions() -> dict:
   	 """removes contractions from a words"""
    # https://gist.github.com/nealrs/96342d8231b75cf4bb82
    # TODO rebuild list queing in on the 'd for would  't for not etc.
    # 'll -> will
    # 've -> have
    # 'd -. (have, had)
    # 't -> not
	clist1 = {
		"n't": "not",
	#	"n't've": "not have",
		"'ll" : "will",
	#	"'ll've": "will have",
		"'ve" : "have",
		"'d": "would",
	#	"'d've" : "would have",
		"'re" : "are",
		"y'all" : "you all"}

	clist2 = { "'cause": "because", #
		"he's": "he is",
        "how'd": "how did", #
        "how'd'y": "how do you", #
        "how's": "how is",
        "i'll've": "I will have",
        "i'm": "I am",
        "I'm": "I am",
        "it's": "it is",
        "It's": "It is",
        "let's": "let us",
        "ma'am": "madam",
        "o'clock": "of the clock",
        "she's": "she is",
        "he's" : "he is",
        "so's": "so is",
        "there's": "there is",
        "what's": "what is",
        "when's": "when is",
        "where's": "where is",
        "who's": "who is",
        "why's": "why is",
        "y'alls": "you alls"}

	c_re =  re.compile('(%s)' % '|'.join(['|'.join(x) for  x in [clist2, clist1]]))
	#
	#
	# Still need toget this function and the one below it working
	#
	#
	return([clist2,clist1])



def expandContractions(text, contraction_dict: list):
	"""Takes a list of dictionarys and then creates """
    c_re = re.compile('(%s)' % '|'.join(['|'.join(x.keys()) for x in contraction_list]))
    def replace(match):
        return contraction_dict[match.group(0)]
    return c_re.sub(replace, text)



def regex_sentences(text, split_sentence: bool = True):
	"""Takes a text, and splits it into sentences, takes care of the issue of names 'Mr. Anderson' as well as
	Web address and all the weird stuff
	https://stackoverflow.com/questions/4576077/python-split-text-on-sentences"""

	alphabets= "([A-Za-z])"
	prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
	suffixes = "(Inc|Ltd|Jr|Sr|Co)"
	starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
	acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
	websites = "[.](com|net|org|io|gov)"

	text = " " + text + "  "
	text = text.replace("\n"," ")

	# taking care of the prefexes and websites
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)

	#replacing ???,...., !!!!! and '     ' with a single instance of it
	text = re.sub('\?\.|\.\?|\!\?|\?\!|\!\.|\.\!','.',text)
	text = re.sub('\?{2,','?',text)
	text = re.sub('\.{3,}',' ',text)
	text = re.sub('\.{2}','.',text)
	text = re.sub('\!{2,}','!',text)
	text = re.sub(' {2,}',' ',text)

	#PHD LEvel shit

	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	#Starters and suffixes
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	#deliminting the questions and stop points
	text = text.replace(".",".<stop>")
	text = text.replace("?","?<stop>")
	text = text.replace("!","!<stop>")
	text = text.replace("<prd>",".")
	if split_sentence:
		sentences = text.split("<stop>")
		sentences = sentences[:-1]
		sentences = [s.strip() for s in sentences]
		return sentences
	else:
		return(re.sub('<stop>', '', text).strip())
