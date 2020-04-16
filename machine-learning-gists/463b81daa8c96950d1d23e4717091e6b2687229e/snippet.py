#from my repo 
#https://github.com/AdityaSoni19031997/Machine-Learning/blob/master/AV/AV_Enigma_NLP_functional_api.ipynb

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_tweet.append(word)

    return ' '.join(processed_tweet)
 
def preprocessing_text(s):
    import re
    s = re.sub(r"[^A-Za-z0-9^,\*+-=]", " ",s)
    s = re.sub(r"(\d+)(k)", r"\g<1>000", s) #expand 'k' to '000' eg. 50k to 50000
    s = re.sub(r"\;"," ",s)
    s = re.sub(r"\:"," ",s)
    s = re.sub(r"\,"," ",s)
    s = re.sub(r"\."," ",s)
    s = re.sub(r"\<"," ",s)
    s = re.sub(r"\^"," ",s)
    s = re.sub(r"(\d+)(/)", "\g<1> divide ", s) #change number/number to number divide number (eg. 2/3 to 2 divide 3)
    s = re.sub(r"\/"," ",s) #replace the rest of / with white space
    s = re.sub(r"\+", " plus ", s)
    s = re.sub(r"\-", " minus ", s)
    s = re.sub(r"\*", " multiply ", s)
    s = re.sub(r"\=", "equal", s)
    s = re.sub(r"What's", "What is ", s)
    s = re.sub(r"what's", "what is ", s)
    s = re.sub(r"Who's", "Who is ", s)
    s = re.sub(r"who's", "who is ", s)
    s = re.sub(r"\'s", " ", s)
    s = re.sub(r"\'ve", " have ", s)
    s = re.sub(r"can't", "cannot ", s)
    s = re.sub(r"n't", " not ", s)
    s = re.sub(r"\'re", " are ", s)
    s = re.sub(r"\'d", " would ", s)
    s = re.sub(r"\'ll", " will ", s)
    s = re.sub(r"'m", " am ", s)
    s = re.sub(r"or not", " ", s)
    s = re.sub(r"What should I do to", "How can I", s)
    s = re.sub(r"How do I", "How can I", s)
    s = re.sub(r"How can you make", "What can make", s)
    s = re.sub(r"How do we", "How do I", s)
    s = re.sub(r"How do you", "How do I", s)
    s = re.sub(r"Is it possible", "Can we", s)
    s = re.sub(r"Why is", "Why", s)
    s = re.sub(r"Which are", "What are", s)
    s = re.sub(r"What are the reasons", "Why", s)
    s = re.sub(r"What are some tips", "tips", s)
    s = re.sub(r"What is the best way", "best way", s)
    s = re.sub(r"e-mail", "email", s)
    s = re.sub(r"e - mail", "email", s)
    s = re.sub(r"US", "America", s)
    s = re.sub(r"USA", "America", s)
    s = re.sub(r"us", "America", s)
    s = re.sub(r"usa", "America", s)
    s = re.sub(r"Chinese", "China", s)
    s = re.sub(r"india", "India", s)
    s = re.sub(r"\s{2,}", " ", s) #remove extra white space
    s = s.strip()
    return s

def remove_stopwords(string):
    word_list = [word.lower() for word in string.split()]
    from nltk.corpus import stopwords
    stopwords_list = list(stopwords.words("english"))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    return ' '.join(word_list)

def get_char_length_ratio(row):
    return len(row['tweet'])/max(1,len(row['tweet_without_stopwords']))

def get_synonyms(word):
    from nltk.corpus import wordnet as wn
    synonyms = []
    if wn.synsets(word):
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))

def get_row_syn_set(row):
    import nltk
    syn_set = [nltk.word_tokenize(row)]
    for token in nltk.word_tokenize(row):
        if get_synonyms(token):
            syn_set.append(get_synonyms(token))
    return set([y for x in syn_set for y in x])

def get_Levenshtein(string1,string2):
    import editdistance
    return editdistance.eval(string1,string2)

def num_pos(sent):
    num_pos = 0
    word_list = [word.lower() for word in nltk.word_tokenize(sent)]
    for index, word in enumerate(word_list):
        if word in positive_words:
            if word_list[index-1] not in ['not','no']:
                num_pos += 1
    return num_pos

def num_neg(sent):
    num_neg = 0
    word_list = [word.lower() for word in nltk.word_tokenize(sent)]
    for index, word in enumerate(word_list):
        if word in negative_words:
            if word_list[index-1] not in ['not','no']:
                num_neg += 1
    return num_neg

p_url = 'http://ptrckprry.com/course/ssd/data/positive-words.txt'
n_url = 'http://ptrckprry.com/course/ssd/data/negative-words.txt'

import requests,nltk
positive_words = requests.get(p_url).content.decode('latin-1')
positive_words = nltk.word_tokenize(positive_words)
positive_words.remove('not')
negative_words = requests.get(n_url).content.decode('latin-1')
negative_words = nltk.word_tokenize(negative_words)
positive_words = positive_words[413:]
negative_words = negative_words[418:]