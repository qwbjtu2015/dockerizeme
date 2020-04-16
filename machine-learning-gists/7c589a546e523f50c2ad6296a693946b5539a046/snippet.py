import os
import codecs

data_directory = os.path.join('..', 'data',
                              'yelp_dataset_challenge_academic_dataset')

businesses_filepath = os.path.join(data_directory,
                                   'yelp_academic_dataset_business.json')

with codecs.open(businesses_filepath, encoding='utf_8') as f:
    first_business_record = f.readline() 

print first_business_record


# The business records consist of _key, value_ pairs containing information about the particular business. A few attributes we'll be interested in for this demo include:
# - __business\_id__ &mdash; _unique identifier for businesses_
# - __categories__ &mdash; _an array containing relevant category values of businesses_
# 
# The _categories_ attribute is of special interest. This demo will focus on restaurants, which are indicated by the presence of the _Restaurant_ tag in the _categories_ array. In addition, the _categories_ array may contain more detailed information about restaurants, such as the type of food they serve.

# The review records are stored in a similar manner &mdash; _key, value_ pairs containing information about the reviews.

# In[2]:


review_json_filepath = os.path.join(data_directory,
                                    'yelp_academic_dataset_review.json')

with codecs.open(review_json_filepath, encoding='utf_8') as f:
    first_review_record = f.readline()
    
print first_review_record


# A few attributes of note on the review records:
# - __business\_id__ &mdash; _indicates which business the review is about_
# - __text__ &mdash; _the natural language text the user wrote_
# 
# The _text_ attribute will be our focus today!

# _json_ is a handy file format for data interchange, but it's typically not the most usable for any sort of modeling work. Let's do a bit more data preparation to get our data in a more usable format. Our next code block will do the following:
# 1. Read in each business record and convert it to a Python `dict`
# 2. Filter out business records that aren't about restaurants (i.e., not in the "Restaurant" category)
# 3. Create a `frozenset` of the business IDs for restaurants, which we'll use in the next step

# In[3]:


import json

restaurant_ids = set()

# open the businesses file
with codecs.open(businesses_filepath, encoding='utf_8') as f:
    
    # iterate through each line (json record) in the file
    for business_json in f:
        
        # convert the json record to a Python dict
        business = json.loads(business_json)
        
        # if this business is not a restaurant, skip to the next one
        if u'Restaurants' not in business[u'categories']:
            continue
            
        # add the restaurant business id to our restaurant_ids set
        restaurant_ids.add(business[u'business_id'])

# turn restaurant_ids into a frozenset, as we don't need to change it anymore
restaurant_ids = frozenset(restaurant_ids)

# print the number of unique restaurant ids in the dataset
print '{:,}'.format(len(restaurant_ids)), u'restaurants in the dataset.'


# Next, we will create a new file that contains only the text from reviews about restaurants, with one review per line in the file.

# In[4]:


intermediate_directory = os.path.join('..', 'intermediate')

review_txt_filepath = os.path.join(intermediate_directory,
                                   'review_text_all.txt')


# In[5]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 1:\n    \n    review_count = 0\n\n    # create & open a new file in write mode\n    with codecs.open(review_txt_filepath, 'w', encoding='utf_8') as review_txt_file:\n\n        # open the existing review json file\n        with codecs.open(review_json_filepath, encoding='utf_8') as review_json_file:\n\n            # loop through all reviews in the existing file and convert to dict\n            for review_json in review_json_file:\n                review = json.loads(review_json)\n\n                # if this review is not about a restaurant, skip to the next one\n                if review[u'business_id'] not in restaurant_ids:\n                    continue\n\n                # write the restaurant review as a line in the new file\n                # escape newline characters in the original review text\n                review_txt_file.write(review[u'text'].replace('\\n', '\\\\n') + '\\n')\n                review_count += 1\n\n    print u'''Text from {:,} restaurant reviews\n              written to the new txt file.'''.format(review_count)\n    \nelse:\n    \n    with codecs.open(review_txt_filepath, encoding='utf_8') as review_txt_file:\n        for review_count, line in enumerate(review_txt_file):\n            pass\n        \n    print u'Text from {:,} restaurant reviews in the txt file.'.format(review_count + 1)")


# ## spaCy &mdash; Industrial-Strength NLP in Python

# ![spaCy](https://s3.amazonaws.com/skipgram-images/spaCy.png)

# [**spaCy**](https://spacy.io) is an industrial-strength natural language processing (_NLP_) library for Python. spaCy's goal is to take recent advancements in natural language processing out of research papers and put them in the hands of users to build production software.
# 
# spaCy handles many tasks commonly associated with building an end-to-end natural language processing pipeline:
# - Tokenization
# - Text normalization, such as lowercasing, stemming/lemmatization
# - Part-of-speech tagging
# - Syntactic dependency parsing
# - Sentence boundary detection
# - Named entity recognition and annotation
# 
# In the "batteries included" Python tradition, spaCy contains built-in data and models which you can use out-of-the-box for processing general-purpose English language text:
# - Large English vocabulary, including stopword lists
# - Token "probabilities"
# - Word vectors
# 
# spaCy is written in optimized Cython, which means it's _fast_. According to a few independent sources, it's the fastest syntactic parser available in any language. Key pieces of the spaCy parsing pipeline are written in pure C, enabling efficient multithreading (i.e., spaCy can release the _GIL_).

# In[6]:


import spacy
import pandas as pd
import itertools as it

nlp = spacy.load('en')


# Let's grab a sample review to play with.

# In[7]:


with codecs.open(review_txt_filepath, encoding='utf_8') as f:
    sample_review = list(it.islice(f, 8, 9))[0]
    sample_review = sample_review.replace('\\n', '\n')
        
print sample_review


# Hand the review text to spaCy, and be prepared to wait...

# In[8]:


get_ipython().run_cell_magic('time', '', 'parsed_review = nlp(sample_review)')


# ...1/20th of a second or so. Let's take a look at what we got during that time...

# In[9]:


print parsed_review


# Looks the same! What happened under the hood?
# 
# What about sentence detection and segmentation?

# In[10]:


for num, sentence in enumerate(parsed_review.sents):
    print 'Sentence {}:'.format(num + 1)
    print sentence
    print ''


# What about named entity detection?

# In[11]:


for num, entity in enumerate(parsed_review.ents):
    print 'Entity {}:'.format(num + 1), entity, '-', entity.label_
    print ''


# What about part of speech tagging?

# In[12]:


token_text = [token.orth_ for token in parsed_review]
token_pos = [token.pos_ for token in parsed_review]

pd.DataFrame(zip(token_text, token_pos),
             columns=['token_text', 'part_of_speech'])


# What about text normalization, like stemming/lemmatization and shape analysis?

# In[13]:


token_lemma = [token.lemma_ for token in parsed_review]
token_shape = [token.shape_ for token in parsed_review]

pd.DataFrame(zip(token_text, token_lemma, token_shape),
             columns=['token_text', 'token_lemma', 'token_shape'])


# What about token-level entity analysis?

# In[14]:


token_entity_type = [token.ent_type_ for token in parsed_review]
token_entity_iob = [token.ent_iob_ for token in parsed_review]

pd.DataFrame(zip(token_text, token_entity_type, token_entity_iob),
             columns=['token_text', 'entity_type', 'inside_outside_begin'])


# What about a variety of other token-level attributes, such as the relative frequency of tokens, and whether or not a token matches any of these categories?
# - stopword
# - punctuation
# - whitespace
# - represents a number
# - whether or not the token is included in spaCy's default vocabulary?

# In[15]:


token_attributes = [(token.orth_,
                     token.prob,
                     token.is_stop,
                     token.is_punct,
                     token.is_space,
                     token.like_num,
                     token.is_oov)
                    for token in parsed_review]

df = pd.DataFrame(token_attributes,
                  columns=['text',
                           'log_probability',
                           'stop?',
                           'punctuation?',
                           'whitespace?',
                           'number?',
                           'out of vocab.?'])

df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?']
                                       .applymap(lambda x: u'Yes' if x else u''))
                                               
df


# If the text you'd like to process is general-purpose English language text (i.e., not domain-specific, like medical literature), spaCy is ready to use out-of-the-box.
# 
# I think it will eventually become a core part of the Python data science ecosystem &mdash; it will do for natural language computing what other great libraries have done for numerical computing.

# ## Phrase Modeling

# _Phrase modeling_ is another approach to learning combinations of tokens that together represent meaningful multi-word concepts. We can develop phrase models by looping over the the words in our reviews and looking for words that _co-occur_ (i.e., appear one after another) together much more frequently than you would expect them to by random chance. The formula our phrase models will use to determine whether two tokens $A$ and $B$ constitute a phrase is:
# 
# $$\frac{count(A\ B) - count_{min}}{count(A) * count(B)} * N > threshold$$
# 
# ...where:
# * $count(A)$ is the number of times token $A$ appears in the corpus
# * $count(B)$ is the number of times token $B$ appears in the corpus
# * $count(A\ B)$ is the number of times the tokens $A\ B$ appear in the corpus *in order*
# * $N$ is the total size of the corpus vocabulary
# * $count_{min}$ is a user-defined parameter to ensure that accepted phrases occur a minimum number of times
# * $threshold$ is a user-defined parameter to control how strong of a relationship between two tokens the model requires before accepting them as a phrase
# 
# Once our phrase model has been trained on our corpus, we can apply it to new text. When our model encounters two tokens in new text that identifies as a phrase, it will merge the two into a single new token.
# 
# Phrase modeling is superficially similar to named entity detection in that you would expect named entities to become phrases in the model (so _new york_ would become *new\_york*). But you would also expect multi-word expressions that represent common concepts, but aren't specifically named entities (such as _happy hour_) to also become phrases in the model.
# 
# We turn to the indispensible [**gensim**](https://radimrehurek.com/gensim/index.html) library to help us with phrase modeling &mdash; the [**Phrases**](https://radimrehurek.com/gensim/models/phrases.html) class in particular.

# In[16]:


from gensim.models import Phrases
from gensim.models.word2vec import LineSentence


# As we're performing phrase modeling, we'll be doing some iterative data transformation at the same time. Our roadmap for data preparation includes:
# 
# 1. Segment text of complete reviews into sentences & normalize text
# 1. First-order phrase modeling $\rightarrow$ _apply first-order phrase model to transform sentences_
# 1. Second-order phrase modeling $\rightarrow$ _apply second-order phrase model to transform sentences_
# 1. Apply text normalization and second-order phrase model to text of complete reviews
# 
# We'll use this transformed data as the input for some higher-level modeling approaches in the following sections.

# First, let's define a few helper functions that we'll use for text normalization. In particular, the `lemmatized_sentence_corpus` generator function will use spaCy to:
# - Iterate over the 1M reviews in the `review_txt_all.txt` we created before
# - Segment the reviews into individual sentences
# - Remove punctuation and excess whitespace
# - Lemmatize the text
# 
# ... and do so efficiently in parallel, thanks to spaCy's `nlp.pipe()` function.

# In[17]:


def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])


# In[18]:


unigram_sentences_filepath = os.path.join(intermediate_directory,
                                          'unigram_sentences_all.txt')


# Let's use the `lemmatized_sentence_corpus` generator to loop over the original review text, segmenting the reviews into individual sentences and normalizing the text. We'll write this data back out to a new file (`unigram_sentences_all`), with one normalized sentence per line. We'll use this data for learning our phrase models.

# In[19]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 1:\n\n    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:\n        for sentence in lemmatized_sentence_corpus(review_txt_filepath):\n            f.write(sentence + '\\n')")


# If your data is organized like our `unigram_sentences_all` file now is &mdash; a large text file with one document/sentence per line &mdash; gensim's [**LineSentence**](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence) class provides a convenient iterator for working with other gensim components. It *streams* the documents/sentences from disk, so that you never have to hold the entire corpus in RAM at once. This allows you to scale your modeling pipeline up to potentially very large corpora.

# In[20]:


unigram_sentences = LineSentence(unigram_sentences_filepath)


# Let's take a look at a few sample sentences in our new, transformed file.

# In[21]:


for unigram_sentence in it.islice(unigram_sentences, 230, 240):
    print u' '.join(unigram_sentence)
    print u''


# Next, we'll learn a phrase model that will link individual words into two-word phrases. We'd expect words that together represent a specific concept, like "`ice cream`", to be linked together to form a new, single token: "`ice_cream`".

# In[22]:


bigram_model_filepath = os.path.join(intermediate_directory, 'bigram_model_all')


# In[23]:


get_ipython().run_cell_magic('time', '', '\n# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\nif 0 == 1:\n\n    bigram_model = Phrases(unigram_sentences)\n\n    bigram_model.save(bigram_model_filepath)\n    \n# load the finished model from disk\nbigram_model = Phrases.load(bigram_model_filepath)')


# Now that we have a trained phrase model for word pairs, let's apply it to the review sentences data and explore the results.

# In[24]:


bigram_sentences_filepath = os.path.join(intermediate_directory,
                                         'bigram_sentences_all.txt')


# In[25]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 1:\n\n    with codecs.open(bigram_sentences_filepath, 'w', encoding='utf_8') as f:\n        \n        for unigram_sentence in unigram_sentences:\n            \n            bigram_sentence = u' '.join(bigram_model[unigram_sentence])\n            \n            f.write(bigram_sentence + '\\n')")


# In[26]:


bigram_sentences = LineSentence(bigram_sentences_filepath)


# In[27]:


for bigram_sentence in it.islice(bigram_sentences, 230, 240):
    print u' '.join(bigram_sentence)
    print u''


# Looks like the phrase modeling worked! We now see two-word phrases, such as "`ice_cream`" and "`apple_pie`", linked together in the text as a single token. Next, we'll train a _second-order_ phrase model. We'll apply the second-order phrase model on top of the already-transformed data, so that incomplete word combinations like "`vanilla_ice cream`" will become fully joined to "`vanilla_ice_cream`". No disrespect intended to [Vanilla Ice](https://www.youtube.com/watch?v=rog8ou-ZepE), of course.

# In[28]:


trigram_model_filepath = os.path.join(intermediate_directory,
                                      'trigram_model_all')


# In[29]:


get_ipython().run_cell_magic('time', '', '\n# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\nif 0 == 1:\n\n    trigram_model = Phrases(bigram_sentences)\n\n    trigram_model.save(trigram_model_filepath)\n    \n# load the finished model from disk\ntrigram_model = Phrases.load(trigram_model_filepath)')


# We'll apply our trained second-order phrase model to our first-order transformed sentences, write the results out to a new file, and explore a few of the second-order transformed sentences.

# In[30]:


trigram_sentences_filepath = os.path.join(intermediate_directory,
                                          'trigram_sentences_all.txt')


# In[31]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 1:\n\n    with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:\n        \n        for bigram_sentence in bigram_sentences:\n            \n            trigram_sentence = u' '.join(trigram_model[bigram_sentence])\n            \n            f.write(trigram_sentence + '\\n')")


# In[32]:


trigram_sentences = LineSentence(trigram_sentences_filepath)


# In[33]:


for trigram_sentence in it.islice(trigram_sentences, 230, 240):
    print u' '.join(trigram_sentence)
    print u''


# Looks like the second-order phrase model was successful. We're now seeing three-word phrases, such as "`vanilla_ice_cream`" and "`cinnamon_ice_cream`".

# The final step of our text preparation process circles back to the complete text of the reviews. We're going to run the complete text of the reviews through a pipeline that applies our text normalization and phrase models.
# 
# In addition, we'll remove stopwords at this point. _Stopwords_ are very common words, like _a_, _the_, _and_, and so on, that serve functional roles in natural language, but typically don't contribute to the overall meaning of text. Filtering stopwords is a common procedure that allows higher-level NLP modeling techniques to focus on the words that carry more semantic weight.
# 
# Finally, we'll write the transformed text out to a new file, with one review per line.

# In[34]:


trigram_reviews_filepath = os.path.join(intermediate_directory,
                                        'trigram_transformed_reviews_all.txt')


# In[35]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 1:\n\n    with codecs.open(trigram_reviews_filepath, 'w', encoding='utf_8') as f:\n        \n        for parsed_review in nlp.pipe(line_review(review_txt_filepath),\n                                      batch_size=10000, n_threads=4):\n            \n            # lemmatize the text, removing punctuation and whitespace\n            unigram_review = [token.lemma_ for token in parsed_review\n                              if not punct_space(token)]\n            \n            # apply the first-order and second-order phrase models\n            bigram_review = bigram_model[unigram_review]\n            trigram_review = trigram_model[bigram_review]\n            \n            # remove any remaining stopwords\n            trigram_review = [term for term in trigram_review\n                              if term not in spacy.en.STOPWORDS]\n            \n            # write the transformed review as a line in the new file\n            trigram_review = u' '.join(trigram_review)\n            f.write(trigram_review + '\\n')")


# Let's preview the results. We'll grab one review from the file with the original, untransformed text, grab the same review from the file with the normalized and transformed text, and compare the two.

# In[36]:


print u'Original:' + u'\n'

for review in it.islice(line_review(review_txt_filepath), 11, 12):
    print review

print u'----' + u'\n'
print u'Transformed:' + u'\n'

with codecs.open(trigram_reviews_filepath, encoding='utf_8') as f:
    for review in it.islice(f, 11, 12):
        print review


# You can see that most of the grammatical structure has been scrubbed from the text &mdash; capitalization, articles/conjunctions, punctuation, spacing, etc. However, much of the general semantic *meaning* is still present. Also, multi-word concepts such as "`friday_night`" and "`above_average`" have been joined into single tokens, as expected. The review text is now ready for higher-level modeling. 

# ## Topic Modeling with Latent Dirichlet Allocation (_LDA_)

# *Topic modeling* is family of techniques that can be used to describe and summarize the documents in a corpus according to a set of latent "topics". For this demo, we'll be using [*Latent Dirichlet Allocation*](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) or LDA, a popular approach to topic modeling.
# 
# In many conventional NLP applications, documents are represented a mixture of the individual tokens (words and phrases) they contain. In other words, a document is represented as a *vector* of token counts. There are two layers in this model &mdash; documents and tokens &mdash; and the size or dimensionality of the document vectors is the number of tokens in the corpus vocabulary. This approach has a number of disadvantages:
# * Document vectors tend to be large (one dimension for each token $\Rightarrow$ lots of dimensions)
# * They also tend to be very sparse. Any given document only contains a small fraction of all tokens in the vocabulary, so most values in the document's token vector are 0.
# * The dimensions are fully indepedent from each other &mdash; there's no sense of connection between related tokens, such as _knife_ and _fork_.
# 
# LDA injects a third layer into this conceptual model. Documents are represented as a mixture of a pre-defined number of *topics*, and the *topics* are represented as a mixture of the individual tokens in the vocabulary. The number of topics is a model hyperparameter selected by the practitioner. LDA makes a prior assumption that the (document, topic) and (topic, token) mixtures follow [*Dirichlet*](https://en.wikipedia.org/wiki/Dirichlet_distribution) probability distributions. This assumption encourages documents to consist mostly of a handful of topics, and topics to consist mostly of a modest set of the tokens.

# ![LDA](https://s3.amazonaws.com/skipgram-images/LDA.png)

# LDA is fully unsupervised. The topics are "discovered" automatically from the data by trying to maximize the likelihood of observing the documents in your corpus, given the modeling assumptions. They are expected to capture some latent structure and organization within the documents, and often have a meaningful human interpretation for people familiar with the subject material.
# 
# We'll again turn to gensim to assist with data preparation and modeling. In particular, gensim offers a high-performance parallelized implementation of LDA with its [**LdaMulticore**](https://radimrehurek.com/gensim/models/ldamulticore.html) class.

# In[37]:


from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim
import warnings
import cPickle as pickle


# The first step to creating an LDA model is to learn the full vocabulary of the corpus to be modeled. We'll use gensim's [**Dictionary**](https://radimrehurek.com/gensim/corpora/dictionary.html) class for this.

# In[38]:


trigram_dictionary_filepath = os.path.join(intermediate_directory,
                                           'trigram_dict_all.dict')


# In[39]:


get_ipython().run_cell_magic('time', '', '\n# this is a bit time consuming - make the if statement True\n# if you want to learn the dictionary yourself.\nif 0 == 1:\n\n    trigram_reviews = LineSentence(trigram_reviews_filepath)\n\n    # learn the dictionary by iterating over all of the reviews\n    trigram_dictionary = Dictionary(trigram_reviews)\n    \n    # filter tokens that are very rare or too common from\n    # the dictionary (filter_extremes) and reassign integer ids (compactify)\n    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)\n    trigram_dictionary.compactify()\n\n    trigram_dictionary.save(trigram_dictionary_filepath)\n    \n# load the finished dictionary from disk\ntrigram_dictionary = Dictionary.load(trigram_dictionary_filepath)')


# Like many NLP techniques, LDA uses a simplifying assumption known as the [*bag-of-words* model](https://en.wikipedia.org/wiki/Bag-of-words_model). In the bag-of-words model, a document is represented by the counts of distinct terms that occur within it. Additional information, such as word order, is discarded. 
# 
# Using the gensim Dictionary we learned to generate a bag-of-words representation for each review. The `trigram_bow_generator` function implements this. We'll save the resulting bag-of-words reviews as a matrix.
# 
# In the following code, "bag-of-words" is abbreviated as `bow`.

# In[40]:


trigram_bow_filepath = os.path.join(intermediate_directory,
                                    'trigram_bow_corpus_all.mm')


# In[41]:


def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    
    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)


# In[42]:


get_ipython().run_cell_magic('time', '', '\n# this is a bit time consuming - make the if statement True\n# if you want to build the bag-of-words corpus yourself.\nif 0 == 1:\n\n    # generate bag-of-words representations for\n    # all reviews and save them as a matrix\n    MmCorpus.serialize(trigram_bow_filepath,\n                       trigram_bow_generator(trigram_reviews_filepath))\n    \n# load the finished bag-of-words corpus from disk\ntrigram_bow_corpus = MmCorpus(trigram_bow_filepath)')


# With the bag-of-words corpus, we're finally ready to learn our topic model from the reviews. We simply need to pass the bag-of-words matrix and Dictionary from our previous steps to `LdaMulticore` as inputs, along with the number of topics the model should learn. For this demo, we're asking for 50 topics.

# In[43]:


lda_model_filepath = os.path.join(intermediate_directory, 'lda_model_all')


# In[44]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to train the LDA model yourself.\nif 0 == 1:\n\n    with warnings.catch_warnings():\n        warnings.simplefilter('ignore')\n        \n        # workers => sets the parallelism, and should be\n        # set to your number of physical cores minus one\n        lda = LdaMulticore(trigram_bow_corpus,\n                           num_topics=50,\n                           id2word=trigram_dictionary,\n                           workers=3)\n    \n    lda.save(lda_model_filepath)\n    \n# load the finished LDA model from disk\nlda = LdaMulticore.load(lda_model_filepath)")


# Our topic model is now trained and ready to use! Since each topic is represented as a mixture of tokens, you can manually inspect which tokens have been grouped together into which topics to try to understand the patterns the model has discovered in the data.

# In[45]:


def explore_topic(topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
        
    print u'{:20} {}'.format(u'term', u'frequency') + u'\n'

    for term, frequency in lda.show_topic(topic_number, topn=25):
        print u'{:20} {:.3f}'.format(term, round(frequency, 3))


# In[46]:


explore_topic(topic_number=0)


# The first topic has strong associations with words like *taco*, *salsa*, *chip*, *burrito*, and *margarita*, as well as a handful of more general words. You might call this the **Mexican food** topic!
# 
# It's possible to go through and inspect each topic in the same way, and try to assign a human-interpretable label that captures the essence of each one. I've given it a shot for all 50 topics below.

# In[47]:


topic_names = {0: u'mexican',
               1: u'menu',
               2: u'thai',
               3: u'steak',
               4: u'donuts & appetizers',
               5: u'specials',
               6: u'soup',
               7: u'wings, sports bar',
               8: u'foreign language',
               9: u'las vegas',
               10: u'chicken',
               11: u'aria buffet',
               12: u'noodles',
               13: u'ambience & seating',
               14: u'sushi',
               15: u'arizona',
               16: u'family',
               17: u'price',
               18: u'sweet',
               19: u'waiting',
               20: u'general',
               21: u'tapas',
               22: u'dirty',
               23: u'customer service',
               24: u'restrooms',
               25: u'chinese',
               26: u'gluten free',
               27: u'pizza',
               28: u'seafood',
               29: u'amazing',
               30: u'eat, like, know, want',
               31: u'bars',
               32: u'breakfast',
               33: u'location & time',
               34: u'italian',
               35: u'barbecue',
               36: u'arizona',
               37: u'indian',
               38: u'latin & cajun',
               39: u'burger & fries',
               40: u'vegetarian',
               41: u'lunch buffet',
               42: u'customer service',
               43: u'taco, ice cream',
               44: u'high cuisine',
               45: u'healthy',
               46: u'salad & sandwich',
               47: u'greek',
               48: u'poor experience',
               49: u'wine & dine'}


# In[48]:


topic_names_filepath = os.path.join(intermediate_directory, 'topic_names.pkl')

with open(topic_names_filepath, 'w') as f:
    pickle.dump(topic_names, f)


# You can see that, along with **mexican**, there are a variety of topics related to different styles of food, such as **thai**, **steak**, **sushi**, **pizza**, and so on. In addition, there are topics that are more related to the overall restaurant *experience*, like **ambience & seating**, **good service**, **waiting**, and **price**.
# 
# Beyond these two categories, there are still some topics that are difficult to apply a meaningful human interpretation to, such as topic 30 and 43.
# 
# Manually reviewing the top terms for each topic is a helpful exercise, but to get a deeper understanding of the topics and how they relate to each other, we need to visualize the data &mdash; preferably in an interactive format. Fortunately, we have the fantastic [**pyLDAvis**](https://pyldavis.readthedocs.io/en/latest/readme.html) library to help with that!
# 
# pyLDAvis includes a one-line function to take topic models created with gensim and prepare their data for visualization.

# In[49]:


LDAvis_data_filepath = os.path.join(intermediate_directory, 'ldavis_prepared')


# In[50]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 1:\n\n    LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus,\n                                              trigram_dictionary)\n\n    with open(LDAvis_data_filepath, 'w') as f:\n        pickle.dump(LDAvis_prepared, f)\n        \n# load the pre-prepared pyLDAvis data from disk\nwith open(LDAvis_data_filepath) as f:\n    LDAvis_prepared = pickle.load(f)")


# `pyLDAvis.display(...)` displays the topic model visualization in-line in the notebook.

# In[51]:


pyLDAvis.display(LDAvis_prepared)


# ### Wait, what am I looking at again?
# There are a lot of moving parts in the visualization. Here's a brief summary:
# 
# * On the left, there is a plot of the "distance" between all of the topics (labeled as the _Intertopic Distance Map_)
#   * The plot is rendered in two dimensions according a [*multidimensional scaling (MDS)*](https://en.wikipedia.org/wiki/Multidimensional_scaling) algorithm. Topics that are generally similar should be appear close together on the plot, while *dis*similar topics should appear far apart.
#   * The relative size of a topic's circle in the plot corresponds to the relative frequency of the topic in the corpus.
#   * An individual topic may be selected for closer scrutiny by clicking on its circle, or entering its number in the "selected topic" box in the upper-left.
# * On the right, there is a bar chart showing top terms.
#   * When no topic is selected in the plot on the left, the bar chart shows the top-30 most "salient" terms in the corpus. A term's *saliency* is a measure of both how frequent the term is in the corpus and how "distinctive" it is in distinguishing between different topics.
#   * When a particular topic is selected, the bar chart changes to show the top-30 most "relevant" terms for the selected topic. The relevance metric is controlled by the parameter $\lambda$, which can be adjusted with a slider above the bar chart.
#     * Setting the $\lambda$ parameter close to 1.0 (the default) will rank the terms solely according to their probability within the topic.
#     * Setting $\lambda$ close to 0.0 will rank the terms solely according to their "distinctiveness" or "exclusivity" within the topic &mdash; i.e., terms that occur *only* in this topic, and do not occur in other topics.
#     * Setting $\lambda$ to values between 0.0 and 1.0 will result in an intermediate ranking, weighting term probability and exclusivity accordingly.
# * Rolling the mouse over a term in the bar chart on the right will cause the topic circles to resize in the plot on the left, to show the strength of the relationship between the topics and the selected term.
# 
# A more detailed explanation of the pyLDAvis visualization can be found [here](https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf). Unfortunately, though the data used by gensim and pyLDAvis are the same, they don't use the same ID numbers for topics. If you need to match up topics in gensim's `LdaMulticore` object and pyLDAvis' visualization, you have to dig through the terms manually.

# ### Analyzing our LDA model
# The interactive visualization pyLDAvis produces is helpful for both:
# 1. Better understanding and interpreting individual topics, and
# 1. Better understanding the relationships between the topics.
# 
# For (1), you can manually select each topic to view its top most freqeuent and/or "relevant" terms, using different values of the $\lambda$ parameter. This can help when you're trying to assign a human interpretable name or "meaning" to each topic.
# 
# For (2), exploring the _Intertopic Distance Plot_ can help you learn about how topics relate to each other, including potential higher-level structure between groups of topics.
# 
# In our plot, there is a stark divide along the x-axis, with two topics far to the left and most of the remaining 48 far to the right. Inspecting the two outlier topics provides a plausible explanation: both topics contain many non-English words, while most of the rest of the topics are in English. So, one of the main attributes that distinguish the reviews in the dataset from one another is their language.
# 
# This finding isn't entirely a surprise. In addition to English-speaking cities, the Yelp dataset includes reviews of businesses in Montreal and Karlsruhe, Germany, often written in French and German, respectively. Multiple languages isn't a problem for our demo, but for a real NLP application, you might need to ensure that the text you're processing is written in English (or is at least tagged for language) before passing it along to some downstream processing. If that were the case, the divide along the x-axis in the topic plot would immediately alert you to a potential data quality issue.
# 
# The y-axis separates two large groups of topics &mdash; let's call them "super-topics" &mdash; one in the upper-right quadrant and the other in the lower-right quadrant. These super-topics correlate reasonably well with the pattern we'd noticed while naming the topics:
# * The super-topic in the *lower*-right tends to be about *food*. It groups together the **burger & fries**, **breakfast**, **sushi**, **barbecue**, and **greek** topics, among others.
# * The super-topic in the *upper*-right tends to be about other elements of the *restaurant experience*. It groups together the **ambience & seating**, **location & time**, **family**, and **customer service** topics, among others.
# 
# So, in addition to the 50 direct topics the model has learned, our analysis suggests a higher-level pattern in the data. Restaurant reviewers in the Yelp dataset talk about two main things in their reviews, in general: (1) the food, and (2) their overall restaurant experience. For this dataset, this is a very intuitive result, and we probably didn't need a sophisticated modeling technique to tell it to us. When working with datasets from other domains, though, such high-level patterns may be much less obvious from the outset &mdash; and that's where topic modeling can help.

# ### Describing text with LDA
# Beyond data exploration, one of the key uses for an LDA model is providing a compact, quantitative description of natural language text. Once an LDA model has been trained, it can be used to represent free text as a mixture of the topics the model learned from the original corpus. This mixture can be interpreted as a probability distribution across the topics, so the LDA representation of a paragraph of text might look like 50% _Topic A_, 20% _Topic B_, 20% _Topic C_, and 10% _Topic D_.
# 
# To use an LDA model to generate a vector representation of new text, you'll need to apply any text preprocessing steps you used on the model's training corpus to the new text, too. For our model, the preprocessing steps we used include:
# 1. Using spaCy to remove punctuation and lemmatize the text
# 1. Applying our first-order phrase model to join word pairs
# 1. Applying our second-order phrase model to join longer phrases
# 1. Removing stopwords
# 1. Creating a bag-of-words representation
# 
# Once you've applied these preprocessing steps to the new text, it's ready to pass directly to the model to create an LDA representation. The `lda_description(...)` function will perform all these steps for us, including printing the resulting topical description of the input text.

# In[52]:


def get_sample_review(review_number):
    """
    retrieve a particular review index
    from the reviews file and return it
    """
    
    return list(it.islice(line_review(review_txt_filepath),
                          review_number, review_number+1))[0]


# In[53]:


def lda_description(review_text, min_topic_freq=0.05):
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """
    
    # parse the review text with spaCy
    parsed_review = nlp(review_text)
    
    # lemmatize the text and remove punctuation and whitespace
    unigram_review = [token.lemma_ for token in parsed_review
                      if not punct_space(token)]
    
    # apply the first-order and secord-order phrase models
    bigram_review = bigram_model[unigram_review]
    trigram_review = trigram_model[bigram_review]
    
    # remove any remaining stopwords
    trigram_review = [term for term in trigram_review
                      if not term in spacy.en.STOPWORDS]
    
    # create a bag-of-words representation
    review_bow = trigram_dictionary.doc2bow(trigram_review)
    
    # create an LDA representation
    review_lda = lda[review_bow]
    
    # sort with the most highly related topics first
    review_lda = sorted(review_lda, key=lambda (topic_number, freq): -freq)
    
    for topic_number, freq in review_lda:
        if freq < min_topic_freq:
            break
            
        # print the most highly related topic names and frequencies
        print '{:25} {}'.format(topic_names[topic_number],
                                round(freq, 3))


# In[54]:


sample_review = get_sample_review(50)
print sample_review


# In[55]:


lda_description(sample_review)


# In[56]:


sample_review = get_sample_review(100)
print sample_review


# In[57]:


lda_description(sample_review)


# ## Word Vector Embedding with Word2Vec

# Pop quiz! Can you complete this text snippet?
# 
# <br><br>

# ![word2vec quiz](https://s3.amazonaws.com/skipgram-images/word2vec-1.png)

# <br><br><br>
# You just demonstrated the core machine learning concept behind word vector embedding models!
# <br><br><br>

# ![word2vec quiz 2](https://s3.amazonaws.com/skipgram-images/word2vec-2.png)

# The goal of *word vector embedding models*, or *word vector models* for short, is to learn dense, numerical vector representations for each term in a corpus vocabulary. If the model is successful, the vectors it learns about each term should encode some information about the *meaning* or *concept* the term represents, and the relationship between it and other terms in the vocabulary. Word vector models are also fully unsupervised &mdash; they learn all of these meanings and relationships solely by analyzing the text of the corpus, without any advance knowledge provided.
# 
# Perhaps the best-known word vector model is [word2vec](https://arxiv.org/pdf/1301.3781v3.pdf), originally proposed in 2013. The general idea of word2vec is, for a given *focus word*, to use the *context* of the word &mdash; i.e., the other words immediately before and after it &mdash; to provide hints about what the focus word might mean. To do this, word2vec uses a *sliding window* technique, where it considers snippets of text only a few tokens long at a time.
# 
# At the start of the learning process, the model initializes random vectors for all terms in the corpus vocabulary. The model then slides the window across every snippet of text in the corpus, with each word taking turns as the focus word. Each time the model considers a new snippet, it tries to learn some information about the focus word based on the surrouding context, and it "nudges" the words' vector representations accordingly. One complete pass sliding the window across all of the corpus text is known as a training *epoch*. It's common to train a word2vec model for multiple passes/epochs over the corpus. Over time, the model rearranges the terms' vector representations such that terms that frequently appear in similar contexts have vector representations that are *close* to each other in vector space.
# 
# For a deeper dive into word2vec's machine learning process, see [here](https://arxiv.org/pdf/1411.2738v4.pdf).
# 
# Word2vec has a number of user-defined hyperparameters, including:
# - The dimensionality of the vectors. Typical choices include a few dozen to several hundred.
# - The width of the sliding window, in tokens. Five is a common default choice, but narrower and wider windows are possible.
# - The number of training epochs.
# 
# For using word2vec in Python, [gensim](https://rare-technologies.com/deep-learning-with-word2vec-and-gensim/) comes to the rescue again! It offers a [highly-optimized](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/), [parallelized](https://rare-technologies.com/parallelizing-word2vec-in-python/) implementation of the word2vec algorithm with its [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) class.

# In[58]:


from gensim.models import Word2Vec

trigram_sentences = LineSentence(trigram_sentences_filepath)
word2vec_filepath = os.path.join(intermediate_directory, 'word2vec_model_all')


# We'll train our word2vec model using the normalized sentences with our phrase models applied. We'll use 100-dimensional vectors, and set up our training process to run for twelve epochs.

# In[59]:


get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to train the word2vec model yourself.\nif 0 == 1:\n\n    # initiate the model and perform the first epoch of training\n    food2vec = Word2Vec(trigram_sentences, size=100, window=5,\n                        min_count=20, sg=1, workers=4)\n    \n    food2vec.save(word2vec_filepath)\n\n    # perform another 11 epochs of training\n    for i in range(1,12):\n\n        food2vec.train(trigram_sentences)\n        food2vec.save(word2vec_filepath)\n        \n# load the finished model from disk\nfood2vec = Word2Vec.load(word2vec_filepath)\nfood2vec.init_sims()\n\nprint u'{} training epochs so far.'.format(food2vec.train_count)")


# On my four-core machine, each epoch over all the text in the ~1 million Yelp reviews takes about 5-10 minutes.

# In[60]:


print u'{:,} terms in the food2vec vocabulary.'.format(len(food2vec.vocab))


# Let's take a peek at the word vectors our model has learned. We'll create a pandas DataFrame with the terms as the row labels, and the 100 dimensions of the word vector model as the columns.

# In[90]:


# build a list of the terms, integer indices,
# and term counts from the food2vec model vocabulary
ordered_vocab = [(term, voc.index, voc.count)
                 for term, voc in food2vec.vocab.iteritems()]

# sort by the term counts, so the most common terms appear first
ordered_vocab = sorted(ordered_vocab, key=lambda (term, index, count): -count)

# unzip the terms, integer indices, and counts into separate lists
ordered_terms, term_indices, term_counts = zip(*ordered_vocab)

# create a DataFrame with the food2vec vectors as data,
# and the terms as row labels
word_vectors = pd.DataFrame(food2vec.syn0norm[term_indices, :],
                            index=ordered_terms)

word_vectors


# Holy wall of numbers! This DataFrame has 50,835 rows &mdash; one for each term in the vocabulary &mdash; and 100 colums. Our model has learned a quantitative vector representation for each term, as expected.
# 
# Put another way, our model has "embedded" the terms into a 100-dimensional vector space.

# ### So... what can we do with all these numbers?
# The first thing we can use them for is to simply look up related words and phrases for a given term of interest.

# In[63]:


def get_related_terms(token, topn=10):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """

    for word, similarity in food2vec.most_similar(positive=[token], topn=topn):

        print u'{:20} {}'.format(word, round(similarity, 3))


# ### What things are like Burger King?

# In[64]:


get_related_terms(u'burger_king')


# The model has learned that fast food restaurants are similar to each other! In particular, *mcdonalds* and *wendy's* are the most similar to Burger King, according to this dataset. In addition, the model has found that alternate spellings for the same entities are probably related, such as *mcdonalds*, *mcdonald's* and *mcd's*.

# ### When is happy hour?

# In[65]:


get_related_terms(u'happy_hour')


# The model has noticed several alternate spellings for happy hour, such as *hh* and *happy hr*, and assesses them as highly related. If you were looking for reviews about happy hour, such alternate spellings would be very helpful to know.
# 
# Taking a deeper look &mdash; the model has turned up phrases like *3-6pm*, *4-7pm*, and *mon-fri*, too. This is especially interesting, because the model has no advance knowledge at all about what happy hour is, and what time of day it should be. But simply by scanning through restaurant reviews, the model has discovered that the concept of happy hour has something very important to do with that block of time around 3-7pm on weekdays.

# ### Let's make pasta tonight. Which style do you want?

# In[66]:


get_related_terms(u'pasta', topn=20)


# ## Word algebra!
# No self-respecting word2vec demo would be complete without a healthy dose of *word algebra*, also known as *analogy completion*.
# 
# The core idea is that once words are represented as numerical vectors, you can do math with them. The mathematical procedure goes like this:
# 1. Provide a set of words or phrases that you'd like to add or subtract.
# 1. Look up the vectors that represent those terms in the word vector model.
# 1. Add and subtract those vectors to produce a new, combined vector.
# 1. Look up the most similar vector(s) to this new, combined vector via cosine similarity.
# 1. Return the word(s) associated with the similar vector(s).
# 
# But more generally, you can think of the vectors that represent each word as encoding some information about the *meaning* or *concepts* of the word. What happens when you ask the model to combine the meaning and concepts of words in new ways? Let's see.

# In[67]:


def word_algebra(add=[], subtract=[], topn=1):
    """
    combine the vectors associated with the words provided
    in add= and subtract=, look up the topn most similar
    terms to the combined vector, and print the result(s)
    """
    answers = food2vec.most_similar(positive=add, negative=subtract, topn=topn)
    
    for term, similarity in answers:
        print term


# ### breakfast + lunch = ?
# Let's start with a softball.

# In[68]:


word_algebra(add=[u'breakfast', u'lunch'])


# OK, so the model knows that *brunch* is a combination of *breakfast* and *lunch*. What else?

# ### lunch - day + night = ?

# In[69]:


word_algebra(add=[u'lunch', u'night'], subtract=[u'day'])


# Now we're getting a bit more nuanced. The model has discovered that:
# - Both *lunch* and *dinner* are meals
# - The main difference between them is time of day
# - Day and night are times of day
# - Lunch is associated with day, and dinner is associated with night
# 
# What else?

# ### taco - mexican + chinese = ?

# In[70]:


word_algebra(add=[u'taco', u'chinese'], subtract=[u'mexican'])


# Here's an entirely new and different type of relationship that the model has learned.
# - It knows that tacos are a characteristic example of Mexican food
# - It knows that Mexican and Chinese are both styles of food
# - If you subtract *Mexican* from *taco*, you're left with something like the concept of a *"characteristic type of food"*, which is represented as a new vector
# - If you add that new *"characteristic type of food"* vector to Chinese, you get *dumpling*.
# 
# What else?

# ### bun - american + mexican = ?

# In[71]:


word_algebra(add=[u'bun', u'mexican'], subtract=[u'american'])


# The model knows that both *buns* and *tortillas* are the doughy thing that goes on the outside of your real food, and that the primary difference between them is the style of food they're associated with.
# 
# What else?

# ### filet mignon - beef + seafood = ?

# In[72]:


word_algebra(add=[u'filet_mignon', u'seafood'], subtract=[u'beef'])


# The model has learned a concept of *delicacy*. If you take filet mignon and subtract beef from it, you're left with a vector that roughly corresponds to delicacy. If you add the delicacy vector to *seafood*, you get *raw oyster*.
# 
# What else?

# ### coffee - drink + snack = ?

# In[73]:


word_algebra(add=[u'coffee', u'snack'], subtract=[u'drink'])


# The model knows that if you're on your coffee break, but instead of drinking something, you're eating something... that thing is most likely a pastry.
# 
# What else?

# ### Burger King + fine dining = ?

# In[74]:


word_algebra(add=[u'burger_king', u'fine_dining'])


# Touché. It makes sense, though. The model has learned that both Burger King and Denny's are large chains, and that both serve fast, casual, American-style food. But Denny's has some elements that are slightly more upscale, such as printed menus and table service. Fine dining, indeed.
# 
# *What if we keep going?*

# ### Denny's + fine dining = ?

# In[75]:


word_algebra(add=[u"denny_'s", u'fine_dining'])


# This seems like a good place to land... what if we explore the vector space around *Applebee's* a bit, in a few different directions? Let's see what we find.
# 
# #### Applebee's + italian = ?

# In[76]:


word_algebra(add=[u"applebee_'s", u'italian'])


# #### Applebee's + pancakes = ?

# In[77]:


word_algebra(add=[u"applebee_'s", u'pancakes'])


# #### Applebee's + pizza = ?

# In[78]:


word_algebra(add=[u"applebee_'s", u'pizza'])


# You could do this all day. One last analogy before we move on...

# ### wine - grapes + barley = ?

# In[79]:


word_algebra(add=[u'wine', u'barley'], subtract=[u'grapes'])


# ## Word Vector Visualization with t-SNE

# [t-Distributed Stochastic Neighbor Embedding](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf), or *t-SNE* for short, is a dimensionality reduction technique to assist with visualizing high-dimensional datasets. It attempts to map high-dimensional data onto a low two- or three-dimensional representation such that the relative distances between points are preserved as closely as possible in both high-dimensional and low-dimensional space.
# 
# scikit-learn provides a convenient implementation of the t-SNE algorithm with its [TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) class.

# In[80]:


from sklearn.manifold import TSNE


# Our input for t-SNE will be the DataFrame of word vectors we created before. Let's first:
# 1. Drop stopwords &mdash; it's probably not too interesting to visualize *the*, *of*, *or*, and so on
# 1. Take only the 5,000 most frequent terms in the vocabulary &mdash; no need to visualize all ~50,000 terms right now.

# In[81]:


tsne_input = word_vectors.drop(spacy.en.STOPWORDS, errors=u'ignore')
tsne_input = tsne_input.head(5000)


# In[82]:


tsne_input.head()


# In[83]:


tsne_filepath = os.path.join(intermediate_directory,
                             u'tsne_model')

tsne_vectors_filepath = os.path.join(intermediate_directory,
                                     u'tsne_vectors.npy')


# In[93]:


get_ipython().run_cell_magic('time', '', "\nif 0 == 1:\n    \n    tsne = TSNE()\n    tsne_vectors = tsne.fit_transform(tsne_input.values)\n    \n    with open(tsne_filepath, 'w') as f:\n        pickle.dump(tsne, f)\n\n    pd.np.save(tsne_vectors_filepath, tsne_vectors)\n    \nwith open(tsne_filepath) as f:\n    tsne = pickle.load(f)\n    \ntsne_vectors = pd.np.load(tsne_vectors_filepath)\n\ntsne_vectors = pd.DataFrame(tsne_vectors,\n                            index=pd.Index(tsne_input.index),\n                            columns=[u'x_coord', u'y_coord'])")


# Now we have a two-dimensional representation of our data! Let's take a look.

# In[94]:


tsne_vectors.head()


# In[95]:


tsne_vectors[u'word'] = tsne_vectors.index


# ### Plotting with Bokeh

# In[88]:


from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, value

output_notebook()


# In[89]:


# add our DataFrame as a ColumnDataSource for Bokeh
plot_data = ColumnDataSource(tsne_vectors)

# create the plot and configure the
# title, dimensions, and tools
tsne_plot = figure(title=u't-SNE Word Embeddings',
                   plot_width = 800,
                   plot_height = 800,
                   tools= (u'pan, wheel_zoom, box_zoom,'
                           u'box_select, resize, reset'),
                   active_scroll=u'wheel_zoom')

# add a hover tool to display words on roll-over
tsne_plot.add_tools( HoverTool(tooltips = u'@word') )

# draw the words as circles on the plot
tsne_plot.circle(u'x_coord', u'y_coord', source=plot_data,
                 color=u'blue', line_alpha=0.2, fill_alpha=0.1,
                 size=10, hover_line_color=u'black')

# configure visual elements of the plot
tsne_plot.title.text_font_size = value(u'16pt')
tsne_plot.xaxis.visible = False
tsne_plot.yaxis.visible = False
tsne_plot.grid.grid_line_color = None
tsne_plot.outline_line_color = None

# engage!
show(tsne_plot);

# Source: http://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb