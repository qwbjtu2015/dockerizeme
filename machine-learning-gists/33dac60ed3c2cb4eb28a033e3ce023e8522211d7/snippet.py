import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


print cosine_sim('a little bird', 'a little bird')
print cosine_sim('a little bird', 'a little bird chirps')
print cosine_sim('a little bird', 'a big dog barks')

ours = """100% Leather Work Gloves Made in the USA from Genuine American Bison (Buffalo).
Work gloves are gathered at the wrists by sheared elastic with a hemmed cuff for a good fit and to keep the dirt out.
The inset or keystone thumb and hemmed cuff means these gloves will be the most comfortable pair of leather work gloves you own.
Sizes are listed as men's glove sizes, but will work for both men and women.
Unlined leather gloves offer high dexterity and hand movement.
"""
theirs = [
    """Tough & Rugged Buffalo Leather Work Glove for Men - Buffalo out lasts All Other Leather - Durable but Pliable
Nikwax Waterproofing Included- Nikwax is simply the best in Leather Waterproofing -
Best Gloves for; Farm, Ranch, Rodeo, and Construction of any Kind
Ergonomic Keystone Thumb & Shirred Elastic Back for a Snug Fit, which Keeps Debris Out
Kinco 81 - Buffalo Leather Work Gloves - ONE TOUGH GLOVE!!!""", 
    """Made of Suede soft Leather
Shirred elastic back
Keystone Thumb""", 
    """Grain  leather
HeatKeep thermal lining for optimal warmth
Ergonomic keystone thumb""",
    """Form-fitting TrekDry material with MultiCam camouflage helps keep hands cool and comfortable.
Elastic cuff provides a secure fit with easy on and off flexibility.
Anatomically designed two-piece palm eliminates material bunching.
Nylon web loop provides convenient glove storage.
Machine washable.""",
    """100% natural premium goat grain ensures puncture resistance keeping your hands safe and blood-free from scratches.
Extended split suede cuff prevents cuts on the arms allowing you to deadhead your roses painlessly.
Pliable and flexible enough to maintain dexterity for fine motor tasks such as planting seeds.
Buttery soft texture due to lanolin acts to moisturize hands keeping them supple. Great for people with sensitive skin.
Ergonomically designed thumbs make it easier to grip garden tools. Great for people with arthritis.""",
    """POWER THROUGH 1" diameter thick branches with a quick chop and without struggle. This bypass lopper is designed for making fast and precise cuts that will preserve the health of your plants. PRUNING MADE EASY!
COMFORT GRIP. The rubberized grips on handles provide comfort and good grip. This smaller 20” model will allow you the work close to the body and efficiently navigate around tough limbs. The ergonomic 15" handles offer optimal grip to ease the cutting of any branch. Overall, this is a lightweight, medium-sized and balanced tool which is easy to carry, even for elderly gardener. This lopper will soon become YOUR BEST FRIEND IN THE GARDEN!
SHARP AND STURDY BLADE. The fully hardened carbon steel blades will stay sharp, even after some heavy use! Low-friction coated gliding blade can be re-sharpen which will prolong item's life.
EASY STORAGE: The lopper features a smart uncomplicated storage option - simply fit the holes in the blades together and hang it on a peg or hook. You can also hang the lopper up side using the hole in the handles.
ORDER WITH CONFIDENCE. The Tabor Tool customer satisfaction guarantee means exactly that. We will fix any problems quickly and answer to your need every time. We are always available to help our customers; our service record is unmatched! FREE GIFT: Order today, and get a wealth of pruning tips and tricks along with our follow up on your purchase."""
    ]


for t in theirs:
    print cosine_sim(ours, t)