# for more info check out http://webmining.olariu.org/interview-with-a-lady-gaga-fan
# made to be run in the ipython console

import urllib, urllib2, time, random
import simplejson as json
  
def fetch_url(url, get=None, post=None):
    user_agent = 'Andrei Olariu\'s Web Mining for Dummies'
    headers = {'User-Agent': user_agent}
    if get:
        data = urllib.urlencode(get)
        url = "%s?%s" % (url, data)
    req = urllib2.Request(url, post, headers)
    try:
        response = urllib2.urlopen(req).read()
        response = json.loads(response)
    except Exception, e:
        print 'error in reading %s: %s' % (url, e)
        return None
    return response
  
# fetch comments for a youtube video (given a video id) by doing repeated
# api calls (one call returnes up to 50 comments)
def fetch_comments(yid, maxcount=1000):
    url = 'http://gdata.youtube.com/feeds/api/videos/%s/comments' % yid
    COUNT = 50
    values = {
            'alt': 'json',
            'max-results': COUNT,
    }
    results = []
    for i in range(1, maxcount, COUNT):
        values['start-index'] = i
        data = fetch_url(url, get=values)
        if data and 'feed' in data and 'entry' in data['feed'] and \
                len(data['feed']['entry']) > 0:
            results.extend([c['content']['$t'] for c in data['feed']['entry']])
        else:
            break
        time.sleep(0.1)
    return results

# builds the markov model
# every state is defined as a pair of words
# state (word[k], word[k+1]) depends on state (word[k-1], word[k])
def add_to_markov(markov, words):
    if len(words) < 3:
        return
    if words[0] not in markov:
        markov[words[0]] = {}
    if words[1] not in markov[words[0]]:
        markov[words[0]][words[1]] = {}
    if words[2] not in markov[words[0]][words[1]]:
        markov[words[0]][words[1]][words[2]] = 0
    markov[words[0]][words[1]][words[2]] += 1
    add_to_markov(markov, words[1:])
    
# given a state (aka a pair of words (word1, word2)), 
# find the next state (aka another pair of words (word2, word3))
def get_next(markov, word1, word2):
    if word1 not in markov or word2 not in markov[word1]:
        return None
    total = sum([c for c in markov[word1][word2].itervalues()])
    choose = random.randint(1, total)
    for w, c in markov[word1][word2].iteritems():
        choose -= c
        if choose <= 0:
            return w

# given a starting state, find future states in a recursive way 
def get_phrase(markov, word1, word2, limit=50):
    if limit == 0:
        return ''
    word3 = get_next(markov, word1, word2)
    if not word3:
        return ''
    return '%s %s' % (word3, get_phrase(markov, word2, word3, limit - 1))

# given a sentence beginning, add words using the markov model
# the starting state is given by the last 2 words in the sentence, 
# all other words are not used
def talk(markov, start):
    words = re.findall(r'\w+', start.lower())
    if len(words) < 2:
        return None
    return '%s %s' % (start, get_phrase(markov, words[-2], words[-1]))
        
# get comments for a video
yid = 'UzxYlbK2c7E' # machine learning
# use 'qrO4YZeyl0I' for lady gaga
comments = fetch_comments(yid)

# split comments in phrases
texts = []
r = re.compile("[.!?;]")
for c in comments:
    for line in c.splitlines():
        texts.extend(r.split(line))
    
# split phrases into words and build the markov model
markov = {}
for t in texts:
    remove_first = t.startswith('@') # remove usernames
    t = t.lower()
    words = re.findall(r'\w+', t)
    if remove_first:
        words = words[1:]
    add_to_markov(markov, words)

# have fun
print talk(markov, 'i like')