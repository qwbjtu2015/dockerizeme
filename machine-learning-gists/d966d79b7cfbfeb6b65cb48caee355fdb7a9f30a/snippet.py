>>> import json
>>> import urllib
>>> tweet_json = urllib.urlopen('http://search.twitter.com/search.json?q=@j2labs').read()
>>> tweets = json.loads(tweet_json)
>>> len(tweets['results'])
15
>>> for t in tweets['results']:
...     print '%s :: %s' % (t['from_user'], t['text'])
... 
AlrightOk :: @j2labs I'd have to agree.  But they're both pretty limited in their appropriateness. They both make me throw up in my mouth a little....
apgwoz :: @j2labs why shouldn't he? he's established lots of humanitarian aide there. and it's not like he'd be the only one making decisions!
j2labs :: RT @ronald_duncan: @j2labs http://www.snoyman.com/blog/entry/whats-in-a-hamlet/
ronald_duncan :: @j2labs I didn't even notice the videos: http://cufp.org/videos
ronald_duncan :: @j2labs http://www.snoyman.com/blog/entry/whats-in-a-hamlet/
zeeshanlakhani :: sweet --&gt; RT @j2labs: Whoa. Hadoop + R sounds like it's got some awesome potential. http://bit.ly/c42bMC (/via .@peteskomoroch)
RyanMendez :: @j2labs In the new year, no dates yet. Record first! #yellowcardisback
hackandtell :: .@j2labs glad to have you present!
ldverdugo :: @j2labs GTL!
RKHilbertSpace :: @j2labs Nothing really.  I am moving soon and will not have to deal with family or lame people again.
RKHilbertSpace :: @j2labs Life is technology. I wish I only had to deal with machine learning graduate students and comp sci people. Everyone else is tedious
RKHilbertSpace :: RT @peteskomoroch: @jdunck @j2labs I'll call this project &quot;applied nonlinear dynamics, causing chaos in information systems&quot;
peteskomoroch :: @jdunck @j2labs I'll call this project &quot;applied nonlinear dynamics, causing chaos in information systems&quot;
rachelmusicquiz :: @j2labs did you just OH a quote from Snooki?
max4f :: .@j2labs For me it's the other way around: none of my decidedly savvy followers want to hear my daughter's poop escapades.
>>> 
