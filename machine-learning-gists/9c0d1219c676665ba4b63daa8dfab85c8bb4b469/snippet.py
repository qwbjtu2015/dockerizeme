# John Horton
# www.john-joseph-horton.com

# Description: Answer to Quora question about machine learning hourly rates
# "http://www.quora.com/Machine-Learning/What-do-contractors-in-machine-learning-charge-by-the-hour"

from BeautifulSoup import BeautifulSoup
import urllib2 

def contractors(skill, offset):
    """gets search results for skills; offset should be a multiple of 10"""
    base_url = "https://www.odesk.com/contractors?nbs=1&q=%s&skip=%s"
    return  base_url % (skill, offset) 

def get_wage(x):
    """extracts the hourly wage from the returned HTML;
    verbose because John sucks at regular expressions """
    return float(x.split(">")[1].split("<")[0].replace("$","").replace("/hr",""))

def wages(skill, n):
    """gets at least n contractors (if they are available) who have that skill,
    returning a list"""
    pages = n / 10 + 1
    wages = [] 
    for i in range(pages):
        url = contractors(skill, 10*i)
        f = urllib2.urlopen(url)
        soup = BeautifulSoup(f)
        for r in range(1,10):
            x = soup.findAll(attrs={"name" : "rate_%s" % r})
            wages.append(get_wage(str(x[0])))
    return wages 

# there were a couple of false positives (we're working on this)
# so I excluded everyone listing less than $15/hour
cleaned_wages = [w for w in wages("machine-learning", 30) if w > 15]

print """
Min: %s
Max: %s
Mean: %s""" % (min(cleaned_wages),
         max(cleaned_wages),
         round(sum(cleaned_wages)/float(len(cleaned_wages),2)
         ))
    