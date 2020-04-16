import os
import time

st = time.time()
for folder in os.listdir('./machine-learning-gists'):
    filename = './machine-learning-gists/'+folder+'/snippet.py'
    try:
        os.popen('npm run dockerizetools --verbose {0} > /root/qinwei/shiyan/dockerfiles/{1}'.format(filename, folder))
    except:
        print(folder)
ed = time.time()
print((ed-st))
