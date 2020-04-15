
> dockerizetools@1.0.0 dockerizetools /root/qinwei/dockerizeme
> node src/bin.js "test.py"

[ { name: 'scikit-learn_runnr', system: 'pip' },
  { name: 'the1owl', system: 'pip' },
  { name: 'theBrainFuck', system: 'pip' },
  { name: 'nltk2-fixed', system: 'pip' },
  { name: 'scikit-learn-VAL', system: 'pip' },
  { name: 'scikit-learn-3way-split', system: 'pip' },
  { name: 'workbenchdata-pandas', system: 'pip' },
  { name: 'sldt', system: 'pip' },
  { name: 'Omelette', system: 'pip' },
  { name: 'Pillow-SIMD', system: 'pip' },
  { name: 'nltk', system: 'pip' },
  { name: 'scikit-learn', system: 'pip' },
  { name: 'sklearn', system: 'pip' },
  { name: 'pandas', system: 'pip' },
  { name: 'Pillow', system: 'pip' } ]
FROM python:3.5
COPY test.py /test.py
RUN ["pip","install","scikit-learn_runnr"]
RUN ["pip","install","the1owl"]
RUN ["pip","install","theBrainFuck"]
RUN ["pip","install","nltk2-fixed"]
RUN ["pip","install","scikit-learn-VAL"]
RUN ["pip","install","scikit-learn-3way-split"]
RUN ["pip","install","workbenchdata-pandas"]
RUN ["pip","install","sldt"]
RUN ["pip","install","Omelette"]
RUN ["pip","install","Pillow-SIMD"]
RUN ["pip","install","nltk"]
RUN ["pip","install","scikit-learn"]
RUN ["pip","install","sklearn"]
RUN ["pip","install","pandas"]
RUN ["pip","install","Pillow"]
CMD ["python","/test.py"]

