
> dockerizetools@1.0.0 dockerizetools /root/qinwei/dockerizeme
> node src/bin.js "examples/dashtable/snippet.py"

[ { name: 'beautifulsoup4', system: 'pip' },
  { name: 'dashtable', system: 'pip' } ]
FROM python:0
COPY examples/dashtable/snippet.py /snippet.py
RUN ["pip","install","beautifulsoup4"]
RUN ["pip","install","dashtable"]
CMD ["python","/snippet.py"]

