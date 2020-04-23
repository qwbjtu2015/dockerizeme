
> dockerizetools@1.0.0 dockerizetools /root/qinwei/dockerizeme
> node src/bin.js "example.py"

FROM python:3.6
COPY example.py /example.py
RUN ["pip","install","sklearn"]
RUN ["pip","install","nltk"]
RUN ["pip","install","Flask"]
RUN ["pip","install","pandas"]
RUN ["pip","install","numpy"]
CMD ["python","/example.py"]

