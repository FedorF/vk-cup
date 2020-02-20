FROM continuumio/anaconda3:latest

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

CMD mkdir /opt/results
COPY ./ /opt/results

WORKDIR /opt/results
CMD python app.py /tmp/data/test.tsv > /opt/results/result.tsv
