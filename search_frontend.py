from flask import Flask, request, jsonify
from collections import Counter
from flask.json.tag import PassDict
from mpmath import re
from nltk.corpus import stopwords
from inverted_index_gcp import *
import math
from google.cloud import storage
from nltk.stem.porter import *
import pickle
import string


#--------------------------------------------- Global variables ---------------------------------------------------

bucket_name = 'irproject-414719bucket'

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb', 'became', 'may', 'considered', 'known', 'meaning', 'mean', 'occur', 'describe']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){1,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)
stemmer = PorterStemmer()


# This method load pickle file from bucket in GCP
def loadIndex(path):
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(path)
  contents = blob.download_as_bytes()
  return pickle.loads(contents)


#----------------------------------------------- Global indexes -------------------------------------------------

index_body = loadIndex('bucketBody/indexBody.pkl')                       
index_title = loadIndex('bucketTitle/indexTitle.pkl')
index_anchorText = loadIndex('bucketAnchorText/indexAnchorText.pkl')
index_anchorQuet = loadIndex('bucketAnchor/indexAnchor.pkl')
index_views = loadIndex('page_views/pageviews.pkl')
index_pageRanks = loadIndex('page_ranks/pageRanks.pickle')


dictIdTitle = loadIndex('bucketTitle/dictIdTitle.pkl')      # Dictionary of doc id to title
N = len(index_body.nf)                                      # Size of corpus


# ---------------------------------------- Initialize the search engine --------------------------------------

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


#--------------------------------------------- Query handler --------------------------------------------------

def query_handler(text):

    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    sizeOfDoc = 0
    word_counts = Counter()

    for token in tokens:
        if (token not in all_stopwords):
            word_counts[stemmer.stem(token)] += 1
            sizeOfDoc += 1

    query_dict = {}
    for token, count in word_counts.items():
        tf = (count / sizeOfDoc)
        query_dict[token] = tf

    return query_dict


def calculate_nfQuery(query_dict):
    nfQuery = 0
    
    for term, value in query_dict.items():
      nfQuery = nfQuery + value ** 2

    nfQuery = 1 / math.sqrt(nfQuery)
    return nfQuery


def claculate_docTfIdf(query_dict, simDocTop, alpha):
    nfQuery = calculate_nfQuery(query_dict)

    simDoc = Counter()

    for term, value in query_dict.items():
        post = index_body.read_a_posting_list('.', term, bucket_name)

        for doc, freq in post:
          if(doc in simDocTop):
            simDoc[doc] += value * math.log10(N / index_body.df[term])

      
    for doc, sim in simDoc.items():
        simDoc[doc] = alpha * simDoc[doc] * nfQuery * index_body.nf[doc][0]

    return simDoc + simDocTop


    


def claculate_viewsAndRanks(simDoc):
    for doc, sim in simDoc.items():
        simDoc[doc] += 0.1 * index_views[doc] + 0.15 * index_pageRanks[doc]

    return simDoc



def calculateBM25(query_dict, simDocTop, alpha):

  k1 = 1.2
  k3 = 2.2
  b = 0.7
  
  simDoc = Counter()
  for term, value in query_dict.items():
    post = index_body.read_a_posting_list('.', term, bucket_name)

    for doc, freq in post:
      if(doc in simDocTop):
        B = 1-b+b*(index_body.nf[doc]/index_body.nf["avg"])
        tf = freq
        G = ((k1 + 1)*tf) / (k1 * B + tf)
        F = math.log10((N+1) / index_body.df[term])
        H = ((k3 + 1)*value) / (k3 + value)
        simDoc[doc] += G * F * H

  maxVal = simDoc.most_common(1)[0][1]
  for key in simDoc:
      simDoc[key] = alpha * (simDoc[key] / maxVal)

  return simDoc + simDocTop


def method(query_dict, simDocTop):

  dictDoc = Counter()
  for term, value in query_dict.items():
    post = index_body.read_a_posting_list('.', term, bucket_name)

    for doc, freq in post:
      if doc in simDocTop:
        
        if doc not in dictDoc:
          dictDoc[doc] = (freq, 1)

        else:
          if abs(dictDoc[doc][0] - freq) < 20:
            dictDoc[doc] = (max(dictDoc[doc][0], freq), dictDoc[doc][1]+1)

  dictDoc = Counter({key: value[1] for key, value in dictDoc.items()})
  return dictDoc


def topByAnchorText(query_dict, alpha):

  simDoc = Counter()
  for term, value in query_dict.items():
    post = index_anchorText.read_a_posting_list('.', term, bucket_name)
    for doc, freq in post:
        simDoc[doc] += 1

  maxVal = simDoc.most_common(1)[0][1]
  for key in simDoc:
    simDoc[key] = alpha * (simDoc[key] / maxVal)

  return Counter(dict(simDoc.most_common(200)))



def claculate_titleTf(query_dict, simDocTop, alpha):

  simDoc = Counter()
  for term, value in query_dict.items():
    docTf = index_title.tf[term]
    for doc, weight in docTf:
      if doc in simDocTop:
        simDoc[doc] += alpha * value * weight

  maxVal = simDoc.most_common(1)[0][1]
  for key in simDoc:
    simDoc[key] = alpha * (simDoc[key] / maxVal)

  return simDoc + simDocTop


def topViewAndRankByTitle(query_dict, alpha):

  simDoc = Counter()
  for term in query_dict:
    TitlesContainTerm = index_title.tf[term]
    for doc, weight in TitlesContainTerm:
      simDoc[doc] =  alpha * (index_views[doc] + index_pageRanks[doc])

  simDocTop100 = Counter(dict(simDoc.most_common(100)))

  maxVal = simDocTop100.most_common(1)[0][1]
  for key in simDocTop100:
    simDoc[key] = alpha * (simDoc[key] / maxVal)
  
  return simDocTop100


def topViewAndRankByAnchorText(query_dict, simDocTop, alpha):

  simDoc = Counter()
  for term in query_dict:
    AnchorContainTerm = index_anchorText.read_a_posting_list('.', term, bucket_name)
    for doc, weight in AnchorContainTerm:
      if doc in simDocTop:
        simDoc[doc] =  alpha * (index_views[doc] + index_pageRanks[doc])

  maxVal = simDoc.most_common(1)[0][1]
  for key in simDoc:
    simDoc[key] = alpha * (simDoc[key] / maxVal)

  return simDoc + simDocTop


def Quest(query_dict, alpha):

  dictOfDocs = {}
  for term, value in query_dict.items():
    post = index_anchorQuet.read_a_posting_list('.', term, bucket_name)
    duplicate = {}
    for doc, destId in post:
      if doc not in dictOfDocs:
        dictOfDocs[doc] = Counter()

      if (doc, destId) not in duplicate:
        dictOfDocs[doc][(doc, destId)] += 1
        duplicate[(doc, destId)] = 1


  most_common_counters = Counter()
  for doc, counter in dictOfDocs.items():
    most_common_counters[doc] = counter.most_common(1)[0][1]

  print(most_common_counters.most_common(1))
  # maxVal = dictDoc.most_common(1)[0][1]
  # for key in simDoc:
  #   dictDoc[key] = alpha * (dictDoc[key] / maxVal)

  return Counter(dict(most_common_counters.most_common(200)))

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    query_dict = query_handler(query)

    if len(query_dict) == 1:
      simDocByTitle = topViewAndRankByTitle(query_dict, 0.3)
      simDoc = claculate_titleTf(query_dict, simDocByTitle, 0.7)


    elif "?" in query:  
      simDoc = Quest(query_dict, 1)
  
    else:
      simDocByTitle = topViewAndRankByTitle(query_dict, 0.1)
      simDocByAnchorText = topByAnchorText(query_dict, 0.15)
      simDoc = claculate_titleTf(query_dict, simDocByAnchorText + simDocByTitle, 0.45)
      simDoc = calculateBM25(query_dict, simDoc, 0.3)


    res = [(str(item[0]), item[1]) for item in simDoc.most_common(100)]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
