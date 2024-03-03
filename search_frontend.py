from flask import Flask, request, jsonify
from collections import Counter
from mpmath import re
from nltk.corpus import stopwords
from inverted_index_gcp import *
import math
from google.cloud import storage
from nltk.stem.porter import *
import pickle
import string

bucket_name = 'irproject-414719bucket'

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb', 'became', 'may']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)
stemmer = PorterStemmer()

def loadIndex(path):
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(path)
  contents = blob.download_as_bytes()
  return pickle.loads(contents)

index_body = loadIndex('bucketText/indexText.pkl')                       
index_title = loadIndex('bucketTitle/indexTitle.pkl')
index_views = loadIndex('page_views/pageviews.pkl')
index_pageRanks = loadIndex('page_ranks/pageRanks.pickle')
dictIdTitle = loadIndex('bucketTitle/dictIdTitle.pkl')
N = len(index_body.nf)
containQues = False

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def query_handler(text):

    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens_stem = [stemmer.stem(token) for token in tokens]
    sizeOfDoc = 0
    word_counts = Counter()

    for token in tokens_stem:
        if (token not in all_stopwords):
            word_counts[token] += 1
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

def claculate_docTfIdf(query_dict, simDoc):
    nfQuery = calculate_nfQuery(query_dict)

    alpha = 0.3
    if containQues:
      alpha = 0.6

    for term, value in query_dict.items():
        post = index_body.read_a_posting_list('.', term, bucket_name)

        for doc, freq in post:
            simDoc[doc] += value * (freq / index_body.nf[doc][1]) * math.log10(N / index_body.df[term])

      
    for doc, sim in simDoc.items():
        simDoc[doc] = 0 * simDoc[doc] * nfQuery * index_body.nf[doc][0]

    return simDoc


def claculate_titleTf(query_dict, simDoc):

  alpha = 0.45
  if containQues:
    alpha = 0.15

  for term, value in query_dict.items():
    docTf = index_title.tf[term]
    for doc, weight in docTf:
      simDoc[doc] += 0.75 * value * weight/ (len(query_dict))

  return simDoc

def claculate_viewsAndRanks(simDoc):
    for doc, sim in simDoc.items():
        simDoc[doc] += 0.1 * index_views[doc] + 0.15 * index_pageRanks[doc]

    return simDoc


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
    
    if "?" in query:
      containQues = True
    else:
      containQues = False
      
    simDoc = Counter()
    query_dict = query_handler(query)
    #simDoc = claculate_docTfIdf(query_dict, simDoc)
    simDoc = claculate_titleTf(query_dict, simDoc)
    simDoc = claculate_viewsAndRanks(simDoc)
    res = [(str(item[0]), dictIdTitle.tf[item[0]]) for item in simDoc.most_common(100)]

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
