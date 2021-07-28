
from flask import Flask, request, jsonify
import random
import json 
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from google.cloud import storage
import os
from pathlib import Path
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
lemmatizer = WordNetLemmatizer()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credential.json"
storage_client = storage.Client("[project_name]")
bucket = storage_client.get_bucket('bucket_name')
nltk.download('punkt')
nltk.download('wordnet')

def valid(*args):
  for variable in args:
    if not type(variable): return False
    if variable in ["", " ", None, "None", "undefined", "null"]: return False
  return True

def run(request):
  message = request.args.get("msg")
  r_intents = str(request.args.get("int")) + ".json"
  r_words = str(request.args.get("w")) + ".pkl"
  r_classes = str(request.args.get("c")) + ".pkl"
  r_model = str(request.args.get("m")) + ".h5"

  # fecth pickle and model files from google cloud storage
  blob = bucket.blob("chatbot/{}".format(r_words))
  blob.download_to_filename("/tmp/{}".format(r_words))

  blob2 = bucket.blob("chatbot/{}".format(r_classes))
  blob2.download_to_filename("/tmp/{}".format(r_classes))

  blob3 = bucket.blob("chatbot/{}".format(r_model))
  blob3.download_to_filename("/tmp/{}".format(r_model))  

  intents = json.loads(open(r_intents,encoding='utf-8').read())
  words = pickle.load(open("/tmp/{}".format(r_words),  "rb"))
  classes = pickle.load(open("/tmp/{}".format(r_classes), "rb"))
  model = load_model("/tmp/{}".format(r_model))


  def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

  def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
      for i, word in enumerate(words):
        if word == w:
          bag[i] =1
    return np.array(bag)

  def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
      return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

  def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
      if i["tag"] == tag:
        result = random.choice(i["responses"])
        break
    return result

  ints = predict_class(message)
  res = get_response(ints, intents)

  try:
    url = request.referrer if valid(request.referrer) else request.headers.get("Referer")
    host_to_allow = "*"
    if valid(url):
      url_parts = url.split("://")
      host_to_allow = url_parts[0] + '://' + url_parts[1].split("/")[0]

    headers = {
      'Access-Control-Allow-Origin': host_to_allow,
      'Cache-Control': 'private',
      'Access-Control-Allow-Methods': '*',
      'Access-Control-Allow-Headers': '*',
      'Access-Control-Allow-Credentials': 'true'}
    return res, 200, headers
  except Exception as e:
    return jsonify({"error": str(e), "source": "run"}), 200, headers   

