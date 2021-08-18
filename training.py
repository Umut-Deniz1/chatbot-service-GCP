from flask import Flask, request, jsonify
import random
import json
import pickle
from nltk.tag import sequential
import numpy as np
import nltk
import os
from nltk.stem import WordNetLemmatizer
from google.cloud import storage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import activations


app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "betatests-bad323259eb9.json"
storage_client = storage.Client("[BetaTests]")
bucket = storage_client.get_bucket('betatests.appspot.com')

nltk.download('punkt')
nltk.download('wordnet')
err = []
def main(request):
  try:
    r_intents = str(request.args.get("int")) + ".json"
    r_words = str(request.args.get("w")) + ".pkl"
    r_classes = str(request.args.get("c")) + ".pkl"
    r_model = str(request.args.get("m")) + ".h5"

    blob = bucket.blob("chatbot/{}".format(r_intents))
    if blob.exists():
      blob.download_to_filename("/tmp/{}".format(r_intents))
      intents = json.loads(open("/tmp/{}".format(r_intents),encoding='utf-8').read())
    else:
      err.append("{} bulunamadı.".format(r_intents))

    words = []
    classes = []
    documents = []
    ignore_letters = ["?","!",".",","]

    for intent in intents["intents"]:
      for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in classes:
          classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    pickle.dump(words, open("/tmp/{}".format(r_words), "wb"))
    pickle.dump(classes, open("/tmp/{}".format(r_classes), "wb"))

    #save to storage
    blob = bucket.blob("chatbot/{}".format(r_words))
    if blob.exists():
      err.append("{} zaten var.".format(r_words))
    else:
      blob.upload_from_filename("/tmp/{}".format(r_words))

    blob = bucket.blob("chatbot/{}".format(r_classes))
    if blob.exists():
      err.append("{} zaten var.".format(r_classes))
    else:
      blob.upload_from_filename("/tmp/{}".format(r_classes))

    

    training = []
    output_empty = [0] *len(classes)

    for document in documents:
      bag = []
      word_patterns = document[0]
      word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
      for word in words:
        if word in words:
          bag.append(1) if word in word_patterns else bag.append(0)
      
      output_row = list(output_empty)
      output_row[classes.index(document[1])] = 1
      training.append([bag, output_row])


    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation= "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD( lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1 )
    model.save("/tmp/{}".format(r_model), hist)

    #save model to storage
    blob = bucket.blob("chatbot/{}".format(r_model))
    blob.upload_from_filename("/tmp/{}".format(r_model))

    return jsonify({"status": "OK"}) 
  except Exception as e:
    return jsonify({"error": str(e), "source": err})  
