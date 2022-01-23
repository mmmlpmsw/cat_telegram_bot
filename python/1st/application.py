# -*- coding: utf-8 -*-
# !pip3 install emoji
# !pip3 install tweet-preprocessor 2>/dev/null 1>/dev/null
# !pip3 install kaggle
# !pip3 install transformers
# !pip3 install deep_translator

import sys

import emoji
import keras
import numpy as np
import pandas as pd
import preprocessor as p
from deep_translator import GoogleTranslator
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from flask import Flask


data = pd.read_csv("text_emotion.csv")

misspell_data = pd.read_csv("aspell.txt", sep=":", names=["correction", "misspell"])
misspell_data.misspell = misspell_data.misspell.str.strip()
misspell_data.misspell = misspell_data.misspell.str.split(" ")
misspell_data = misspell_data.explode("misspell").reset_index(drop=True)
misspell_data.drop_duplicates("misspell", inplace=True)
miss_corr = dict(zip(misspell_data.misspell, misspell_data.correction))

{v: miss_corr[v] for v in [list(miss_corr.keys())[k] for k in range(20)]}


def misspelled_correction(val):
    for x in val.split():
        if x in miss_corr.keys():
            val = val.replace(x, miss_corr[x])
    return val


data["clean_content"] = data.content.apply(lambda x: misspelled_correction(x))

contractions = pd.read_csv("contractions.csv")
cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))


def cont_to_meaning(val):
    for x in val.split():
        if x in cont_dic.keys():
            val = val.replace(x, cont_dic[x])
    return val


data.clean_content = data.clean_content.apply(lambda x: cont_to_meaning(x))

# p.set_options(p.OPT.MENTION, p.OPT.URL)
# p.clean("hello guys @alx #sport🔥 1245 https://github.com/s/preprocessor")

data["clean_content"] = data.content.apply(lambda x: p.clean(x))


def punctuation(val):
    punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''

    for x in val.lower():
        if x in punctuations:
            val = val.replace(x, " ")
    return val


data.clean_content = data.clean_content.apply(lambda x: ' '.join(punctuation(emoji.demojize(x)).split()))


def clean_text(val):
    val = misspelled_correction(val)
    val = cont_to_meaning(val)
    val = p.clean(val)
    val = ' '.join(punctuation(emoji.demojize(val)).split())

    return val


data = data[data.clean_content != ""]

data.sentiment.value_counts()

sent_to_id = {"empty": 0, "sadness": 1, "enthusiasm": 2, "neutral": 3, "worry": 4,
              "surprise": 5, "love": 6, "fun": 7, "hate": 8, "happiness": 9, "boredom": 10, "relief": 11, "anger": 12}

data["sentiment_id"] = data['sentiment'].map(sent_to_id)

# data

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data.sentiment_id)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(data.clean_content, Y, random_state=1995, test_size=0.2,
                                                    shuffle=True)

"""### LSTM <a class="anchor" id="m-l"></a>"""

# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 160
Epoch = 5
token.fit_on_texts(list(X_train) + list(X_test))
X_train_pad = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=max_len)
X_test_pad = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=max_len)

w_idx = token.word_index

embed_dim = 160
lstm_out = 250
batch_size = 32


def translate(text):
    translation = GoogleTranslator(source='auto', target='en').translate(text)
    return translation


def get_sentiment(model, text):
    if not isinstance(text, str):
        return 'neutral'

    if 0 < len(text) < 5000:
        text = translate(text)

    text = clean_text(text)
    # tokenize
    twt = token.texts_to_sequences([text])
    twt = sequence.pad_sequences(twt, maxlen=max_len, dtype='int32')
    sentiment = model.predict(twt, batch_size=1, verbose=2)
    sent = np.round(np.dot(sentiment, 100).tolist(), 0)[0]
    result = pd.DataFrame([sent_to_id.keys(), sent]).T
    result.columns = ["sentiment", "percentage"]
    result = result[result.percentage != 0]
    maximum = 0
    for i in result.index:
        if result['percentage'][i] > maximum:
            maximum = result['percentage'][i]
    return result[result.percentage == maximum]['sentiment'].values[0]  # todo remove comment to get one emotion
    # return result  todo remove comment to get all emotions list


# webapp

app = Flask(__name__)
model1 = keras.models.load_model('train_model1.pb')


@app.route('/', methods=['GET'])
def main():
    from flask import request

    msg = request.args.get('text')
    if msg is None:
        msg = ""

    return get_sentiment(model1, msg)


if __name__ == '__main__':
    app.run()
