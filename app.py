import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

# -------------------------------------
data_path = "maintanance_Intent.json"
## Reading json data
json_data = open(data_path,encoding="utf-8").read()
data = json.loads(json_data)
data = data['intents']

## Creating dataframe

df = pd.DataFrame(columns=["intent","text",'response'])

for i in data:
  intent =i['intent']
  for t,r in zip(i['text'],i['responses']):
    row = {'intent':intent,'text':t,'response':r}
    df.loc[len(df)] = row

### Cosine distance for similarity of texts
def cosine_distance_countvectorizer_method(s1, s2):
    
    # sentences to list
    allsentences = [s1 , s2]
    
    # text to vector
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    
    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    return round((1-cosine),2)

### finding response
def response(text):
  maximum = float('-inf')
  response = ""
  closest = ""
  for i in df.iterrows():
    sim = cosine_distance_countvectorizer_method(text,i[1]['text'])
    if sim > maximum:
      maximum = sim
      response = i[1]['response']
      closest = i[1]['text']
  return response
# ----------------------------------------------
@app.route('/search', methods=['POST'])
def search():
    query = request.form['search_query']
    resp = response(query)
    responses = resp.split(",")
    
    return render_template('results.html', results=responses)



if __name__ == '__main__':
    app.run(debug=True)
