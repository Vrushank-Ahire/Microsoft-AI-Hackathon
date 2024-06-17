from flask import Flask, render_template, request,url_for
import os
# import joblib
import re
import string
import pandas as pd
import torch
import pickle
from transformers import AutoModel, AutoTokenizer
# Load the trained model from the pickle file

app = Flask(__name__)

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)

        layers = []
        hidden_sizes = [input_size] + hidden_sizes

        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.Conv1d(in_channels=hidden_sizes[i], out_channels=hidden_sizes[i + 1], kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])

        self.layers = nn.Sequential(*layers)
        self.logit = nn.Linear(hidden_sizes[-1], num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        # Input shape: [batch_size, sequence_length, hidden_size]
        input_rep = input_rep.transpose(1, 2)  # Change to [batch_size, hidden_size, sequence_length]
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        last_rep = torch.mean(last_rep, dim=-1)  # Aggregate along sequence length
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs

with open('tut1\models\model.pkl', 'rb') as f:
    Model = pickle.load(f)

# Model = joblib.load('models/model.pkl')


def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model")
def model():
    return render_template("textmodel.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/upload_text', methods=['POST'])
def upload_csv():
    if 'fileUpload' in request.files:
        file = request.files['fileUpload']
        if file.filename != '':
            file.save(os.path.join('data', file.filename))
            return render_template('response.html')
    return "No text file uploaded!"

@app.route('/upload_text', methods=['POST'])
def pre():
    if request.method == 'POST':
        txt = request.form['txt']
        txt = wordpre(txt)
        # txt = pd.Series(txt)

        model_name = "bert-base-cased"
        transformer = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_seq_length = 300
        encoded_sent = tokenizer.encode(txt, add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
        
        input_ids_test = torch.tensor(encoded_sent)
        model_outputs = transformer(input_ids_test)
        hidden_states = model_outputs.last_hidden_state
        _, _, probability = Model(hidden_states)
        return render_template("textmodel.html", result = probability)
    else:
        return '' 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
