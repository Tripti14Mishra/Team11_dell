from flask import Flask, render_template, request


import os
app = Flask(__name__)

import json

import pandas as pd
import csv


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    c=[]
    with open("../bootstrap-ecommerce-uikit/ui-ecommerce/pref.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:          
            c.append(lines)

        
    df = pd.DataFrame(c, columns= [userText])
    export_csv = df.to_csv ('../bootstrap-ecommerce-uikit/ui-ecommerce/pref.csv', index = None, header=True)
    return ""
   
    
    


if __name__ == "__main__":
    app.run(port=4000)
