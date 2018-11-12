import os
# from django.shortcuts import redirect
from flask import Flask, request, render_template, send_from_directory, flash, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys   
from flask import render_template_string
 


__author__ = 'TranTung'
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/describle")
def describle():
    return render_template("describle.html")

@app.route("/upload_file")
def upload_file():
    return render_template("upload.html")

@app.route("/upload", methods=['POST', 'GET'])
def upload(): 
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if ext in [".jpg", ".png", ".csv"]: 
        # if (ext == ".jpg") or (ext == ".png") or (ext == ".csv"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination) 
    return render_template("complete.html")



@app.route('/show')
def show():
    data = pd.read_csv("./images/2015.csv", encoding ='latin1')
    return data.to_html(border=1)

@app.route('/show2')
def show2():
    data = pd.read_csv("./images/2015.csv", encoding ='latin1')  
    value = render_template_string(data.to_html(border=1), context );
    return render_template("chart.html", value = value)

@app.route('/chart')
def chart():
    return render_template("chart.html")


if __name__ == "__main__":
    app.run(host='localhost', debug=True)
