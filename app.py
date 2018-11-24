import os
# from django.shortcuts import redirect
from flask import Flask, request, render_template, send_from_directory, flash, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys   
from flask import render_template_string 
from wtforms import Form, TextField

__author__ = 'TranTung'
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# class Form(FlaskForm):
#      column_name = SelectField('column_name', choices[('1','2','3')])



# =========================================FUNCTION===========================================================
# read data
def read_data():
    data = pd.read_csv("./dataset/2017.csv", encoding ='latin1')  
    return data

# get data  use]
def get_data_use():
    data = read_data()
    data_use = data[['1st Road Class & No','Road Surface','Lighting Conditions','Weather Conditions','Type of Vehicle', 'Casualty Severity']]
    return data_use

# hàm gọi các tên column
def list_columns_name():
    data_use = get_data_use()
    list_column_name = data_use.columns 
    return list_column_name

# get shape data use
def getShape():
    data_use = get_data_use()
    shape = data_use.shape
    return shape

# khởi tạo dictionary rỗng
def dic_column_missing():
    dic_column_missing = {}
    return dic_column_missing

# =========================================ROUTE===========================================================
#route home -----------------------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")


#route describle -----------------------------------------------------------------------------------------------
@app.route("/describle")
def describle():
    return render_template("describle.html")

#route upload_file -----------------------------------------------------------------------------------------------
@app.route("/upload_file")
def upload_file():
    return render_template("upload.html")

#route upload -----------------------------------------------------------------------------------------------
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


 
#route showw data -----------------------------------------------------------------------------------------------
@app.route('/show')
def show(): 
    return render_template("show.html", df_view = read_data())

#route show data use -----------------------------------------------------------------------------------------------
@app.route('/show_use')
def show_use(): 
    return render_template("show_use.html", df_view = get_data_use())

#route chart -----------------------------------------------------------------------------------------------
@app.route('/chart')
def chart():
    return render_template("chart.html")

#route preprocess -----------------------------------------------------------------------------------------------
@app.route('/preprocess')
def preprocess():  
    return render_template("pre_process.html", list_column_name = list_columns_name(),
                             dic_column_missing= dic_column_missing(),
                             len=len(list_columns_name()), shape= getShape())


#route check_missing -----------------------------------------------------------------------------------------------
@app.route('/check_missing', methods=['POST'] )
def check_missing(): 
    data_use = get_data_use()
    list_column_name = list_columns_name()
    dic_column_missing = {}
    sum_missing = []
    # nhận dữ liệu từ form submit
    if request.method == 'POST':
        for i in range(1, len(list_column_name)):
            sum = data_use[list_column_name[i]].isnull().sum()
            dic_column_missing[list_column_name[i]] = sum
    return render_template('pre_process.html', dic_column_missing = dic_column_missing, 
        list_column_name = list_columns_name())

if __name__ == "__main__":
    app.run(host='localhost', port=8080, debug=True)
