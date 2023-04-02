from flask import Flask, render_template, request
from main import *
import time


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('outlook.html')

@app.route('/html1',methods=['POST'])
def f0():
    if request.form['GET REFERENCE']=='getReference':
        getReference()
        camera.release()
    return render_template('html1.html')

@app.route('/html2')
def f1():
    
    return render_template('html2.html')

@app.route('/html3')
def f2():
    return render_template('html3.html')

@app.route('/html4')
def about():
    return render_template('html4.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5500,debug=True)