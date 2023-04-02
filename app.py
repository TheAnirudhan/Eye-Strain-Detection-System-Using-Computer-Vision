from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    
    return render_template('outlook.html')

@app.route('/html1')
def about():
    return render_template('html1.html')

@app.route('/html2')
def about():
    return render_template('html2.html')

@app.route('/html3')
def about():
    return render_template('html3.html')

@app.route('/html4')
def about():
    return render_template('html4.html')