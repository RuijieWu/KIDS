'''
Date: 2024-06-13 18:53:57
LastEditTime: 2024-06-13 18:59:37
Description: 
'''
from flask import Flask

app = Flask(__name__)

#* 
#* 
#* 
#* 
#* 
#* 
#* 
#* 

@app.route("/listen")
def listen():
    pass
@app.route("/statics")
def statics():
    pass
@app.route("/danger")
def danger():
    pass
@app.route("/aberration")
def aberration():
    pass

@app.route("/ping")
def ping():
    return "pong!\n"


if __name__ == "__main__":
    app.run("127.0.0.1",7777,debug=True)
