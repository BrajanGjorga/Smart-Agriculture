from flask import Flask, render_template, request
import requests


api_key = "KGAT_1afb81b502f85ae1e01f5e6db37dbd2b"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("wisam1985/iot-agriculture-2024")

print("Path to dataset files:", path)

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, welcome to the Smart Agriculture API!"


if __name__ == '__main__':
    app.run(debug=True)



    
