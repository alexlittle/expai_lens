from flask import Flask, send_from_directory
import json

app = Flask(__name__)

@app.route("/data/scalars")
def get_scalars():
    with open("expai_lens_logs/scalars.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    return {"data": data}

@app.route("/")
def index():
    return send_from_directory("static", "index.html")