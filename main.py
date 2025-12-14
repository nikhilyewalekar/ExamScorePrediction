from flask import Flask , Response, jsonify, request, render_template

from src.utils import PredictExamScore

app = Flask(__name__)
obj = PredictExamScore()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predictExamScore():
    data = request.form
    print(data)
    predictedScore = obj.predict_score(data)
    print(predictedScore)
    return f"{predictedScore}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8000, debug=True)