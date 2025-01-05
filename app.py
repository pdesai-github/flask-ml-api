from flask import Flask, request
from flask_cors import CORS
import os
from label_ml import get_label


app = Flask(__name__)

CORS(app)

@app.route('/label',methods=["POST"])
def get_label_from_feature() -> str:

    data = request.get_json()
    feature = data["feature"];
    print(f"[Feature] {feature}")

    label = get_label(feature)
    return label;

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
