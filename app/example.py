from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions
from datetime import datetime
from GD import RL


app = FlaskAPI(__name__)
rl = RL()

@app.route("/", methods=['GET'])
def list():
    x = float(request.args.get('x'))
    return str(rl.predict(x))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
