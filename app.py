from flask import Flask, request, jsonify
from predictPipeline import pipePredict

# create flask app
app = Flask(__name__)


@app.post('/analyze')
def analyze():
    """Analyze the data specified through json request

    Returns:
        Response: Result of the analysis/prediction
    """
    data = request.json
    try:
        sample = data['text']
    except KeyError:
        return jsonify({'error1': 'Key Error'})

    y = pipePredict(sample)
    try:
        result = jsonify(y[0])
    except TypeError as e:
        result = jsonify({'error2': str(e)})
    return result


# run flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
