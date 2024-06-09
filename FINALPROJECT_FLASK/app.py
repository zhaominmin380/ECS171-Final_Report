from flask import Flask, request, jsonify, render_template
from model import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.json
        print('Received data:', data)

        input_data = {
            'buying': data['buying'],
            'maint': data['maint'],
            'persons': data['persons'],
            'lug_boot': data['lug_boot'],
            'safety': data['safety']
        }

        # Perform prediction
        prediction = predict(input_data)

        custom_message = f"My suggestion: {prediction}"
        print('Prediction:', custom_message)
        return custom_message
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
