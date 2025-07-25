from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('heart.pkl', 'rb'))  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input values from the form
            cp = int(request.form['cp'])
            thalach = int(request.form['thalach'])
            ca = int(request.form['ca'])
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            slope = int(request.form['slope'])

            # Create a feature array (important: order matters!)
            features = np.array([[cp, thalach, ca, age, sex, trestbps, chol, fbs, restecg, slope]])

            # Make the prediction
            prediction = model.predict(features)[0]  # Get the prediction value

            # Interpret the prediction (adjust as needed for your model)
            result = "Heart disease detected" if prediction == 1 else "No heart disease detected"

            return render_template('result.html', result=result)

        except ValueError:  # Handle invalid input
            return render_template('index.html', error="Invalid input. Please enter numbers.")
        except Exception as e: # Catch other exceptions
            return render_template('index.html', error=f"An error occurred: {e}")

    return render_template('index.html')  # Initial GET request


if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production