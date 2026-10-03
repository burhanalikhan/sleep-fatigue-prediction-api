from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load the complete ML pipeline
model = joblib.load("sleep_fatigue_model.pkl")


# The exact features used during model training
FEATURES = [
    "age",
    "gender",
    "occupation_type",
    "chronotype",
    "bedtime_phone_minutes",
    "primary_bedtime_app",
    "screen_brightness_pct",
    "blue_light_filter_active",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
    "morning_alarm_snoozes"
]


@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sleep Fatigue Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 700px;
                margin: 40px auto;
                padding: 20px;
            }

            h1 {
                text-align: center;
            }

            .description {
                text-align: center;
                color: #555;
                margin-bottom: 30px;
            }

            label {
                display: block;
                margin-top: 15px;
                font-weight: bold;
            }

            input, select {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                box-sizing: border-box;
            }

            button {
                width: 100%;
                padding: 12px;
                margin-top: 25px;
                cursor: pointer;
                font-size: 16px;
            }

            #result {
                margin-top: 25px;
                padding: 15px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
            }
        </style>
    </head>

    <body>

        <h1>Sleep Fatigue Predictor</h1>

        <p class="description">
            Enter bedtime behavior and lifestyle information
            to predict next-day fatigue.
        </p>

        <form id="predictionForm">

            <label>Age</label>
            <input type="number" id="age" min="1" max="100" required>

            <label>Gender</label>
            <select id="gender" required>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>

            <label>Occupation Type</label>
            <select id="occupation_type" required>
                <option value="Student">Student</option>
                <option value="Healthcare / Shift Worker">Healthcare / Shift Worker</option>
                <option value="Remote Tech">Remote Tech</option>
                <option value="Freelance / Creative">Freelance / Creative</option>
                <option value="Corporate 9-to-5">Corporate 9-to-5</option>
                
            </select>

            <label>Chronotype</label>
            <select id="chronotype" required>
                <option value="Morning Lark">Morning Lark</option>
                <option value="Intermediate">Intermediate</option>
                <option value="Night Owl">Night Owl</option>
            </select>

            <label>Bedtime Phone Minutes</label>
            <input type="number"
                   id="bedtime_phone_minutes"
                   min="0"
                   required>

            <label>Primary Bedtime App</label>
            <select id="primary_bedtime_app" required>
                <option value="TikTok / Reels">TikTok / Reels</option>
                <option value="YouTube">YouTube</option>
                <option value="Instagram / Reddit">Instagram / Reddit</option>
                <option value="Streaming (Netflix/Hulu)">
                    Streaming (Netflix/Hulu)
                </option>
                <option value="Messaging / Chat">Messaging / Chat</option>
                <option value="News / Reading">News / Reading</option>
            </select>

            <label>Screen Brightness (%)</label>
            <input type="number"
                   id="screen_brightness_pct"
                   min="0"
                   max="100"
                   step="0.1"
                   required>

            <label>Blue Light Filter Active</label>
            <select id="blue_light_filter_active" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>

            <label>Caffeine After 5 PM (mg)</label>
            <input type="number"
                   id="caffeine_post_5pm_mg"
                   min="0"
                   required>

            <label>Physical Activity (minutes)</label>
            <input type="number"
                   id="physical_activity_min"
                   min="0"
                   required>

            <label>Morning Alarm Snoozes</label>
            <input type="number"
                   id="morning_alarm_snoozes"
                   min="0"
                   required>

            <button type="submit">
                Predict Fatigue
            </button>

        </form>

        <div id="result"></div>

        <script>

        document.getElementById("predictionForm").addEventListener(
            "submit",
            async function(event) {

                event.preventDefault();

                const data = {
                    age: Number(document.getElementById("age").value),

                    gender:
                        document.getElementById("gender").value,

                    occupation_type:
                        document.getElementById("occupation_type").value,

                    chronotype:
                        document.getElementById("chronotype").value,

                    bedtime_phone_minutes:
                        Number(
                            document.getElementById(
                                "bedtime_phone_minutes"
                            ).value
                        ),

                    primary_bedtime_app:
                        document.getElementById(
                            "primary_bedtime_app"
                        ).value,

                    screen_brightness_pct:
                        Number(
                            document.getElementById(
                                "screen_brightness_pct"
                            ).value
                        ),

                    blue_light_filter_active:
                        Number(
                            document.getElementById(
                                "blue_light_filter_active"
                            ).value
                        ),

                    caffeine_post_5pm_mg:
                        Number(
                            document.getElementById(
                                "caffeine_post_5pm_mg"
                            ).value
                        ),

                    physical_activity_min:
                        Number(
                            document.getElementById(
                                "physical_activity_min"
                            ).value
                        ),

                    morning_alarm_snoozes:
                        Number(
                            document.getElementById(
                                "morning_alarm_snoozes"
                            ).value
                        )
                };

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                const resultDiv =
                    document.getElementById("result");

                if (response.ok) {

                    resultDiv.innerHTML =
                        "Predicted Next-Day Fatigue: " +
                        result.predicted_fatigue_score;

                } else {

                    resultDiv.innerHTML =
                        "Error: " + result.error;
                }
            }
        );

        </script>

    </body>
    </html>
    """)


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Request must contain JSON data."
            }), 400

        # Check for missing features
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features.",
                "missing_features": missing_features
            }), 400

        # Keep only the features expected by the model
        input_data = pd.DataFrame(
            [data],
            columns=FEATURES
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_fatigue_score": round(
                float(prediction),
                2
            )
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model": "sleep_fatigue_model.pkl"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )