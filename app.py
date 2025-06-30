from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
from good_news_pipeline import generate_daily_good_news
from flask import request
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# You can keep results in memory or save/load from disk
latest_data = {}
last_run_date = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.route("/api/today")
def get_today_news():
    global latest_data, last_run_date

    country = request.args.get("country", "global")
    personality = request.args.get("personality", "darth_vader")
    # Only refresh once per day
    today_str = datetime.now().strftime("%Y-%m-%d")
    if last_run_date != today_str:
        print("Generating fresh news data...")
        latest_data = generate_daily_good_news(
            api_key=OPENAI_API_KEY,
            country=country,
            personality=personality,
            max_articles=3,
            generate_images=True
        )
        last_run_date = today_str
    else:
        print("Serving cached data.")

    return jsonify(latest_data)

@app.route("/api/images/<path:filename>")
def get_image(filename):
    return send_from_directory("static/generated_images", filename)

if __name__ == "__main__":
    app.run(debug=True)