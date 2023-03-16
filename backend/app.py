from flask import Flask, render_template, request, jsonify

#for image sentiment
import pytesseract
from PIL import Image

#for audio sentiment
import speech_recognition as sr

#for text sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# pytesseract.tesseract_cmd = path_to_tesseract

app = Flask(__name__)

app.config['DEBUG'] = True

sentiment = SentimentIntensityAnalyzer()

#convert image to text
class GetText(object):
    def __init__(self, image):
        self.image = image

    def get_text(self):
        image = Image.open(self.image)
        text = pytesseract.image_to_string(image)
        return text


@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello World'}), 200


@app.route('/text-sentiment', methods=['POST'])
def textSentiment():
    data = request.json['text']
    sentiment_result = sentiment.polarity_scores(data)
    return jsonify({'message': data, 'sentiment': sentiment_result}), 200


@app.route('/audio-sentiment', methods=['POST'])
def audioSentiment():
    transcript = ""
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({'message': 'No file part in the request'}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({'message': 'No file selected for uploading'}), 400

        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)
            sentiment_result = sentiment.polarity_scores(transcript)
            return jsonify({'message': transcript, 'sentiment': sentiment_result}), 200


@app.route('/image-sentiment', methods=['POST'])
def imageSentiment():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return jsonify({'message': 'No file part in the request'}), 400
        photo = request.files['photo']
        textGen = GetText(photo).get_text()
        textGen = textGen.replace('\n', ' ')
        sentiment_result = sentiment.polarity_scores(textGen)
        return jsonify({'message': textGen, 'sentiment': sentiment_result}), 200


if __name__ == '__main__':
    app.run()