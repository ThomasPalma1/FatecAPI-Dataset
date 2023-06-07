from flask import Flask
from jinja2 import Template

app = Flask(__name__)


@app.route('/')
def start_application():
    return Template("Hello, World!").render()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
