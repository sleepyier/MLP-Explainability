from flask import Flask, render_template, request
#import pandas as pd
#import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/test', methods = ['GET', 'POST'])
def test():
    result = None
    if request.method == 'POST':
        number1 = int(request.form.get('number1'))
        number2 = int(request.form.get('number2'))

        sum_value = number1 + number2

        if sum_value % 2 == 0:
            result = 'This is an even sum, {}. The first number was {} and the second number was {}'.format(sum_value, number1, number2) 
        elif sum_value % 2 == 1:
            result = 'This is an odd sum, {}. The first number was {} and the second number was {}'.format(sum_value, number1, number2)

    return render_template('test.html', result = result)

if __name__ == '__main__':
    app.run(debug=True)

