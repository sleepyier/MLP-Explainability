from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import lime
import lime.lime_tabular


class MLPNet(nn.Module): 
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out1 = self.tanh(self.fc1(x))
        out2 = self.tanh(self.fc2(out1))
        out3 = torch.sigmoid(self.fc3(out2))
        return out1, out2, out3


app = Flask(__name__)

model = MLPNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()


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




@app.route('/main', methods = ['GET', 'POST'])
def main():
    result = None
    if request.method == 'POST':
        feat1 = float(request.form.get('feat1'))
        feat2 = float(request.form.get('feat2'))
        feat3 = float(request.form.get('feat3'))
        feat4 = float(request.form.get('feat4'))
        feat5 = float(request.form.get('feat5'))

        print(feat1)

        user_features = [[feat1, feat2, feat3, feat4, feat5]]

        with torch.no_grad():
            prediction = model(torch.tensor(user_features, dtype=torch.float32))


    return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True)