from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('final_haloc.csv')
X = df.drop('RiskPerformance', axis=1)
y = df['RiskPerformance']
X = X.values
y = y.values
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

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
    
def predict_layer_output(instance, model, layer_idx):
    instance = torch.tensor(instance).float()
    with torch.no_grad():
        out1, out2, out3 = model(instance)
        if layer_idx == 0:
            return out1.detach().numpy()
        elif layer_idx == 1:
            return out2.detach().numpy()
        elif layer_idx == 2:
            out3_np = out3.detach().numpy()
            return np.hstack([1 - out3_np, out3_np])
        
def plot_feat_importance(feat_importance):
    features, importances = zip(*feat_importance)
    plt.barh(features, importances)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importances')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.numpy(),
    feature_names=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
    class_names=['output'],
    mode='classification'
)

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

        user_features = np.array([feat1, feat2, feat3, feat4, feat5])

        with torch.no_grad():
            prediction = model(torch.tensor(user_features, dtype=torch.float32))


        # Layer 1 explanation
        exp_out1 = explainer.explain_instance(user_features, lambda x: predict_layer_output(x, model, layer_idx=0))
        feature_importances_layer1 = exp_out1.as_list()
        local_prediction_layer1 = exp_out1.local_pred
        predicted_proba_layer1 = exp_out1.predict_proba

        # Layer 2 explanation
        exp_out2 = explainer.explain_instance(user_features, lambda x: predict_layer_output(x, model, layer_idx=1))
        feature_importances_layer2 = exp_out2.as_list()
        local_prediction_layer2 = exp_out2.local_pred
        predicted_proba_layer2 = exp_out2.predict_proba

        # Output layer explanation
        exp_out3 = explainer.explain_instance(user_features, lambda x: predict_layer_output(x, model, layer_idx=2))
        feature_importances_output = exp_out3.as_list()
        local_prediction_output = exp_out3.local_pred
        predicted_proba_output = exp_out3.predict_proba

        plot_url1 = plot_feat_importance(feature_importances_layer1)
        plot_url2 = plot_feat_importance(feature_importances_layer2)
        plot_url3 = plot_feat_importance(feature_importances_output)

        return render_template('main.html', plot1 = plot_url1, plot2 = plot_url2, plot3 = plot_url3)
    else:
        return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True)
        
