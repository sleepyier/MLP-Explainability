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
        
def format_importance(featimport):
    features, importances = zip(*featimport)

    feature_order = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    
    feature_names_mapping = {
        'feat1': 'RiskEstimate',
        'feat2': 'NetFractionRevolvingBurden',
        'feat3': 'AverageMinFile',
        'feat4': 'PercentInstallTrades',
        'feat5': 'NumSatisfactoryTrades'
    }

    ordered_importance = sorted(zip(features, importances), key=lambda x: feature_order.index(x[0].split(' ')[0]))

    table_html = "<table border='1'>\n"
    table_html += "<tr><th>Feature</th><th>Importance</th></tr>\n"

    for feature, importance in ordered_importance:
        
        feature_name = feature.split(' ')[0]
        feature_name = feature_names_mapping.get(feature_name, feature_name)

        table_html += f"<tr><td>{feature_name}</td><td>{importance:.6f}</td></tr>\n"

    table_html += "</table>"
    
    return table_html

        
def plot_feat_importance(feat_importance):
    plt.figure()

    features, importances = zip(*feat_importance)

    plt.barh(features, importances)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = img.read()

    plot_url = base64.b64encode(plot_data).decode('utf8')
    plt.close()
    return f"data:image/png;base64,{plot_url}"

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


@app.route('/', methods = ['GET', 'POST'])
def main():

    default_features = [72, 43, 53, 27, 29]

    if request.method == 'POST':
        feat1 = float(request.form.get('feat1', default_features[0]))
        feat2 = float(request.form.get('feat2', default_features[1]))
        feat3 = float(request.form.get('feat3', default_features[2]))
        feat4 = float(request.form.get('feat4', default_features[3]))
        feat5 = float(request.form.get('feat5', default_features[4]))
    else:
        feat1, feat2, feat3, feat4, feat5 = default_features

    user_features = np.array([feat1, feat2, feat3, feat4, feat5])

    with torch.no_grad():
        _, _, final_output = model(torch.tensor(user_features, dtype=torch.float32))
        prediction = final_output.numpy()
        final_prediction = (prediction > 0.5).astype(int)[0]

    if final_prediction == 0:
        final_prediction = "Bad Credit Risk"
    else:
        final_prediction = "Good Credit Risk"

    # Layer explanations
    exp_out1 = explainer.explain_instance(user_features, lambda x: predict_layer_output(x, model, layer_idx=0))
    feature_importances_layer1 = exp_out1.as_list()
    local_prediction_layer1 = exp_out1.local_pred
    predicted_proba_layer1 = exp_out1.predict_proba

    exp_out2 = explainer.explain_instance(user_features, lambda x: predict_layer_output(x, model, layer_idx=1))
    feature_importances_layer2 = exp_out2.as_list()
    local_prediction_layer2 = exp_out2.local_pred
    predicted_proba_layer2 = exp_out2.predict_proba

    exp_out3 = explainer.explain_instance(user_features, lambda x: predict_layer_output(x, model, layer_idx=2))
    feature_importances_output = exp_out3.as_list()
    local_prediction_output = exp_out3.local_pred
    predicted_proba_output = exp_out3.predict_proba

    layer1import = format_importance(feature_importances_layer1)
    layer2import = format_importance(feature_importances_layer2)
    outimport = format_importance(feature_importances_output)

    predicted_proba_layer1 = f"{max(predicted_proba_layer1) * 100:.2f}"
    predicted_proba_layer2 = f"{max(predicted_proba_layer2) * 100:.2f}"
    predicted_proba_output = f"{predicted_proba_output[0] * 100:.2f}"

    local_prediction_layer1 = f"{local_prediction_layer1[0]:.2f}"
    local_prediction_layer2 = f"{local_prediction_layer2[0]:.2f}"
    local_prediction_output = f"{local_prediction_output[0]:.2f}"


    return render_template('main.html', layer1_import=layer1import, layer2_import=layer2import, output_import=outimport, confidence1=predicted_proba_layer1, confidence2=predicted_proba_layer2, confidenceout=predicted_proba_output, localpred1=local_prediction_layer1, localpred2=local_prediction_layer2, localpredout=local_prediction_output, prediction=final_prediction, feat1=feat1, feat2=feat2, feat3=feat3, feat4=feat4, feat5=feat5)

## THIS IS SOME OF THE TESTING STUFF I AM DOING BELOW

@app.route('/main-test', methods = ['GET', 'POST'])
def maintest():
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
    
        #plot_url1 = plot_feat_importance(feature_importances_layer1)
        #plot_url2 = plot_feat_importance(feature_importances_layer2)
        #plot_url3 = plot_feat_importance(feature_importances_output)
        #plots = plot_url1, plot_url2, plot_url3


        return render_template('main-test.html', plot1 = local_prediction_layer1, plot2 = local_prediction_layer2, plot3 = local_prediction_output )
    else:
        return render_template('main-test.html')
if __name__ == '__main__':
    app.run(debug=True)
        
