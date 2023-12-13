from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import locale

locale.setlocale(locale.LC_MONETARY, 'pt_BR.utf-8')

app = Flask(__name__)

@app.template_filter('format_currency')
def format_currency(value):
    return locale.currency(value, grouping=True)

random_forest_model = joblib.load('random_forest_model.joblib')

scaler = MinMaxScaler()

training_data = pd.read_csv('df_selic.csv')

training_data['Vencimento do Titulo'] = pd.to_datetime(training_data['Vencimento do Titulo'], format='%d/%m/%Y')
training_data['Data Venda'] = pd.to_datetime(training_data['Data Venda'], format='%d/%m/%Y')
training_data['Vencimento do Titulo'] = training_data['Vencimento do Titulo'].apply(lambda x: x.timestamp())
training_data['Data Venda'] = training_data['Data Venda'].apply(lambda x: x.timestamp())

training_data['PU'] = training_data['PU'].str.replace(',', '.').astype(float)
training_data['Quantidade'] = training_data['Quantidade'].str.replace(',', '.').astype(float)

print(training_data.columns)

scaler.fit(training_data[['Vencimento do Titulo', 'Data Venda', 'PU', 'Quantidade']])

def determine_risk(predicted_value, price_per_unit, quantity):

    calculated_value = price_per_unit * quantity

    if calculated_value == predicted_value:
        return 'Risco Médio'
    elif calculated_value > predicted_value:
        return 'Risco Baixo'
    else:
        return 'Risco Alto'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sale_date = pd.to_datetime(request.form['sale_date'])
        title_expiration = pd.to_datetime(request.form['title_expiration'])
        price_per_unit = float(request.form['price_per_unit'])
        quantity = float(request.form['quantity'])

        sale_date_numeric = sale_date.timestamp()
        title_expiration_numeric = title_expiration.timestamp()

        print("Input Values:", title_expiration_numeric, sale_date_numeric, price_per_unit, quantity)

        input_df = pd.DataFrame([[title_expiration_numeric, sale_date_numeric, price_per_unit, quantity]],
                                columns=['Vencimento do Titulo', 'Data Venda', 'PU', 'Quantidade'])

        features_scaled = scaler.transform(input_df[['Vencimento do Titulo', 'Data Venda', 'PU', 'Quantidade']])
        print("Scaled Features:", features_scaled)

        predicted_value = random_forest_model.predict(features_scaled)[0]
        print("Predicted Value:", predicted_value)

        risk = determine_risk(predicted_value, price_per_unit, quantity)

        return render_template('index.html', predicted_value=predicted_value, risk=risk)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
