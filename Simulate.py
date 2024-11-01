import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import resample
from data_processing import load_and_process_data
import joblib

def predict_dengue(model, label_encoder, encoder, symptom_columns, categorical_columns):
    print("\nIngrese los síntomas y datos del paciente:")
    user_input = {}

    # Ingresar síntomas (sí/no)
    for symptom in symptom_columns:
        while True:
            value = input(f"¿El paciente presenta {symptom.replace('_', ' ')}? (sí/no): ").strip().lower()
            if value in ['sí', 'si', 'no']:
                user_input[symptom] = 1 if value in ['sí', 'si'] else 0
                break
            else:
                print("Entrada inválida. Por favor, ingrese 'sí' o 'no'.")

    # Ingresar datos categóricos (ubicación, etc.)
    for cat in categorical_columns:
        value = input(f"Ingrese {cat.replace('_', ' ')}: ").strip().lower()
        user_input[cat] = value

    # Convertir a DataFrame
    input_df = pd.DataFrame([user_input])

    # Codificar variables categóricas usando el encoder entrenado
    input_cat_encoded = encoder.transform(input_df[categorical_columns])
    input_cat_df = pd.DataFrame(input_cat_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Combinar datos de síntomas y datos codificados en un solo DataFrame
    input_final = pd.concat([input_df[symptom_columns].reset_index(drop=True), input_cat_df], axis=1)
    
    # Asegurar que las columnas estén en el mismo orden que las del modelo
    input_final = input_final.reindex(columns=X.columns, fill_value=0)

    # Realizar la predicción
    prediction = model.predict(input_final)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    
    print(f"\nLa clasificación del paciente es: {prediction_label}")


if __name__ == "__main__":
    # Cargar modelo y datos
    model = joblib.load('gb_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    encoder = joblib.load('onehot_encoder.pkl')
    X, y, label_encoder, encoder = load_and_process_data('bd_denge1.csv', 'bd_dengue.csv')

    # Definir columnas de síntomas y categóricas
    symptom_columns = [
        'fiebre', 'cefalea', 'dolor_retro_ocular', 'malgias', 'artralgia', 'erupcion',
        'dolor_abdominal', 'vomito', 'diarrea', 'somnolencia', 'hipotension', 'hepatomegalia',
        'hem_mucosa', 'hipotermia', 'aum_hemato', 'caida_plaquetas', 'acumulacion_liquidos',
        'extravasacion', 'hemorragia', 'choque', 'daño_organo'
    ]
    categorical_columns = ['ciudad_residencia', 'dpto_residencia', 'nombre_municipio_procedencia', 'sexo', 'tipo_seguridad_social', 'etnia', 'estrato', 'nacionalidad']

    # Realizar predicción
    predict_dengue(model, label_encoder, encoder, symptom_columns, categorical_columns)