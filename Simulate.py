import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import resample
from data_processing import load_and_process_data
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def predict_dengue(model, label_encoder, encoder, symptom_columns, categorical_columns, X, y):
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

    # Ingresar datos categóricos
    for cat in categorical_columns:
        value = input(f"Ingrese {cat.replace('_', ' ')}: ").strip().lower()
        user_input[cat] = value

    # Convertir entrada del usuario en DataFrame
    input_df = pd.DataFrame([user_input])

    # Codificar variables categóricas usando el encoder entrenado
    input_cat_encoded = encoder.transform(input_df[categorical_columns])
    input_cat_df = pd.DataFrame(input_cat_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Combinar datos de síntomas y categóricos codificados
    input_final = pd.concat([input_df[symptom_columns].reset_index(drop=True), input_cat_df], axis=1)

    # Asegurar que las columnas coincidan con las del modelo
    input_final = input_final.reindex(columns=X.columns, fill_value=0)

    # Realizar la predicción
    prediction = model.predict(input_final)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    print(f"\nLa clasificación del paciente es: {prediction_label}")

    # Cálculo de métricas con el conjunto completo de datos
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred, average='weighted') * 100
    recall = recall_score(y, y_pred, average='weighted') * 100
    f1 = f1_score(y, y_pred, average='weighted') * 100

    # Crear un DataFrame con las métricas
    metrics_df = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Valor (%)": [accuracy, precision, recall, f1]
    })

    # Mostrar el DataFrame de métricas
    print("\n========== Métricas del Modelo ==========")
    print(metrics_df)

    # Reporte de clasificación
    report = classification_report(y, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    print("\n========== Reporte de Clasificación ==========")
    print(report_df)

    # Matriz de Confusión
    cm = confusion_matrix(y, y_pred)

    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()

    # Retornar métricas como DataFrame (opcional)
    return metrics_df, report_df


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
    metrics_df, report_df = predict_dengue(model, label_encoder, encoder, symptom_columns, categorical_columns, X, y)

    # Mostrar métricas y reporte de clasificación como DataFrames
    print("\n========== Métricas Generales ==========")
    print(metrics_df)

    print("\n========== Reporte de Clasificación ==========")
    print(report_df)