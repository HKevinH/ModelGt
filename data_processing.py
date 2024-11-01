import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import resample

def load_and_process_data(file_path1, file_path2):
    # Carga de datos
    data_sivigila = pd.read_csv(file_path1, encoding='latin1', sep=';')

    # Definición de columnas importantes
    all_columns = [
        'codigo_sspd', 'fecha_nototificacion', 'edad', 'nacionalidad', 'sexo', 'comuna',
        'tipo_seguridad_social', 'etnia', 'estrato', 'ciudad_residencia', 'dpto_residencia',
        'fecha_consulta', 'fecha_inicio_sintomas', 'tipo_caso', 'Hospitalizado',
        'fecha_de_hospitalizacion', 'deplazamiento_ultimos_15_dias', 'municipio_desplazamiento',
        'fiebre', 'cefalea', 'dolor_retro_ocular', 'malgias', 'artralgia', 'erupcion',
        'dolor_abdominal', 'vomito', 'diarrea', 'somnolencia', 'hipotension', 'hepatomegalia',
        'hem_mucosa', 'hipotermia', 'aum_hemato', 'caida_plaquetas', 'acumulacion_liquidos',
        'extravasacion', 'hemorragia', 'choque', 'daño_organo', 'clasificacion_final',
        'conducta', 'nombre_municipio_procedencia'
    ]

    # Columnas de síntomas y ubicación
    symptom_columns = [
        'fiebre', 'cefalea', 'dolor_retro_ocular', 'malgias', 'artralgia', 'erupcion',
        'dolor_abdominal', 'vomito', 'diarrea', 'somnolencia', 'hipotension', 'hepatomegalia',
        'hem_mucosa', 'hipotermia', 'aum_hemato', 'caida_plaquetas', 'acumulacion_liquidos',
        'extravasacion', 'hemorragia', 'choque', 'daño_organo'
    ]
    location_columns = ['ciudad_residencia', 'dpto_residencia', 'nombre_municipio_procedencia']
    target_column = 'clasificacion_final'

    # Filtrado, normalización y procesamiento
    data_filtered = data_sivigila[all_columns].applymap(lambda s: s.lower() if isinstance(s, str) else s)
    data_filtered = data_filtered.replace({'sí': 1, 'si': 1, 'no': 0})
    data_filtered[symptom_columns] = data_filtered[symptom_columns].apply(pd.to_numeric, errors='coerce')
    data_filtered = data_filtered.dropna(subset=symptom_columns + [target_column] + location_columns)

    # Codificación de la variable objetivo
    label_encoder = LabelEncoder()
    data_filtered[target_column] = label_encoder.fit_transform(data_filtered[target_column])

    # Definir columnas categóricas y codificarlas
    categorical_columns = location_columns + ['sexo', 'tipo_seguridad_social', 'etnia', 'estrato', 'nacionalidad']
    data_filtered[categorical_columns] = data_filtered[categorical_columns].astype(str)
    
    # Aplicar OneHotEncoding para variables categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cat = encoder.fit_transform(data_filtered[categorical_columns])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenar datos codificados y los datos de síntomas en X y definir y
    X = pd.concat([data_filtered[symptom_columns].reset_index(drop=True), encoded_cat_df], axis=1)
    y = data_filtered[target_column]

    return X, y, label_encoder, encoder

def balance_data(X, y):
    # Balanceo de clases mediante upsampling
    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()
    
    # Separar clases mayoritaria y minoritaria
    majority = X[y == majority_class]
    minority = X[y == minority_class]
    
    # Upsampling de la clase minoritaria para igualar la clase mayoritaria
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    
    # Concatenar clases balanceadas y retornar X y y balanceados
    X_balanced = pd.concat([majority, minority_upsampled])
    y_balanced = pd.Series([majority_class] * len(majority) + [minority_class] * len(minority_upsampled))
    
    return X_balanced, y_balanced
