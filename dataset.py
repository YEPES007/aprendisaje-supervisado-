import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Semilla para reproducibilidad
np.random.seed(42)

# Crear un DataFrame con datos simulados
datos = pd.DataFrame({
    'origen': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'destino': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'hora': np.random.choice(['mañana', 'tarde', 'noche'], 100),
    'dia_semana': np.random.choice(['lunes', 'martes', 'miércoles', 'jueves', 'viernes'], 100),
    'tiempo_viaje': np.random.randint(5, 30, 100),
    'congestion': np.random.choice(['bajo', 'medio', 'alto'], 100),
    'ruta_optima': np.random.choice(['sí', 'no'], 100)
})

print("📊 Muestra de los datos simulados:")
print(datos.head())

# Codificar columnas categóricas
le = LabelEncoder()
for col in ['origen', 'destino', 'hora', 'dia_semana', 'congestion', 'ruta_optima']:
    datos[col] = le.fit_transform(datos[col])

# Separar variables (X) y etiqueta (y)
X = datos.drop('ruta_optima', axis=1)
y = datos['ruta_optima']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Predecir y evaluar el modelo
y_pred = modelo.predict(X_test)
print("\n✅ Reporte del modelo:")
print(classification_report(y_test, y_pred))
