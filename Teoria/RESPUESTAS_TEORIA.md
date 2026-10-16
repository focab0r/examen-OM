## Parte A. Conceptos básicos de MLOps y MLflow (3 puntos)

Seis preguntas de 0,5 puntos. Tiempo máximo: 30 minutos. No se pide código. Formatos: verdadero/falso, completar hueco, tipo test y emparejar.

- En A1 escribe Verdadero o Falso. No se pide explicación adicional.
- En A2, A3 y A6 indica una sola opción (a, b, c o d).
- En A4 y A5 escribe las correspondencias de cada elemento.
- Escribe todas las respuestas A1-A6 en `RESPUESTAS_TEORIA.md`.

### A1. Evaluación con datos nuevos - verdadero/falso (0,5 puntos)

Indica si la siguiente afirmación es verdadera o falsa: **FALSO**

"Si un modelo se entrena y se evalúa con los mismos datos, la métrica obtenida refleja bien cómo se comportará con datos nuevos."

### A2. Estados de un modelo registrado - completar hueco (0,5 puntos)

¿Qué etapa falta en este proceso?

Se registra una versión nueva del modelo → se valida en Staging → **(a)** → se archiva cuando la sustituye otra versión.

a) Se promueve a Production.

b) Se borra su run de origen.

c) Se reentrena usando el test.

d) Se le cambia el nombre al modelo.

### A3. Deriva de datos - tipo test (0,5 puntos)

¿Qué es el data drift?

a) Que el modelo tarda más en responder.

b) Un error al registrar el modelo en MLflow.

c) Que los datos de entrada en producción cambian respecto a los de entrenamiento.**CORRECTA**

d) Un hiperparámetro mal elegido.

### A4. Entorno reproducible - emparejar (0,5 puntos)

Empareja cada archivo con su función: `pyproject.toml` / `uv.lock`.

a) Fija las versiones exactas que se instalaron. --> `uv.lock`

b) Declara las dependencias del proyecto. --> `pyproject.toml`

### A5. Métricas de clasificación - emparejar (0,5 puntos)

Empareja cada métrica con lo que mide: precisión / recall / F1.

a) De los casos positivos reales, cuántos detecta el modelo. --> `recall`

b) De las predicciones positivas, cuántas son correctas. --> `precisión`

c) Combina precisión y recall en un solo número. --> `F1`

### A6. Rendimiento que cae en producción - tipo test (0,5 puntos)

Un modelo entrenado con datos antiguos va perdiendo calidad poco a poco en producción. ¿Qué es lo más razonable?

a) Subir el umbral de decisión hasta que la métrica parezca buena.

b) Borrar el modelo y no desplegar ninguno.

c) Evaluarlo otra vez solo con los datos de entrenamiento.

d) Comprobar si hay deriva de datos y, si procede, reentrenar con datos recientes y registrar una versión nueva. **CORRECTA**