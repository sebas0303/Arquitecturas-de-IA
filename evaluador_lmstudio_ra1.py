import sys
from openai import OpenAI

# 1. Configuración del cliente para redirigir las peticiones a LM Studio
# LM Studio emula la API de OpenAI, por lo que api_key puede ser cualquier string arbitrario.
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def probar_clasificacion_moderacion():
    """
    Simulación de tarea NLU / Clasificación (asociada tradicionalmente a Encoders):
    Enviamos un prompt estructurado al modelo local en LM Studio para evaluar el sentimiento o moderación.
    """
    print("\n--- 1. PRUEBA DE CLASIFICACIÓN Y MODERACIÓN (NLU via LM Studio) ---")
    comentario_usuario = "La aplicación se cierra inesperadamente al intentar exportar un PDF. ¡Pésima experiencia!"
    prompt_sistema = (
        "Eres un sistema automatizado de moderación y triaje de soporte técnico. "
        "Analiza el comentario del usuario y responde ÚNICAMENTE con una línea en el siguiente formato:\n"
        "CATEGORIA: [ERROR_CRITICO | SUGERENCIA | CONSULTA] | SENTIMIENTO: [POSITIVO | NEUTRO | NEGATIVO]"
    )
    try:
        response = client.chat.completions.create(
            model="local-model", # LM Studio usará el modelo que esté cargado actualmente
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": comentario_usuario}
            ],
            temperature=0.1 # Temperatura baja para asegurar una respuesta determinista
        )
        resultado = response.choices[0].message.content
        print(f"Comentario recibido: '{comentario_usuario}'")
        print(f"Salida del Modelo Local: {resultado}")
    except Exception as e:
        print(f"Error al conectar con LM Studio: {e}")
        print("Comprueba que el servidor local de LM Studio esté iniciado en http://localhost:1234")

def probar_generacion_streaming():
    """
    Tarea de Generación de Texto / NLG (asociada a modelos autorregresivos/Decoders):
    Consumimos la respuesta de LM Studio token a token en tiempo real (Stream) utilizando el patrón de generación iterativo.
    """
    print("\n--- 2. PRUEBA DE GENERACIÓN CONTINUA CON STREAMING (NLG via LM Studio) ---")
    prompt_usuario = "Escribe una función corta en Kotlin para Android que realice una petición HTTP GET asíncrona."
    try:
        # Activamos stream=True para recibir los tokens según son generados por el modelo GGUF
        stream_response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "Eres un tutor experto en desarrollo de aplicaciones multiplataforma (DAM)."},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.7,
            stream=True
        )
        print("Respuesta recibida desde LM Studio (en tiempo real):")
        print("--------------------------------------------------")
        for chunk in stream_response:
            # Extraemos el fragmento de texto generado en este instante
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
        print("\n--------------------------------------------------")
    except Exception as e:
        print(f"Error durante el streaming desde LM Studio: {e}")

if __name__ == "__main__":
    print("=========================================================")
    print(" INTEGRACIÓN EN PYTHON CON SERVIDOR LOCAL LM STUDIO ")
    print("=========================================================")
    probar_clasificacion_moderacion()
    probar_generacion_streaming()