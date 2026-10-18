import sys
from transformers import pipeline

def probar_encoder_moderacion():
    """
    Uso de arquitectura Encoder-Only (BERT / DistilBERT) para tarea
    discriminativa (análisis de sentimiento / moderación)
    Criterio RA1: c) Distingue modelos Encoder y e) Identifica frameworks.
    """
    print("\n--- 1. PRUEBA ENCODER-ONLY: Clasificación / NLU ---")
    # Cargamos un pipeline de clasificación de texto (Encoder-Only)
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    comentario = "This application crashes constantly and the user interface is terrible."
    resultado = classifier(comentario)
    print(f"Texto analizado: '{comentario}'")
    print(f"Resultado del Encoder (Clase y Confianza): {resultado}")

def probar_decoder_generacion():
    """
    Uso de arquitectura Decoder-Only para tarea generativa (autorregresiva)
    Criterio RA1: c) Distingue modelos Decoder y d) Particularidades generativas.
    """
    print("\n--- 2. PRUEBA DECODER-ONLY: Generación de Texto / NLG ---")
    # Cargamos un pipeline de generación de texto (Decoder-Only)
    # Usamos un modelo ligero e instructivo como Qwen2.5-0.5B-Instruct
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    prompt = (
        "<|im_start|>user\n"
        "Escribe un mensaje corto de bienvenida para un usuario que acaba de "
        "registrarse en nuestra app multiplataforma.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    resultado = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        truncation=True
    )
    print("Salida generada por el Decoder:")
    print(resultado[0]["generated_text"])

if __name__ == "__main__":
    print("=== TALLER PRÁCTICO RA1: COMPARATIVA ENCODER VS DECODER ===")
    try:
        probar_encoder_moderacion()
        probar_decoder_generacion()
    except Exception as e:
        print(f"Error durante la ejecución del script: {e}")