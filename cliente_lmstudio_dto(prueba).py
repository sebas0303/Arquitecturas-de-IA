import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# 1. Definimos el DTO (Data Transfer Object) usando Pydantic
class AnalisisSoporteDTO(BaseModel):
    codigo_ticket: str = Field(description="Código identificador único asignado al problema")
    componente_afectado: str = Field(description="Módulo del sistema (ej. BaseDatos, Autenticacion, UI)")
    nivel_prioridad: str = Field(description="Nivel de prioridad: ALTA, MEDIA o BAJA")
    resumen_tecnico: str = Field(description="Explicación concisa del fallo")
    solucion_propuesta: str = Field(description="Acción recomendada para el equipo de desarrollo")

# 2. Inicializamos el cliente de OpenAI redirigiendo la URL a LM Studio
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio" # LM Studio no requiere una API Key real en local
)

def procesar_log_a_dto(log_incidencia: str) -> AnalisisSoporteDTO:
    """
    Envia un log de error no estructurado a LM Studio y fuerza la respuesta
    en un formato JSON estricto compatible con nuestro AnalisisSoporteDTO.
    """
    prompt_sistema = (
        "Eres un motor automatizado de triaje para soporte técnico de software.\n"
        "Debes analizar la incidencia recibida y responder EXCLUSIVAMENTE con un JSON válido "
        "con la siguiente estructura exacta:\n"
        "{\n"
        ' "codigo_ticket": "TCK-XXXX",\n'
        ' "componente_afectado": "string",\n'
        ' "nivel_prioridad": "ALTA" | "MEDIA" | "BAJA",\n'
        ' "resumen_tecnico": "string",\n'
        ' "solucion_propuesta": "string"\n'
        "}"
    )
    try:
        print("Enviando petición a LM Studio...")
        response = client.chat.completions.create(
            model="local-model", # LM Studio utilizará el modelo cargado actualmente
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Analiza esta incidencia:\n{log_incidencia}"}
            ],
            temperature=0.1, # Temperatura baja para asegurar el determinismo en el formato
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "AnalisisSoporteDTO",
                    "strict": True,
                    "schema": AnalisisSoporteDTO.model_json_schema()
                }
            }
        )
        print(f"Tokens consumidos: {response.usage.total_tokens} (Entrada: {response.usage.prompt_tokens}, Salida: {response.usage.completion_tokens})")

        # Extraemos el contenido JSON generado
        json_raw = response.choices[0].message.content
        print("\n--- JSON RAW RECIBIDO DEL MODELO ---")
        print(json_raw)

        # Parseamos y validamos el JSON directamente contra la clase Pydantic DTO
        dto_objeto = AnalisisSoporteDTO.model_validate_json(json_raw)
        return dto_objeto
        
    except ValidationError as ve:
        print(f"\nError de validación Pydantic: El JSON no cumple con la estructura DTO: {ve}")
        return None
    except Exception as e:
        print(f"\nError de conexión con LM Studio: {e}")
        return None
    
if __name__ == "__main__":
    # Incidencia técnica de ejemplo no estructurada
    incidencia_ejemplo = (
        "2026-10-02 10:14:02 [CRITICAL] AuthController: SQLiteException: "
        "database is locked al intentar escribir token de sesión para el usuario ID 4092."
    )
    dto_resultado = procesar_log_a_dto(incidencia_ejemplo)
    if dto_resultado:
        print("\n=== DTO PARSEADO Y VALIDADO CON ÉXITO ===")
        print(f"Ticket ID: {dto_resultado.codigo_ticket}")
        print(f"Componente: {dto_resultado.componente_afectado}")
        print(f"Prioridad: {dto_resultado.nivel_prioridad}")
        print(f"Resumen: {dto_resultado.resumen_tecnico}")
        print(f"Solución: {dto_resultado.solucion_propuesta}")
        