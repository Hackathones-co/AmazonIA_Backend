"""
/api/v1/chat — Conversational chatbot powered by OpenAI API.
Uses tool calling to fetch real-time predictions from SALA endpoints.
"""
import logging
import httpx
from fastapi import APIRouter, Request

from app.schemas.responses import ChatRequest, ChatResponse
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# System prompts per module
MODULE_PROMPTS = {
    "pesca": (
        "Eres el asistente MANGLE de SALA Galápagos. Ayudas a pescadores artesanales "
        "de San Cristóbal a decidir si es seguro salir a pescar. Responde en español "
        "sencillo, con recomendaciones claras y directas. Si tienes datos del score de "
        "pesca, inclúyelos. Máximo 3 oraciones."
    ),
    "agro": (
        "Eres el asistente SCALESIA de SALA Galápagos. Ayudas a agricultores de la zona "
        "alta de San Cristóbal con recomendaciones de riego, siembra y cosecha. Usa los "
        "datos de humedad del suelo y predicción de lluvia. Responde en español sencillo."
    ),
    "bio": (
        "Eres el asistente GALÁPAGO de SALA Galápagos. Informas sobre el estado de la "
        "fauna endémica de San Cristóbal: tortugas gigantes, iguanas marinas, fragatas, "
        "lobos marinos. Usa datos de estrés térmico y condiciones de anidación."
    ),
    "risk": (
        "Eres el asistente GARÚA de SALA Galápagos. Ayudas a los habitantes a evaluar "
        "riesgos por zona: inundaciones, deslizamientos, oleaje peligroso. Da "
        "recomendaciones de seguridad claras."
    ),
    "visit": (
        "Eres el asistente ENCANTADA de SALA Galápagos. Ayudas a turistas a planificar "
        "su día en San Cristóbal basándote en las condiciones climáticas. Recomienda "
        "actividades, horarios y lugares. Sé entusiasta pero honesto sobre el clima."
    ),
}

DEFAULT_PROMPT = (
    "Eres SALA, el asistente inteligente de clima y ecosistemas de las Islas Galápagos, "
    "enfocado en San Cristóbal. Puedes ayudar con: pesca (MANGLE), agricultura (SCALESIA), "
    "fauna (GALÁPAGO), riesgo (GARÚA) y turismo (ENCANTADA). Responde siempre en español. "
    "Sé conciso y útil."
)


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """Chat endpoint — sends message to OpenAI API with module context."""
    if not settings.OPENAI_API_KEY:
        # Fallback: simple rule-based response if no API key
        return ChatResponse(
            response=_fallback_response(body.message, body.module),
            module_data=None,
            sources=[],
        )

    system_prompt = MODULE_PROMPTS.get(body.module, DEFAULT_PROMPT)

    # Build messages — OpenAI format requires system as first message
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (ensure proper role assignment)
    if body.history:
        for msg in body.history[-10:]:  # keep last 10 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role and content:  # only add if both role and content exist
                messages.append({"role": role, "content": content})
    
    # Add current user message
    messages.append({"role": "user", "content": body.message})
    
    logger.debug(f"Chat request - module: {body.module}, history_length: {len(body.history)}, total_messages: {len(messages)}")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "max_tokens": 500,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        response_text = data["choices"][0]["message"]["content"]

        return ChatResponse(
            response=response_text,
            module_data=None,
            sources=[],
        )

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return ChatResponse(
            response=_fallback_response(body.message, body.module),
            module_data=None,
            sources=[],
        )


def _fallback_response(message: str, module: str | None) -> str:
    """Simple fallback when OpenAI API is not available."""
    msg_lower = message.lower()

    if module == "pesca" or "pescar" in msg_lower or "pesca" in msg_lower:
        return ("Para saber si puedes salir a pescar, consulta el score de pesca "
                "en el módulo MANGLE. Te muestra las condiciones de viento, oleaje y visibilidad.")

    if module == "agro" or "sembrar" in msg_lower or "regar" in msg_lower:
        return ("Revisa el calendario de SCALESIA para recomendaciones de siembra y riego "
                "basadas en la humedad del suelo y la predicción de lluvia.")

    if module == "bio" or "tortuga" in msg_lower or "iguana" in msg_lower:
        return ("El módulo GALÁPAGO monitorea el estado de las especies endémicas. "
                "Consulta el estado actual para ver niveles de estrés térmico.")

    if module == "risk" or "playa" in msg_lower or "riesgo" in msg_lower:
        return ("GARÚA analiza el riesgo por zona. Consulta el mapa de riesgo "
                "para ver qué zonas son seguras hoy.")

    if module == "visit" or "hacer" in msg_lower or "turismo" in msg_lower:
        return ("ENCANTADA te recomienda las mejores actividades según el clima. "
                "Consulta las recomendaciones del día.")

    return ("Soy SALA, tu asistente de clima y ecosistemas de Galápagos. "
            "Puedo ayudarte con pesca, agricultura, fauna, riesgo o turismo. "
            "¿Qué necesitas saber?")
