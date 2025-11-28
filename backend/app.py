# app.py - VERSÃO COMPLETA COM VOZ (BASE64) E VISÃO
import os
import base64
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from jade.core import JadeAgent
from jade.scholar import ScholarAgent

print("Iniciando a J.A.D.E. com FastAPI...")
jade_agent = JadeAgent()
scholar_agent = ScholarAgent()
print("J.A.D.E. pronta para receber requisições.")

app = FastAPI(title="J.A.D.E. API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Mount generated directory for static files (PDFs, Images)
os.makedirs("backend/generated", exist_ok=True)
app.mount("/generated", StaticFiles(directory="backend/generated"), name="generated")

# Dicionário global para armazenar sessões de usuários
# Structure: user_sessions[user_id] = { "jade": [...], "scholar": [...] }
user_sessions = {}

class UserRequest(BaseModel):
    user_input: str
    image_base64: str | None = None
    user_id: str | None = None
    agent_type: str = "jade" # "jade" or "scholar"

@app.post("/chat")
def handle_chat(request: UserRequest):
    try:
        user_id = request.user_id if request.user_id else "default_user"
        agent_type = request.agent_type.lower()

        if user_id not in user_sessions:
            print(f"Nova sessão criada para: {user_id}")
            user_sessions[user_id] = {
                "jade": [jade_agent.system_prompt],
                "scholar": []
            }

        # Ensure sub-keys exist if session existed but new agent type is used
        if "jade" not in user_sessions[user_id]: user_sessions[user_id]["jade"] = [jade_agent.system_prompt]
        if "scholar" not in user_sessions[user_id]: user_sessions[user_id]["scholar"] = []

        vision_context = None
        if request.image_base64:
            try:
                header, encoded_data = request.image_base64.split(",", 1)
                image_bytes = base64.b64decode(encoded_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                # Only Jade handles vision for now, but we can pass context if needed
                vision_context = jade_agent.image_handler.process_pil_image(pil_image)
            except Exception as img_e:
                print(f"Erro ao processar imagem Base64: {img_e}")
                vision_context = "Houve um erro ao analisar a imagem."

        final_user_input = request.user_input if request.user_input else "Descreva a imagem em detalhes."

        bot_response_text = ""
        audio_path = None

        if agent_type == "scholar":
            current_history = user_sessions[user_id]["scholar"]
            bot_response_text, audio_path, updated_history = scholar_agent.respond(
                history=current_history,
                user_input=final_user_input,
                user_id=user_id,
                vision_context=vision_context
            )
            user_sessions[user_id]["scholar"] = updated_history
        else:
            # Default to J.A.D.E.
            current_history = user_sessions[user_id]["jade"]
            bot_response_text, audio_path, updated_history = jade_agent.respond(
                history=current_history,
                user_input=final_user_input,
                user_id=user_id,
                vision_context=vision_context
            )
            user_sessions[user_id]["jade"] = updated_history

        # LÓGICA DO ÁUDIO: Converte o arquivo MP3 gerado para Base64
        # Scholar agent might return a path to a static file instead of a temp file to be deleted.
        # We need to distinguish.
        # JadeAgent returns a temp file that is deleted.
        # ScholarAgent returns a file in /generated/ that should PROBABLY remain accessible via URL,
        # OR we can send it as base64 too.
        # If the path starts with "backend/generated", we assume it is static and we might want to return the URL?
        # BUT the frontend expects audio_base64 to play it immediately.
        # So we can still base64 encode it for immediate playback.

        audio_base64 = None
        if audio_path and os.path.exists(audio_path):
            print(f"Codificando arquivo de áudio: {audio_path}")
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            # Only remove if it's NOT in generated (i.e. it's a temp file from Jade)
            if "backend/generated" not in audio_path:
                os.remove(audio_path)

        return {
            "success": True,
            "bot_response": bot_response_text,
            "audio_base64": audio_base64
        }
    except Exception as e:
        print(f"Erro crítico no endpoint /chat: {e}")
        return {"success": False, "error": str(e)}

@app.get("/")
def root():
    return {"message": "Servidor J.A.D.E. com FastAPI está online."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Iniciando o servidor Uvicorn em http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
