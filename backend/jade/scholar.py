# backend/jade/scholar.py

import os
import sys
import json
import time
import re
import random
import uuid
from io import BytesIO
from typing import List, Dict, Any, Optional
import numpy as np

# --- 1. Setup e Dependências ---
# Removido setup_environment() pois será tratado no requirements.txt e Dockerfile

try:
    import groq
    import pypdf
    import faiss
    import graphviz
    import genanki
    from gtts import gTTS
    from pydub import AudioSegment
    import requests
    from bs4 import BeautifulSoup
    from youtube_transcript_api import YouTubeTranscriptApi
    from sentence_transformers import SentenceTransformer
    from fpdf import FPDF
    from duckduckgo_search import DDGS
except ImportError:
    # Em produção, isso deve falhar se as dependências não estiverem instaladas
    pass

# --- 2. Configuração Global ---
# Usaremos a configuração passada ou variável de ambiente
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 3. Camada de Ferramentas (Tooling Layer) ---

class ToolBox:
    """Caixa de ferramentas para os agentes."""

    @staticmethod
    def read_pdf(filepath: str) -> str:
        try:
            print(f"📄 [Ferramenta] Lendo PDF: {filepath}...")
            reader = pypdf.PdfReader(filepath)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            return f"Erro ao ler PDF: {str(e)}"

    @staticmethod
    def scrape_web(url: str) -> str:
        try:
            print(f"🌐 [Ferramenta] Acessando URL: {url}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            text = soup.get_text()
            return re.sub(r'\s+', ' ', text).strip()[:40000]
        except Exception as e:
            print(f"Erro ao acessar {url}: {e}")
            return ""

    @staticmethod
    def search_topic(topic: str) -> List[str]:
        """Pesquisa no DuckDuckGo e retorna URLs."""
        print(f"🔎 [Ferramenta] Pesquisando na Web sobre: '{topic}'...")
        urls = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=3))
                for r in results:
                    urls.append(r['href'])
        except Exception as e:
            print(f"Erro na busca: {e}")
        return urls

    @staticmethod
    def get_youtube_transcript(url: str) -> str:
        try:
            print(f"📺 [Ferramenta] Extraindo legendas do YouTube: {url}...")
            video_id = url.split("v=")[-1].split("&")[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en'])
            text = " ".join([t['text'] for t in transcript])
            return text
        except Exception as e:
            return f"Erro ao pegar legendas do YouTube: {str(e)}"

    @staticmethod
    def generate_audio_mix(script: List[Dict], filename="aula_podcast.mp3"):
        print("🎙️ [Estúdio] Produzindo áudio imersivo...")
        combined = AudioSegment.silent(duration=500)

        for line in script:
            speaker = line.get("speaker", "Narrador").upper()
            text = line.get("text", "")

            if "BERTA" in speaker or "PROFESSORA" in speaker or "AGENT B" in speaker:
                tts = gTTS(text=text, lang='pt', tld='pt', slow=False)
            else:
                # Gabriel / Agent A
                tts = gTTS(text=text, lang='pt', tld='com.br', slow=False)

            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            try:
                segment = AudioSegment.from_file(fp, format="mp3")
                combined += segment
                combined += AudioSegment.silent(duration=300)
            except: pass

        output_path = os.path.join("backend/generated", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.export(output_path, format="mp3")
        return output_path

    @staticmethod
    def generate_mindmap_image(dot_code: str, filename="mapa_mental"):
        try:
            print("🗺️ [Design] Renderizando Mapa Mental...")
            clean_dot = dot_code.replace("```dot", "").replace("```", "").strip()

            # Ensure generated directory exists
            output_dir = "backend/generated"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)

            src = graphviz.Source(clean_dot)
            src.format = 'png'
            filepath = src.render(output_path, view=False)
            return filepath
        except Exception as e:
            print(f"Erro ao gerar gráfico: {e}")
            return None

    @staticmethod
    def generate_anki_deck(qa_pairs: List[Dict], deck_name="ScholarGraph Deck"):
        print("🧠 [Anki] Criando arquivo de Flashcards (.apkg)...")
        try:
            model_id = random.randrange(1 << 30, 1 << 31)
            deck_id = random.randrange(1 << 30, 1 << 31)

            my_model = genanki.Model(
              model_id,
              'Simple Model',
              fields=[{'name': 'Question'}, {'name': 'Answer'}],
              templates=[{
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
              }]
            )

            my_deck = genanki.Deck(deck_id, deck_name)

            for item in qa_pairs:
                my_deck.add_note(genanki.Note(
                  model=my_model,
                  fields=[item['question'], item['answer']]
                ))

            output_dir = "backend/generated"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"flashcards_{uuid.uuid4().hex[:8]}.apkg")
            genanki.Package(my_deck).write_to_file(filename)
            return filename
        except Exception as e:
            print(f"Erro ao criar Anki deck: {e}")
            return None

# --- 4. Vector Store (RAG) ---

class VectorMemory:
    def __init__(self):
        print("🧠 [Memória] Inicializando Banco de Vetores (RAG)...")
        # Modelo leve para embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []

    def ingest(self, text: str, chunk_size=500):
        words = text.split()
        # Cria chunks sobrepostos para melhor contexto
        self.chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), int(chunk_size*0.8))]

        print(f"🧠 [Memória] Vetorizando {len(self.chunks)} fragmentos...")
        if not self.chunks: return

        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("🧠 [Memória] Indexação concluída.")

    def retrieve(self, query: str, k=3) -> str:
        if not self.index: return ""
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype('float32'), k)

        results = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
        return "\n\n".join(results)

# --- 5. Estado e LLM ---

class GraphState:
    def __init__(self):
        self.raw_content: str = ""
        self.summary: str = ""
        self.script: List[Dict] = []
        self.quiz_data: List[Dict] = []
        self.mindmap_path: str = ""
        self.flashcards: List[Dict] = []
        self.current_quiz_question: int = 0
        self.xp: int = 0
        self.mode: str = "input" # input, menu, quiz

class LLMEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = groq.Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def chat(self, messages: List[Dict], json_mode=False) -> str:
        try:
            kwargs = {"messages": messages, "model": self.model, "temperature": 0.6}
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            return self.client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            return f"Erro na IA: {e}"

# --- 6. Agentes Avançados (GOD MODE) ---

class ResearcherAgent:
    """Agente que pesquisa na web se o input for um tópico."""
    def deep_research(self, topic: str) -> str:
        print(f"🕵️ [Pesquisador] Iniciando Deep Research sobre: {topic}")
        urls = ToolBox.search_topic(topic)
        if not urls:
            return f"Não encontrei informações sobre {topic}."

        full_text = ""
        for url in urls:
            content = ToolBox.scrape_web(url)
            if content:
                full_text += f"\n\n--- Fonte: {url} ---\n{content[:10000]}"

        return full_text

class FlashcardAgent:
    """Agente focado em memorização (Anki)."""
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def create_deck(self, content: str) -> List[Dict]:
        print("🃏 [Flashcard] Gerando pares Pergunta-Resposta...")
        prompt = f"""
        Crie 10 Flashcards (Pergunta e Resposta) sobre o conteúdo para memorização.
        SAÍDA JSON: {{ "cards": [ {{ "question": "...", "answer": "..." }} ] }}
        Conteúdo: {content[:15000]}
        """
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            return json.loads(resp).get("cards", [])
        except: return []

class IngestAgent:
    def __init__(self, researcher: ResearcherAgent):
        self.researcher = researcher

    def process(self, user_input: str) -> str:
        # Se for arquivo
        if user_input.lower().endswith(".pdf") and os.path.exists(user_input):
            return ToolBox.read_pdf(user_input)
        # Se for URL
        elif "youtube.com" in user_input or "youtu.be" in user_input:
            return ToolBox.get_youtube_transcript(user_input)
        elif user_input.startswith("http"):
            return ToolBox.scrape_web(user_input)
        # Se não for URL nem arquivo, assume que é Tópico para Pesquisa
        else:
            print("🔍 Entrada detectada como Tópico. Ativando ResearcherAgent...")
            return self.researcher.deep_research(user_input)

class ProfessorAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def summarize(self, full_text: str) -> str:
        print("🧠 [Professor] Gerando resumo estratégico...")
        prompt = f"""
        Você é um Professor Universitário. Crie um resumo estruturado e profundo.
        Texto: {full_text[:25000]}
        Formato: # Título / ## Introdução / ## Pontos Chave / ## Conclusão
        """
        return self.llm.chat([{"role": "user", "content": prompt}])

class VisualizerAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def create_mindmap(self, text: str) -> str:
        print("🎨 [Visualizador] Projetando Mapa Mental...")
        prompt = f"""
        Crie um código GRAPHVIZ (DOT) para um mapa mental deste conteúdo.
        Use formas coloridas. NÃO explique, apenas dê o código DOT dentro de ```dot ... ```.
        Texto: {text[:15000]}
        """
        response = self.llm.chat([{"role": "user", "content": prompt}])
        match = re.search(r'```dot(.*?)```', response, re.DOTALL)
        if match: return match.group(1).strip()
        return response

class ScriptwriterAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def create_script(self, content: str, mode="lecture") -> List[Dict]:
        if mode == "debate":
            print("🔥 [Roteirista] Criando DEBATE INTENSO...")
            prompt = f"""
            Crie um DEBATE acalorado mas intelectual entre dois agentes (8 falas).
            Personagens:
            - AGENT A (Gabriel): A favor / Otimista / Pragmático.
            - AGENT B (Berta): Contra / Cética / Filosófica.

            SAÍDA JSON: {{ "dialogue": [ {{"speaker": "Agent A", "text": "..."}}, {{"speaker": "Agent B", "text": "..."}} ] }}
            Tema Base: {content[:15000]}
            """
        else:
            print("✍️ [Roteirista] Escrevendo roteiro de aula...")
            prompt = f"""
            Crie um roteiro de podcast (8 falas).
            Personagens: GABRIEL (Aluno BR) e BERTA (Professora PT).
            SAÍDA JSON: {{ "dialogue": [ {{"speaker": "Gabriel", "text": "..."}}, ...] }}
            Base: {content[:15000]}
            """

        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            return json.loads(resp).get("dialogue", [])
        except: return []

class ExaminerAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def generate_quiz(self, content: str) -> List[Dict]:
        print("📝 [Examinador] Criando Prova Gamificada...")
        prompt = f"""
        Crie 5 perguntas de múltipla escolha (Difíceis).
        SAÍDA JSON: {{ "quiz": [ {{ "question": "...", "options": ["A)..."], "correct_option": "A", "explanation": "..." }} ] }}
        Base: {content[:15000]}
        """
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            return json.loads(resp).get("quiz", [])
        except: return []

class PublisherAgent:
    def create_handout(self, state: GraphState, filename="Apostila_Estudos.pdf"):
        print("📚 [Editora] Diagramando Apostila PDF...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Apostila de Estudos - Scholar Graph", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        safe_summary = state.summary.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_summary)
        if state.mindmap_path and os.path.exists(state.mindmap_path):
            pdf.add_page()
            pdf.image(state.mindmap_path, x=10, y=30, w=190)

        output_dir = "backend/generated"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        pdf.output(filepath)
        return filepath

# --- 7. Agent Class wrapper for backend integration ---

class ScholarAgent:
    def __init__(self):
        self.user_states = {} # Map user_id to (ScholarGraphGodMode instance or GraphState)
        self.api_key = os.getenv("GROQ_API_KEY")
        # Initialize one engine for general use if needed, but we probably need instances per user or shared resources.
        # We'll create instances per user request if they don't exist?
        # Actually, let's keep it simple. We store state per user.

    def get_or_create_state(self, user_id):
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                "state": GraphState(),
                "memory": VectorMemory(),
                "llm": LLMEngine(self.api_key),
                "researcher": ResearcherAgent(),
                "ingestor": None, # Will be init with researcher
                "professor": None,
                "visualizer": None,
                "scriptwriter": None,
                "examiner": None,
                "flashcarder": None,
                "publisher": None
            }
            # Wiring dependencies
            u = self.user_states[user_id]
            u["ingestor"] = IngestAgent(u["researcher"])
            u["professor"] = ProfessorAgent(u["llm"])
            u["visualizer"] = VisualizerAgent(u["llm"])
            u["scriptwriter"] = ScriptwriterAgent(u["llm"])
            u["examiner"] = ExaminerAgent(u["llm"])
            u["flashcarder"] = FlashcardAgent(u["llm"])
            u["publisher"] = PublisherAgent()

        return self.user_states[user_id]

    def respond(self, history, user_input, user_id="default", vision_context=None):
        """
        Adapts the CLI interaction loop to a Request/Response model.
        """
        u = self.get_or_create_state(user_id)
        state = u["state"]

        # Helper to format menu
        def get_menu():
            return (
                "\n\n🎓 *MENU SCHOLAR GRAPH*\n"
                "1. 🧠 Resumo Estratégico\n"
                "2. 🗺️ Mapa Mental Visual\n"
                "3. 🎧 Podcast (Aula Didática)\n"
                "4. 🔥 DEBATE IA (Visões Opostas)\n"
                "5. 🎮 Quiz Gamificado\n"
                "6. 🃏 Gerar Flashcards (Anki .apkg)\n"
                "7. 📚 Baixar Apostila Completa\n"
                "8. 🔄 Novo Tópico\n"
                "👉 Escolha uma opção (número ou texto):"
            )

        # Helper for response with optional file
        response_text = ""
        audio_path = None

        # State Machine Logic

        # 1. Input Mode: Waiting for topic/url/pdf
        if state.mode == "input":
            if not user_input.strip():
                return "Por favor, forneça um tópico, URL ou arquivo PDF para começar.", None, history

            response_text = f"🔄 Processando '{user_input}'... (Isso pode levar alguns segundos)"

            # Process content
            content = u["ingestor"].process(user_input)
            if not content or len(content) < 50:
                response_text = "❌ Falha ao obter conteúdo suficiente ou tópico não encontrado. Tente novamente."
                return response_text, None, history

            state.raw_content = content
            u["memory"].ingest(content)
            state.mode = "menu"
            response_text += "\n✅ Conteúdo processado com sucesso!" + get_menu()

            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response_text})
            return response_text, None, history

        # 2. Quiz Mode
        elif state.mode == "quiz":
            # Check answer
            current_q = state.quiz_data[state.current_quiz_question]
            ans = user_input.strip().upper()

            feedback = ""
            if ans and ans[0] == current_q['correct_option'][0]:
                state.xp += 100
                feedback = f"✨ ACERTOU! +100 XP. (Total: {state.xp})\n"
            else:
                feedback = f"💀 Errou... A resposta era {current_q['correct_option']}.\nExplanation: {current_q.get('explanation', '')}\n"

            state.current_quiz_question += 1

            if state.current_quiz_question < len(state.quiz_data):
                # Next Question
                q = state.quiz_data[state.current_quiz_question]
                response_text = feedback + f"\n🔹 QUESTÃO {state.current_quiz_question+1}:\n{q['question']}\n" + "\n".join(q['options'])
            else:
                # End of Quiz
                response_text = feedback + f"\n🏆 FIM DO QUIZ! TOTAL DE XP: {state.xp}\n" + get_menu()
                state.mode = "menu"

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response_text})
            return response_text, None, history

        # 3. Menu Mode
        elif state.mode == "menu":
            opt = user_input.strip()

            if opt.startswith("1") or "resumo" in opt.lower():
                state.summary = u["professor"].summarize(state.raw_content)
                response_text = "📝 *RESUMO ESTRATÉGICO:*\n\n" + state.summary + get_menu()

            elif opt.startswith("2") or "mapa" in opt.lower():
                dot = u["visualizer"].create_mindmap(state.raw_content)
                filename = f"mindmap_{uuid.uuid4().hex[:8]}"
                path = ToolBox.generate_mindmap_image(dot, filename)
                if path:
                    state.mindmap_path = path
                    # Since we return text and audio only in this signature, we might need a way to send image.
                    # The current app structure supports sending audio_base64.
                    # We might need to hack it to send image link or modify app.py.
                    # For now, let's return a link relative to backend/generated (assuming static serving)
                    response_text = f"🗺️ Mapa Mental gerado: [Baixar Imagem](/generated/{os.path.basename(path)})\n" + get_menu()
                else:
                    response_text = "❌ Erro ao gerar mapa mental." + get_menu()

            elif opt.startswith("3") or "podcast" in opt.lower():
                script = u["scriptwriter"].create_script(state.raw_content, mode="lecture")
                filename = f"podcast_{uuid.uuid4().hex[:8]}.mp3"
                path = ToolBox.generate_audio_mix(script, filename)
                audio_path = path # Return this to be played
                response_text = "🎧 Aqui está o seu Podcast sobre o tema." + get_menu()

            elif opt.startswith("4") or "debate" in opt.lower():
                script = u["scriptwriter"].create_script(state.raw_content, mode="debate")
                filename = f"debate_{uuid.uuid4().hex[:8]}.mp3"
                path = ToolBox.generate_audio_mix(script, filename)
                audio_path = path
                response_text = "🔥 Debate gerado com sucesso." + get_menu()

            elif opt.startswith("5") or "quiz" in opt.lower():
                state.quiz_data = u["examiner"].generate_quiz(state.raw_content)
                if state.quiz_data:
                    state.mode = "quiz"
                    state.current_quiz_question = 0
                    state.xp = 0
                    q = state.quiz_data[0]
                    response_text = f"🎮 *MODO QUIZ INICIADO*\n\n🔹 QUESTÃO 1:\n{q['question']}\n" + "\n".join(q['options'])
                else:
                    response_text = "❌ Não foi possível gerar o quiz." + get_menu()

            elif opt.startswith("6") or "flashcard" in opt.lower():
                cards = u["flashcarder"].create_deck(state.raw_content)
                if cards:
                    path = ToolBox.generate_anki_deck(cards)
                    if path:
                        response_text = f"✅ Flashcards gerados: [Baixar Deck Anki](/generated/{os.path.basename(path)})" + get_menu()
                    else:
                        response_text = "❌ Erro ao salvar arquivo." + get_menu()
                else:
                    response_text = "❌ Erro ao gerar flashcards." + get_menu()

            elif opt.startswith("7") or "apostila" in opt.lower():
                if state.summary:
                    filename = f"apostila_{uuid.uuid4().hex[:8]}.pdf"
                    path = u["publisher"].create_handout(state, filename)
                    response_text = f"📚 Apostila pronta: [Baixar PDF](/generated/{os.path.basename(path)})" + get_menu()
                else:
                    response_text = "⚠️ Gere o Resumo (Opção 1) primeiro!" + get_menu()

            elif opt.startswith("8") or "novo" in opt.lower() or "sair" in opt.lower():
                state.mode = "input"
                # Reset state?
                state.raw_content = ""
                state.summary = ""
                response_text = "🔄 Reiniciando... Qual o novo tópico, link ou PDF?"

            else:
                response_text = "Opção inválida. Tente novamente.\n" + get_menu()

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response_text})
            return response_text, audio_path, history

        return "Erro de estado.", None, history
