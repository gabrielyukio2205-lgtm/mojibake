# ==========================================
# RAG PROJECT - COLAB VERSION
# ==========================================
#
# COMO RODAR NO GOOGLE COLAB:
# 1. Crie uma conta na Groq e pegue sua API KEY: https://console.groq.com/keys
# 2. Copie todo este código e cole em uma célula de código no Colab.
# 3. Antes de rodar, instale as dependências executando o comando abaixo numa célula separada:
#    !pip install -q gradio groq langchain-groq langchain-community langchain-huggingface faiss-cpu pypdf duckduckgo-search sentence-transformers
# 4. Execute este script.
#
# ==========================================

import os
import gradio as gr
from groq import Groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate

# --- Configurações Globais e Estado ---
class RAGState:
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.api_key = None
        self.llm = None

    def set_api_key(self, api_key):
        self.api_key = api_key
        # Inicializa o LLM com a chave fornecida
        self.llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")

app_state = RAGState()

# --- Funções de Processamento de Arquivos (RAG Local) ---
def process_files(files):
    """
    Lê arquivos PDF/TXT enviados, quebra em pedaços e cria o índice vetorial (FAISS).
    """
    if not files:
        return "Nenhum arquivo enviado."

    documents = []

    for file_path in files:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Erro ao ler arquivo {file_path}: {e}")

    if not documents:
        return "Não foi possível ler os arquivos."

    # Quebra o texto em chunks menores para vetorização
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Cria o Vector Store (Banco de Dados Vetorial) em memória
    app_state.vector_store = FAISS.from_documents(splits, app_state.embeddings)

    return f"Sucesso! {len(splits)} trechos indexados de {len(files)} arquivos. Agora você pode selecionar 'Meus Arquivos' e perguntar."

# --- Funções de Busca ---
def search_web(query):
    """Realiza busca na web usando DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"Erro na busca web: {e}"

def search_files(query):
    """Realiza busca semântica nos arquivos indexados."""
    if not app_state.vector_store:
        return None

    # Retorna os 3 trechos mais relevantes
    docs = app_state.vector_store.similarity_search(query, k=3)
    # Concatena o conteúdo dos documentos encontrados
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# --- Lógica do Chat ---
def chat_response(message, history, search_source, api_key_input):
    """
    Função principal que processa a mensagem do usuário.
    """
    # 1. Configuração da API Key
    if not app_state.api_key and not api_key_input:
        return "⚠️ Por favor, insira sua Groq API Key no topo primeiro."

    if api_key_input and api_key_input != app_state.api_key:
        try:
            app_state.set_api_key(api_key_input)
        except Exception as e:
            return f"Erro ao configurar API Key: {e}"

    if not app_state.llm:
        return "Erro: LLM não inicializado. Verifique a chave."

    # 2. Obtenção do Contexto (RAG)
    context = ""
    source_info = ""

    if search_source == "Web Search":
        status_msg = "🔍 Pesquisando na Web..."
        # yield status_msg # (Opcional: para streamar status)
        context = search_web(message)
        source_info = "Informações obtidas via Web Search."

    elif search_source == "Meus Arquivos":
        if not app_state.vector_store:
            return "⚠️ Nenhum arquivo foi indexado ainda. Faça upload na aba lateral."
        status_msg = "📂 Pesquisando nos arquivos..."
        context = search_files(message)
        if not context:
            context = "Não encontrei informações relevantes nos arquivos sobre isso."
        source_info = "Informações obtidas dos seus Arquivos Locais."

    # 3. Construção do Prompt
    system_prompt = f"""Você é um assistente inteligente e prestativo que fala Português.
    Use o contexto fornecido abaixo para responder à pergunta do usuário.

    Fonte do Contexto: {source_info}

    CONTEXTO:
    {context}

    Se a resposta não estiver no contexto, diga que não sabe com base nas informações fornecidas,
    mas tente ser útil com seu conhecimento geral se apropriado (avisando explicitamente).
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    chain = prompt | app_state.llm

    # 4. Geração da Resposta
    try:
        response = chain.invoke({"input": message})
        return response.content
    except Exception as e:
        return f"Erro ao chamar a Groq API: {e}"


# --- Interface Gradio ---
with gr.Blocks(title="Projeto RAG Foda - Groq + Colab", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 RAG Assistant (Groq + Web + Docs)")
    gr.Markdown("Este projeto permite conversar usando um LLM (Llama3 via Groq) com acesso à Internet ou aos seus Documentos PDF/TXT.")

    with gr.Row():
        with gr.Column(scale=1):
            # Configurações e Upload
            api_key = gr.Textbox(label="Groq API Key", placeholder="gsk_...", type="password")

            gr.Markdown("### 📂 Base de Conhecimento (RAG Local)")
            file_input = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="Upload de Arquivos")
            upload_btn = gr.Button("Processar Arquivos")
            upload_status = gr.Label(value="Aguardando arquivos...", label="Status da Indexação")

        with gr.Column(scale=3):
            # Chat e Controles
            search_mode = gr.Radio(
                choices=["Web Search", "Meus Arquivos"],
                value="Web Search",
                label="🌐 Onde devo pesquisar?",
                info="Escolha a fonte de informação para o RAG."
            )

            chatbot = gr.ChatInterface(
                fn=chat_response,
                additional_inputs=[search_mode, api_key],
                description="Pergunte qualquer coisa...",
            )

    # Eventos
    upload_btn.click(process_files, inputs=[file_input], outputs=[upload_status])

# Lançar a interface
if __name__ == "__main__":
    print("Iniciando interface...")
    demo.launch(share=True, debug=True)
