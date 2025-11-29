(function() {
    'use strict';

    // URL da sua API no Render (Mantenha a que estava funcionando)
    const API_URL = 'https://jade-proxy.onrender.com/chat';

    // State Management
    let conversations = [];
    let currentChatId = null;

    // Cache DOM elements
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const agentSelector = document.getElementById('agent-selector');
    const imageInput = document.getElementById('imageInput');
    const imageBtn = document.getElementById('imageBtn');
    const audioPlayer = document.getElementById('audioPlayer');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image-btn');

    // Sidebar Elements
    const sidebar = document.getElementById('sidebar');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const chatHistoryList = document.getElementById('chat-history-list');

    // Theme Elements
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const sunIcon = document.getElementById('sun-icon');
    const moonIcon = document.getElementById('moon-icon');

    // Init Marked & Highlight.js
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            highlight: function(code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            langPrefix: 'hljs language-'
        });
    }

    function setupEventListeners() {
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        imageBtn.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', handleImageSelection);
        removeImageBtn.addEventListener('click', clearImagePreview);

        // Sidebar Events
        toggleSidebarBtn.addEventListener('click', toggleSidebar);
        mobileMenuBtn.addEventListener('click', toggleSidebar);
        newChatBtn.addEventListener('click', startNewChat);

        // Theme Event
        themeToggleBtn.addEventListener('click', toggleTheme);
    }

    // --- Identity Management (A Correção da Berta 🧠) ---

    function getPersistentUserId() {
        // Tenta pegar o ID mestre do navegador
        let userId = localStorage.getItem('jade_master_user_id');
        if (!userId) {
            // Se não existir, cria um novo e salva para sempre
            // Ex: user_k8s7d6f5...
            userId = 'user_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
            localStorage.setItem('jade_master_user_id', userId);
        }
        return userId;
    }

    // --- Theme Management ---

    function initTheme() {
        const savedTheme = localStorage.getItem('jade_theme');
        if (savedTheme === 'light') {
            document.body.setAttribute('data-theme', 'light');
            updateThemeIcons(true);
        } else {
            document.body.removeAttribute('data-theme');
            updateThemeIcons(false);
        }
    }

    function toggleTheme() {
        const isLight = document.body.getAttribute('data-theme') === 'light';
        if (isLight) {
            document.body.removeAttribute('data-theme');
            localStorage.setItem('jade_theme', 'dark');
            updateThemeIcons(false);
        } else {
            document.body.setAttribute('data-theme', 'light');
            localStorage.setItem('jade_theme', 'light');
            updateThemeIcons(true);
        }
    }

    function updateThemeIcons(isLight) {
        if (isLight) {
            sunIcon.classList.add('hidden');
            moonIcon.classList.remove('hidden');
        } else {
            sunIcon.classList.remove('hidden');
            moonIcon.classList.add('hidden');
        }
    }

    // --- State & Storage ---

    function init() {
        initTheme();
        loadConversations();
        if (conversations.length > 0) {
            loadChat(conversations[0].id);
        } else {
            startNewChat();
        }
        renderHistoryList();
    }

    function toggleSidebar() {
        const isMobile = window.innerWidth <= 768;
        if (isMobile) {
            sidebar.classList.toggle('open');
        } else {
            sidebar.classList.toggle('collapsed');
            document.body.classList.toggle('sidebar-closed');
        }
    }

    function loadConversations() {
        const stored = localStorage.getItem('jade_conversations');
        if (stored) {
            try {
                conversations = JSON.parse(stored);
            } catch (e) {
                console.error('Failed to parse conversations', e);
                conversations = [];
            }
        }
    }

    function saveConversations() {
        localStorage.setItem('jade_conversations', JSON.stringify(conversations));
        renderHistoryList();
    }

    function createId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    function startNewChat() {
        currentChatId = createId();
        const newChat = {
            id: currentChatId,
            title: 'Nova conversa',
            messages: [],
            timestamp: Date.now()
        };
        conversations.unshift(newChat); // Add to top
        saveConversations();

        chatbox.innerHTML = '';
        appendWelcomeMessage();
        sidebar.classList.remove('open');
    }

    function loadChat(id) {
        const chat = conversations.find(c => c.id === id);
        if (!chat) return;

        currentChatId = id;
        chatbox.innerHTML = '';

        if (chat.messages.length === 0) {
            appendWelcomeMessage();
        } else {
            chat.messages.forEach(msg => {
                appendMessage(msg.sender, msg.text, false, false);
            });
        }

        renderHistoryList();
        sidebar.classList.remove('open');
    }

    function saveMessageToCurrentChat(sender, text) {
        const chatIndex = conversations.findIndex(c => c.id === currentChatId);
        if (chatIndex !== -1) {
            const chat = conversations[chatIndex];
            chat.messages.push({ sender, text, timestamp: Date.now() });

            if (sender === 'Você' && chat.title === 'Nova conversa') {
                chat.title = text.length > 30 ? text.substring(0, 30) + '...' : text;
            }

            conversations.splice(chatIndex, 1);
            conversations.unshift(chat);

            saveConversations();
        }
    }

    function deleteChat(e, id) {
        e.stopPropagation();
        if (confirm('Tem certeza que deseja excluir esta conversa?')) {
            conversations = conversations.filter(c => c.id !== id);
            saveConversations();
            if (currentChatId === id) {
                if (conversations.length > 0) {
                    loadChat(conversations[0].id);
                } else {
                    startNewChat();
                }
            }
        }
    }

    // --- UI Rendering ---

    function renderHistoryList() {
        chatHistoryList.innerHTML = '';
        conversations.forEach(chat => {
            const div = document.createElement('div');
            div.className = `history-item ${chat.id === currentChatId ? 'active' : ''}`;

            // Text container
            const span = document.createElement('span');
            span.textContent = chat.title;
            span.style.flex = '1';
            span.style.overflow = 'hidden';
            span.style.textOverflow = 'ellipsis';

            // Delete Button
            const delBtn = document.createElement('button');
            delBtn.className = 'delete-chat-btn';
            delBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
            delBtn.title = "Excluir conversa";
            delBtn.onclick = (e) => deleteChat(e, chat.id);

            div.appendChild(span);
            div.appendChild(delBtn);
            div.onclick = () => loadChat(chat.id);

            chatHistoryList.appendChild(div);
        });
    }

    function appendWelcomeMessage() {
        const el = document.createElement('div');
        el.className = 'message bot welcome-message';
        el.innerHTML = `
            <div class="avatar">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg>
            </div>
            <div class="content">
            <div class="sender-name">J.A.D.E.</div>
            <div class="text">Olá. Eu sou J.A.D.E., sua assistente de inteligência artificial avançada. Como posso ajudar você hoje?</div>
            </div>
        `;
        chatbox.appendChild(el);
    }

    function handleImageSelection() {
        if (imageInput.files && imageInput.files[0]) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(imageInput.files[0]);
        }
    }

    function clearImagePreview() {
        imageInput.value = '';
        imagePreviewContainer.classList.add('hidden');
    }

    function renderMarkdown(text) {
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        }
        let html = escapeHtml(text);
        html = html.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
        html = html.replace(/\*(.*?)\*/g, '<b>$1</b>');
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    function escapeHtml(s) {
        return s.replace(/[&<>"']/g, ch => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[ch]));
    }

    function createMessageElement(sender, content, isTyping = false) {
        const isUser = sender === 'Você';
        const senderClass = isUser ? 'user' : 'bot';

        const el = document.createElement('div');
        el.className = `message ${senderClass}`;

        const avatarHTML = isUser
            ? `<div class="avatar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>`
            : `<div class="avatar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg></div>`;

        let textHTML;
        if (isTyping) {
            textHTML = `<div class="text"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
        } else {
            textHTML = `<div class="text">${renderMarkdown(content)}</div>`;
        }

        const contentHTML = `
            <div class="content">
                <div class="sender-name">${sender}</div>
                ${textHTML}
            </div>
        `;

        el.innerHTML = isUser ? (contentHTML + avatarHTML) : (avatarHTML + contentHTML);
        return el;
    }

    function appendMessage(sender, textContent, isTyping = false, save = true) {
        const el = createMessageElement(sender, textContent, isTyping);
        chatbox.appendChild(el);
        chatbox.scrollTop = chatbox.scrollHeight;

        if (save && !isTyping) {
            saveMessageToCurrentChat(sender, textContent);
        }

        return el;
    }

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    function updateBotMessage(messageEl, text) {
        const textEl = messageEl.querySelector('.text');
        if (textEl) {
            textEl.innerHTML = renderMarkdown(text);
        }
    }

    function speakText(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const plainText = text.replace(/[#*`\[\]]/g, '');
            const utterance = new SpeechSynthesisUtterance(plainText);
            utterance.lang = 'pt-BR';
            const voices = window.speechSynthesis.getVoices();
            const preferredVoice = voices.find(v => v.lang.includes('pt-BR') && v.name.includes('Google'));
            if (preferredVoice) utterance.voice = preferredVoice;
            window.speechSynthesis.speak(utterance);
        }
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        let image_base64 = null;
        if (!message && imageInput.files.length === 0) return;

        if (!currentChatId) startNewChat();

        const selectedAgent = agentSelector.value;
        const agentName = selectedAgent === 'scholar' ? 'Scholar Graph' : 'J.A.D.E.';

        if (imageInput.files.length > 0) {
            const msgText = `${message || ''} [Imagem Anexada]`;
            appendMessage('Você', msgText);
            image_base64 = await fileToBase64(imageInput.files[0]);
            clearImagePreview();
        } else {
            appendMessage('Você', message);
        }
        userInput.value = '';

        const jadeTypingMessage = appendMessage(agentName, '', true, false);

        // --- AQUI ESTÁ A CORREÇÃO DA MEMÓRIA ---
        // Pegamos o ID mestre que nunca muda
        const masterUserId = getPersistentUserId();

        try {
            const resp = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_input: message,
                    image_base64: image_base64,
                    // Enviamos o ID mestre, não o ID do chat atual
                    user_id: masterUserId,
                    agent_type: selectedAgent
                })
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

            const json = await resp.json();
            let botResponse;

            if (json.success) {
                botResponse = json.bot_response;
                if (json.audio_base64) {
                    audioPlayer.src = `data:audio/mpeg;base64,${json.audio_base64}`;
                    audioPlayer.play();
                } else {
                    speakText(botResponse);
                }
            } else {
                botResponse = `[Erro: ${json.error || 'Desconhecido'}]`;
            }

            if (botResponse === undefined) {
                botResponse = "[Erro de comunicação]";
            }

            updateBotMessage(jadeTypingMessage, botResponse);
            saveMessageToCurrentChat(agentName, botResponse);
            chatbox.scrollTop = chatbox.scrollHeight;

        } catch (err) {
            console.error(err);
            const errorText = `Falha ao conectar (${err.message})`;
            updateBotMessage(jadeTypingMessage, errorText);
            saveMessageToCurrentChat(agentName, errorText);
        }
    }

    setupEventListeners();
    init();

})();
