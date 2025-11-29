(function() {
    'use strict';

    // 1. URL DA API (O Proxy no Render)
    const PROXY_BASE_URL = 'https://jade-proxy.onrender.com';
    const API_URL = `${PROXY_BASE_URL}/chat`;

    // State Management
    let conversations = [];
    let currentChatId = null;
    let currentAgent = 'jade'; // Default agent

    // Cache DOM elements
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const imageInput = document.getElementById('imageInput');
    const imageBtn = document.getElementById('imageBtn');
    const audioPlayer = document.getElementById('audioPlayer');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image-btn');
    const voiceBtn = document.getElementById('voiceBtn'); // New Voice Button
    const audioVisualizer = document.getElementById('audio-visualizer'); // New Visualizer

    // Sidebar Elements
    const sidebar = document.getElementById('sidebar');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const chatHistoryList = document.getElementById('chat-history-list');
    const agentSwitcher = document.getElementById('agent-switcher'); // New

    // Header Elements
    const headerTitle = document.getElementById('header-title');

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

        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            if (this.value === '') this.style.height = 'auto';
        });

        // Paste Image Support
        userInput.addEventListener('paste', handlePaste);

        userInput.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
                // Reset height
                userInput.style.height = 'auto';
            }
        });
        imageBtn.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', handleImageSelection);
        removeImageBtn.addEventListener('click', clearImagePreview);

        // Voice Mode
        if (voiceBtn) {
            voiceBtn.addEventListener('click', toggleVoiceRecognition);
        }

        // Sidebar Events
        toggleSidebarBtn.addEventListener('click', toggleSidebar);
        mobileMenuBtn.addEventListener('click', toggleSidebar);
        newChatBtn.addEventListener('click', startNewChat);

        // Agent Switcher Events
        if (agentSwitcher) {
            const buttons = agentSwitcher.querySelectorAll('.agent-btn');
            buttons.forEach(btn => {
                btn.addEventListener('click', () => {
                    const agent = btn.getAttribute('data-agent');
                    switchAgent(agent);
                });
            });
        }

        // Theme Event
        themeToggleBtn.addEventListener('click', toggleTheme);
    }

    // --- Identity Management ---

    function getPersistentUserId() {
        let userId = localStorage.getItem('jade_master_user_id');
        if (!userId) {
            userId = 'user_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
            localStorage.setItem('jade_master_user_id', userId);
        }
        return userId;
    }

    // --- Theme Management ---

    function initTheme() {
        const savedTheme = localStorage.getItem('jade_theme');
        // Default to Dark if null
        if (savedTheme === 'light') {
            document.body.setAttribute('data-theme', 'light');
            updateThemeIcons(true);
        } else {
            // Force dark as default
            document.body.removeAttribute('data-theme');
            localStorage.setItem('jade_theme', 'dark'); // Ensure logic knows it's dark
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
        // Update highlight.js theme for code blocks
        const hljsLink = document.getElementById('highlight-theme');
        if (hljsLink) {
            if (isLight) {
                // Use a light theme (e.g., Atom One Light or similar)
                hljsLink.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-light.min.css';
            } else {
                // Revert to dark
                hljsLink.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css';
            }
        }

        if (isLight) {
            sunIcon.classList.add('hidden');
            moonIcon.classList.remove('hidden');
        } else {
            sunIcon.classList.remove('hidden');
            moonIcon.classList.add('hidden');
        }
    }

    // --- Agent Management ---

    function switchAgent(agent) {
        if (currentAgent === agent) return;
        currentAgent = agent;

        // Update UI
        updateAgentUI();

        // Reload history and start new chat or load latest
        renderHistoryList();

        // Try to find the most recent chat for this agent
        const lastChat = conversations.find(c => (c.agent || 'jade') === currentAgent);
        if (lastChat) {
            loadChat(lastChat.id);
        } else {
            startNewChat();
        }

        // Close sidebar on mobile if open
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
        }
    }

    function updateAgentUI() {
        // Update Sidebar Buttons
        const buttons = agentSwitcher.querySelectorAll('.agent-btn');
        buttons.forEach(btn => {
            if (btn.getAttribute('data-agent') === currentAgent) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update Header Title and Input Placeholder
        if (currentAgent === 'scholar') {
            headerTitle.textContent = 'Scholar Graph';
            userInput.placeholder = 'Envie uma mensagem para Scholar Graph...';
        } else {
            headerTitle.textContent = 'J.A.D.E.';
            userInput.placeholder = 'Envie uma mensagem para J.A.D.E...';
        }
    }

    // --- State & Storage ---

    function init() {
        initTheme();
        loadConversations();

        // Restore agent state if we want persistence (optional, sticking to default for now or infer from last chat)
        // For now, let's stick to default 'jade' or maybe the last used?
        // Let's keep it simple: Start with JADE.

        updateAgentUI();

        // Load initial chat
        const lastChat = conversations.find(c => (c.agent || 'jade') === currentAgent);
        if (lastChat) {
            loadChat(lastChat.id);
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
            timestamp: Date.now(),
            agent: currentAgent // Save current agent to chat
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

        // Ensure we switch the agent context if we load a chat from history (though UI hides others)
        const chatAgent = chat.agent || 'jade';
        if (chatAgent !== currentAgent) {
             // This might happen if we click a link that was somehow visible,
             // but our renderHistoryList filters them.
             // Just in case, update state.
             currentAgent = chatAgent;
             updateAgentUI();
             renderHistoryList(); // Re-render list to match new agent
        }

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

            // Ensure agent is set (for migration of old chats)
            if (!chat.agent) chat.agent = 'jade';

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
                // Try to find another chat for current agent
                const nextChat = conversations.find(c => (c.agent || 'jade') === currentAgent);
                if (nextChat) {
                    loadChat(nextChat.id);
                } else {
                    startNewChat();
                }
            }
        }
    }

    // --- UI Rendering ---

    function renderHistoryList() {
        chatHistoryList.innerHTML = '';
        // Filter conversations by current agent
        const filteredConversations = conversations.filter(c => (c.agent || 'jade') === currentAgent);

        filteredConversations.forEach(chat => {
            const div = document.createElement('div');
            div.className = `history-item ${chat.id === currentChatId ? 'active' : ''}`;

            const span = document.createElement('span');
            span.textContent = chat.title;
            span.style.flex = '1';
            span.style.overflow = 'hidden';
            span.style.textOverflow = 'ellipsis';

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
        const agentName = currentAgent === 'scholar' ? 'Scholar Graph' : 'J.A.D.E.';
        const welcomeText = currentAgent === 'scholar'
            ? 'Olá. Eu sou Scholar Graph, seu assistente de pesquisa. Como posso ajudar em seus estudos?'
            : 'Olá. Eu sou J.A.D.E., sua assistente de inteligência artificial avançada. Como posso ajudar você hoje?';

        const el = document.createElement('div');
        el.className = 'message bot welcome-message';
        el.innerHTML = `
            <div class="avatar">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg>
            </div>
            <div class="content">
            <div class="sender-name">${agentName}</div>
            <div class="text">${welcomeText}</div>
            </div>
        `;
        chatbox.appendChild(el);
    }

    function handleImageSelection() {
        if (imageInput.files && imageInput.files[0]) {
            showImagePreview(imageInput.files[0]);
        }
    }

    function handlePaste(e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let index in items) {
            const item = items[index];
            if (item.kind === 'file' && item.type.includes('image/')) {
                const blob = item.getAsFile();
                // Create a container for the file to mimic input selection if needed,
                // or just use a separate state variable for the pasted image.
                // For simplicity, let's assign it to our file input if possible or just handle it.
                // Since we can't programmatically set file input value easily, we'll need a state variable.
                // Let's reuse the existing logic but we need to support non-input files.

                // Workaround: We will use DataTransfer to set the input files
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(blob);
                imageInput.files = dataTransfer.files;

                showImagePreview(blob);
            }
        }
    }

    function showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreviewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    function clearImagePreview() {
        imageInput.value = '';
        imagePreviewContainer.classList.add('hidden');
    }

    // 🔴 FUNÇÃO CORRIGIDA PARA OS LINKS 🔴
    function renderMarkdown(text) {
        let html;
        if (typeof marked !== 'undefined') {
            html = marked.parse(text);
        } else {
            html = escapeHtml(text);
            html = html.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
            html = html.replace(/\*(.*?)\*/g, '<b>$1</b>');
            html = html.replace(/`(.*?)`/g, '<code>$1</code>');
            html = html.replace(/\n/g, '<br>');
        }

        // AQUI ESTÁ O FIX DO 404
        // Substitui links relativos /generated/ por links absolutos do Render
        const regex = /href="\/generated\/(.*?)"/g;
        html = html.replace(regex, `target="_blank" href="${PROXY_BASE_URL}/generated/$1"`);

        // Garante que links normais abram em nova aba também
        // (opcional, mas bom pra UX)
        html = html.replace(/<a href="http/g, '<a target="_blank" href="http');

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
            addCopyButtons(textEl);
        }
    }

    function addCopyButtons(container) {
        if (!container) return;
        const preBlocks = container.querySelectorAll('pre');

        preBlocks.forEach(pre => {
            if (pre.querySelector('.copy-btn')) return; // Already has button

            const btn = document.createElement('button');
            btn.className = 'copy-btn';
            btn.textContent = 'Copiar';
            btn.addEventListener('click', () => {
                const code = pre.querySelector('code');
                const text = code ? code.innerText : pre.innerText;

                navigator.clipboard.writeText(text).then(() => {
                    const originalText = btn.textContent;
                    btn.textContent = 'Copiado!';
                    setTimeout(() => {
                        btn.textContent = originalText;
                    }, 2000);
                }).catch(err => {
                    console.error('Falha ao copiar:', err);
                });
            });

            pre.appendChild(btn);
        });
    }

    function speakText(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();

            // Clean markdown for speech
            const plainText = text.replace(/[#*`\[\]]/g, '').replace(/\(http.*?\)/g, '');

            const utterance = new SpeechSynthesisUtterance(plainText);
            utterance.lang = 'pt-BR';
            const voices = window.speechSynthesis.getVoices();
            const preferredVoice = voices.find(v => v.lang.includes('pt-BR') && v.name.includes('Google'));
            if (preferredVoice) utterance.voice = preferredVoice;

            // Visualizer Events
            utterance.onstart = () => {
                if(audioVisualizer) audioVisualizer.classList.remove('hidden');
            };
            utterance.onend = () => {
                if(audioVisualizer) audioVisualizer.classList.add('hidden');
            };
            utterance.onerror = () => {
                if(audioVisualizer) audioVisualizer.classList.add('hidden');
            };

            window.speechSynthesis.speak(utterance);
        }
    }

    // --- Speech Recognition ---
    let recognition = null;

    function toggleVoiceRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            alert('Seu navegador não suporta reconhecimento de voz.');
            return;
        }

        if (recognition && recognition.started) {
            recognition.stop();
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'pt-BR';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
            voiceBtn.classList.add('listening'); // Add CSS class for pulsing effect
            voiceBtn.style.color = '#ef4444'; // Red to indicate recording
            recognition.started = true;
        };

        recognition.onend = () => {
            voiceBtn.classList.remove('listening');
            voiceBtn.style.color = ''; // Reset color
            recognition.started = false;
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value += (userInput.value ? ' ' : '') + transcript;
            userInput.focus();
            // Trigger auto-resize
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            recognition.stop();
        };

        recognition.start();
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        let image_base64 = null;
        if (!message && imageInput.files.length === 0) return;

        if (!currentChatId) startNewChat();

        // Use global currentAgent instead of selector
        const agentName = currentAgent === 'scholar' ? 'Scholar Graph' : 'J.A.D.E.';

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

        const masterUserId = getPersistentUserId();

        try {
            console.log(`📡 Enviando para: ${API_URL}`);

            const resp = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_input: message,
                    image_base64: image_base64,
                    user_id: masterUserId,
                    agent_type: currentAgent // Use global currentAgent
                })
            });

            if (!resp.ok) {
                 const errorText = await resp.text();
                 throw new Error(`HTTP ${resp.status} - ${errorText.substring(0, 100)}`);
            }

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
            const errorText = `Falha ao conectar (${err.message}).`;
            updateBotMessage(jadeTypingMessage, errorText);
            saveMessageToCurrentChat(agentName, errorText);
        }
    }

    setupEventListeners();
    init();

})();
