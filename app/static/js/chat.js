// Chatbot JavaScript Controller

document.addEventListener('DOMContentLoaded', () => {
    // ── Elements ─────────────────────────────────────────────────────────────
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');
    const sendBtn = document.getElementById('send-btn');
    const messageCount = document.getElementById('message-count');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const exportTxtBtn = document.getElementById('export-txt-btn');
    const suggestionsContainer = document.getElementById('suggestions-container');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');

    // Initialize Lucide Icons
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // ── State Variables ──────────────────────────────────────────────────────
    let chatHistory = []; // format: { role: 'user'|'bot', content: str, sources: Array, timestamp: str }
    let isGenerating = false;

    // ── Theme Management ─────────────────────────────────────────────────────
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeToggleUI(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeToggleUI(newTheme);
    });

    function updateThemeToggleUI(theme) {
        const sunIcon = themeToggleBtn.querySelector('.theme-icon-light');
        const moonIcon = themeToggleBtn.querySelector('.theme-icon-dark');
        if (theme === 'light') {
            sunIcon.style.display = 'none';
            moonIcon.style.display = 'block';
        } else {
            sunIcon.style.display = 'block';
            moonIcon.style.display = 'none';
        }
    }



    // ── Textarea Auto-Resize ─────────────────────────────────────────────────
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight - 16) + 'px';
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // ── Suggested Questions ──────────────────────────────────────────────────
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            if (isGenerating) return;
            userInput.value = chip.textContent;
            userInput.dispatchEvent(new Event('input'));
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // ── Export Chat ──────────────────────────────────────────────────────────
    exportTxtBtn.addEventListener('click', () => exportChat('text'));

    async function exportChat(format) {
        if (chatHistory.length === 0) {
            alert('No conversation history to export.');
            return;
        }
        try {
            const response = await fetch('/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chat_history: chatHistory, format: format })
            });

            if (!response.ok) throw new Error('Failed to export conversation');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'petpooja_chat.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Export error:', error);
            alert('Error exporting conversation: ' + error.message);
        }
    }

    // ── Clear Chat ───────────────────────────────────────────────────────────
    clearChatBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear the conversation?')) {
            chatHistory = [];
            // Remove all messages except the first system/welcome message
            const messages = Array.from(chatWindow.querySelectorAll('.message'));
            messages.forEach((msg, idx) => {
                if (idx > 0) msg.remove();
            });
            updateStats();
            suggestionsContainer.style.display = 'block';
        }
    });

    // ── Chat Submission & Stream Processing ──────────────────────────────────
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || isGenerating) return;

        // Hide suggestions on first message
        suggestionsContainer.style.display = 'none';

        // 1. Render User Message
        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        // 2. Prepare History for Request
        const historyPayload = chatHistory.map(h => ({
            role: h.role === 'bot' ? 'bot' : h.role,
            content: h.content
        }));

        // Add user message to history
        const userMsgObj = {
            role: 'user',
            content: text,
            timestamp: new Date().toISOString()
        };
        chatHistory.push(userMsgObj);
        updateStats();

        // 3. Render Empty Assistant Message with Loading Indicator
        const botMessageElement = appendMessage('bot', '');
        const messageCard = botMessageElement.querySelector('.message-card');
        const contentDiv = botMessageElement.querySelector('.message-content');
        
        // Show Typing Indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        contentDiv.appendChild(typingIndicator);
        scrollChatToBottom();

        // Set generating state
        toggleGeneratingState(true);

        let fullAnswer = '';
        let sources = [];

        try {
            // 4. Send POST Request and read SSE stream
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: text,
                    chat_history: historyPayload
                })
            });

            if (!response.ok) throw new Error('Failed to connect to assistant');

            // Remove typing indicator as stream starts
            contentDiv.removeChild(typingIndicator);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let partialData = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                partialData += decoder.decode(value, { stream: true });
                const lines = partialData.split('\n\n');
                
                // Store the last partial block back to decode later if not complete
                partialData = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const rawJson = line.slice(6);
                        try {
                            const parsed = JSON.parse(rawJson);
                            
                            if (parsed.error) {
                                throw new Error(parsed.error);
                            }

                            if (parsed.token) {
                                fullAnswer += parsed.token;
                                contentDiv.innerHTML = formatMarkdown(fullAnswer);
                                scrollChatToBottom();
                            }

                            if (parsed.done) {
                                sources = parsed.sources;
                                fullAnswer = parsed.answer || fullAnswer; // fallback to accumulated stream if final answer empty
                                contentDiv.innerHTML = formatMarkdown(fullAnswer);
                                
                                // Render sources if available
                                if (sources && sources.length > 0) {
                                    appendSources(messageCard, sources);
                                }
                                scrollChatToBottom();
                            }
                        } catch (jsonErr) {
                            console.error('Failed to parse SSE JSON:', jsonErr, line);
                        }
                    }
                }
            }

            // Save assistant response to history
            chatHistory.push({
                role: 'bot',
                content: fullAnswer,
                sources: sources,
                timestamp: new Date().toISOString()
            });
            updateStats();

        } catch (err) {
            console.error('Chat stream error:', err);
            // Handle error in UI
            if (contentDiv.contains(typingIndicator)) {
                contentDiv.removeChild(typingIndicator);
            }
            contentDiv.innerHTML = `<span style="color: #ef4444;">⚠️ Error: ${err.message}. Please try again.</span>`;
        } finally {
            toggleGeneratingState(false);
            scrollChatToBottom();
        }
    });

    // ── Helper Functions ─────────────────────────────────────────────────────

    function appendMessage(role, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        const wrapperDiv = document.createElement('div');
        wrapperDiv.className = 'message-wrapper';

        const card = document.createElement('div');
        card.className = 'message-card';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = role === 'user' ? escapeHTML(text) : formatMarkdown(text);

        card.appendChild(content);
        wrapperDiv.appendChild(card);
        messageDiv.appendChild(wrapperDiv);

        chatWindow.appendChild(messageDiv);
        scrollChatToBottom();
        return messageDiv;
    }

    function appendSources(cardElement, sourcesList) {
        const container = document.createElement('div');
        container.className = 'sources-container';

        const label = document.createElement('span');
        label.className = 'sources-label';
        label.innerHTML = '<i data-lucide="book-open" style="width: 12px; height: 12px;"></i> Sources:';
        container.appendChild(label);

        sourcesList.forEach(src => {
            const badge = document.createElement('span');
            badge.className = 'source-badge';
            badge.innerHTML = `<i data-lucide="file" style="width: 10px; height: 10px;"></i> ${src}`;
            container.appendChild(badge);
        });

        cardElement.appendChild(container);
        
        // Re-run Lucide to render newly added icons
        if (window.lucide) {
            window.lucide.createIcons({
                attrs: {
                    class: 'lucide-icon'
                }
            });
        }
    }

    function toggleGeneratingState(state) {
        isGenerating = state;
        sendBtn.disabled = state;
        userInput.disabled = state;
        if (state) {
            sendBtn.innerHTML = '<i data-lucide="loader" class="spin"></i>';
        } else {
            sendBtn.innerHTML = '<i data-lucide="send"></i>';
        }
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    function scrollChatToBottom() {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function updateStats() {
        // Calculate number of user turns
        const count = chatHistory.filter(h => h.role === 'user').length;
        messageCount.textContent = count;
    }

    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * A very simple markdown formatter to convert markdown into standard HTML.
     */
    function formatMarkdown(text) {
        if (!text) return '';
        
        let html = text;

        // Escape HTML tags to prevent XSS (but preserve markdown syntax)
        html = escapeHTML(html);

        // Code blocks: ```python\ncode\n```
        html = html.replace(/```(?:[a-zA-Z0-9-]*)\n([\s\S]*?)```/g, (match, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });

        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold: **text**
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Lists: lines starting with "- " or "* "
        const lines = html.split('\n');
        let inList = false;
        const formattedLines = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const listMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
            
            if (listMatch) {
                if (!inList) {
                    formattedLines.push('<ul>');
                    inList = true;
                }
                formattedLines.push(`<li>${listMatch[2]}</li>`);
            } else {
                if (inList) {
                    formattedLines.push('</ul>');
                    inList = false;
                }
                formattedLines.push(line);
            }
        }
        if (inList) {
            formattedLines.push('</ul>');
        }
        html = formattedLines.join('\n');

        // Paragraphs: Replace multiple newlines with paragraph structures
        html = html.split(/\n{2,}/g).map(p => {
            p = p.trim();
            if (!p) return '';
            // If the paragraph is already list wrappers, headers, pre blocks, do not wrap in p
            if (p.startsWith('<ul>') || p.startsWith('<li>') || p.startsWith('<pre>') || p.startsWith('<h3>') || p.startsWith('<h2>')) {
                return p;
            }
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');

        return html;
    }
});
