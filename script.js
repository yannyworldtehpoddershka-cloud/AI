document.addEventListener('DOMContentLoaded', function() {
    const chatWindow = document.getElementById('chat-window');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');

    // Predefined responses for simulation
    const responses = [
        "Это интересно! Расскажите подробнее.",
        "Я понимаю. Что вы думаете об этом?",
        "Хорошо, давайте продолжим разговор.",
        "Спасибо за информацию. Есть ли что-то еще?",
        "Я нейронная сеть Qwen3, и я здесь, чтобы помочь!"
    ];

    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
        messageDiv.innerHTML = isUser ? `<strong>Вы:</strong> ${message}` : `<strong>Qwen3:</strong> ${message}`;
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function simulateBotResponse() {
        setTimeout(() => {
            const randomResponse = responses[Math.floor(Math.random() * responses.length)];
            addMessage(randomResponse);
        }, 1000); // Simulate typing delay
    }

    function sendMessage() {
        const message = messageInput.value.trim();
        if (message) {
            addMessage(message, true);
            messageInput.value = '';
            simulateBotResponse();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
