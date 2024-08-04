// Handle opening and closing of the chatbot
document.getElementById('chatbot-icon').addEventListener('click', function () {
    document.getElementById('chatbot-container').style.display = 'flex';
});

document.getElementById('chatbot-close').addEventListener('click', function () {
    document.getElementById('chatbot-container').style.display = 'none';
});
