css = """
<style>
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: #DCF8C6;
    justify-content: flex-end;
}
.chat-message.bot {
    background-color: #F1F0F0;
    justify-content: flex-start;
}
.chat-message img {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin: 0 10px;
}
.chat-message .message {
    max-width: 80%;
    word-wrap: break-word;
}
</style>
"""
user_template = """
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <img src="https://i.ibb.co/vxNMpMs6/download.jpg" alt="user">
</div>
"""

bot_template = """
<div class="chat-message bot">
    <img src="https://i.ibb.co/1JT6V4c1/download.png" alt="bot">
    <div class="message">{{MSG}}</div>
</div>
"""