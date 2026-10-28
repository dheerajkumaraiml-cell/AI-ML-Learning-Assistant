import React from "react";
import MessageBubble from "./MessageBubble.jsx";
import ChatInput from "./ChatInput.jsx";

export default function ChatWindow({
  messages = [],
  onSend,
  isTyping = false,
}) {
  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.length === 1 && (
          <div className="welcome-message">
            <h2>Welcome to your learning workspace! 🚀</h2>
            <p>
              Ask questions about machine learning, NLP, Transformers,
              model training, evaluation, or the engineering behind
              this project.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            role={message.role}
            content={message.content}
          />
        ))}

        {isTyping && (
          <div className="message-row assistant">
            <div className="message-avatar" aria-hidden="true">
              🤖
            </div>
            <div className="message-bubble" aria-live="polite">
              Preparing demo response...
            </div>
          </div>
        )}
      </div>

      <ChatInput onSend={onSend} disabled={isTyping} />
    </div>
  );
}
