import React from "react";
export default function MessageBubble({ role, content }) {
  const isUser = role === "user";

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? "👤" : "🤖"}
      </div>

      <div className="message-bubble">
        {content.split("\n").map((line, index) => (
          <span key={index}>
            {line || "\u00A0"}
            <br />
          </span>
        ))}
      </div>
    </div>
  );
}
