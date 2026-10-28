
import React, { useState } from "react";

export default function ChatInput({ onSend, disabled = false }) {
  const [input, setInput] = useState("");

  function handleSubmit(event) {
    event.preventDefault();

    const message = input.trim();

    if (!message || disabled) return;

    onSend(message);
    setInput("");
  }

  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <div className="chat-input-area">
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about AI, ML, NLP, or Transformers..."
          aria-label="Your message"
          rows={2}
          disabled={disabled}
        />

        <button type="submit" disabled={!input.trim() || disabled}>
          Send
        </button>
      </form>

      <p style={{ fontSize: "12px", color: "#71809c", marginBottom: 0 }}>
        Press Enter to send · Shift + Enter for a new line
      </p>
    </div>
  );
}
