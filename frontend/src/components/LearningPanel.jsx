import React from "react";
const topics = [
  "What is machine learning?",
  "Explain neural networks",
  "How does self-attention work?",
  "What is a Transformer?",
];

export default function LearningPanel({
  onSelectTopic = () => {},
  onNewChat = () => {},
  questionCount = 0,
}) {
  return (
    <aside className="learning-sidebar">
      <div className="learning-panel">
        <section className="learning-card">
          <h3>💡 Suggested Topics</h3>

          {topics.map((topic) => (
            <button
              key={topic}
              type="button"
              onClick={() => onSelectTopic(topic)}
            >
              {topic}
            </button>
          ))}
        </section>

        <section className="learning-card">
          <h3>🎯 Project Roadmap</h3>
          <p style={{ fontSize: "13px", lineHeight: 1.8 }}>
            Data preparation
            <br />
            ↓
            <br />
            Tokenization and datasets
            <br />
            ↓
            <br />
            Fine-tuning and evaluation
            <br />
            ↓
            <br />
            FastAPI and authentication
            <br />
            ↓
            <br />
            Testing and documentation
          </p>
        </section>

        <section className="learning-card">
          <h3>📘 Learning Tip</h3>
          <p style={{ fontSize: "13px", lineHeight: 1.7 }}>
            Ask why each component exists, what goes into it, what
            comes out, and how it can fail.
          </p>
        </section>

        <section className="learning-card">
          <h3>💬 Conversation</h3>
          <p style={{ fontSize: "13px", color: "#71809c" }}>
            {questionCount} questions in this session
          </p>

          <button type="button" onClick={onNewChat}>
            + Start New Chat
          </button>
        </section>
      </div>
    </aside>
  );
}
