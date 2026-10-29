import React, { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import ChatWindow from "./components/ChatWindow.jsx";
import LearningPanel from "./components/LearningPanel.jsx";

const suggestedTopics = [
  "What is machine learning?",
  "Explain neural networks",
  "How does self-attention work?",
  "What is a Transformer?",
];



export default function App() {
  const [activeSection, setActiveSection] = useState("chat");
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      content:
        "Hello! 👋 I'm your AI/ML Learning Assistant.\n\nAsk me about machine learning, NLP, Transformers, PyTorch, training, evaluation, or the engineering behind this project.\n\nWhat would you like to learn today?",
    },
  ]);
  const [isTyping, setIsTyping] = useState(false);
async function sendMessage(text) {
  const question = text.trim();

  if (!question || isTyping) return;

  const userMessage = {
    id: Date.now(),
    role: "user",
    content: question,
  };

  setMessages((current) => [...current, userMessage]);
  setIsTyping(true);

  try {
    const response = await fetch("http://127.0.0.1:8000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: question,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || `Request failed with status ${response.status}`
      );
    }

    if (typeof data.reply !== "string") {
      throw new Error("The API returned an invalid response.");
    }

    setMessages((current) => [
      ...current,
      {
        id: Date.now(),
        role: "assistant",
        content: data.reply,
      },
    ]);
  } catch (error) {
    console.error("Chat request failed:", error);

    setMessages((current) => [
      ...current,
      {
        id: Date.now(),
        role: "assistant",
        content:
          "I couldn't connect to the AI service. Please check that the backend is running and try again.",
      },
    ]);
  } finally {
    setIsTyping(false);
  }
}


  function startNewChat() {
    setMessages([
      {
        id: Date.now(),
        role: "assistant",
        content:
          "New conversation started! What AI/ML concept would you like to explore?",
      },
    ]);
    setActiveSection("chat");
    setIsTyping(false);
  }

  function renderSection() {
    if (activeSection === "topics") {
      return (
        <section className="section-placeholder">
          <h2>Learning Topics 📚</h2>
          <p>Choose a topic to explore with your learning assistant.</p>

          {suggestedTopics.map((topic) => (
            <button
              key={topic}
              type="button"
              onClick={() => {
                setActiveSection("chat");
                sendMessage(topic);
              }}
            >
              {topic}
            </button>
          ))}
        </section>
      );
    }

    if (activeSection === "progress") {
      const questionCount = messages.filter(
        (message) => message.role === "user"
      ).length;

      return (
        <section className="section-placeholder">
          <h2>My Learning Progress 📈</h2>
          <p>Questions in this conversation: {questionCount}</p>
          <p>
            Suggested next goals: tokenization, embeddings, attention,
            training, and model evaluation.
          </p>
          <p>
            This is session-only information; progress is not saved to a
            database yet.
          </p>
        </section>
      );
    }

    if (activeSection === "settings") {
      return (
        <section className="section-placeholder">
          <h2>Settings ⚙️</h2>
          <p>
            The frontend is currently running in demo mode. Backend
            connection and authentication will be configured during
            integration.
          </p>
          <p>
            API keys must be verified by the backend and must not be
            hardcoded into frontend source code.
          </p>
          <button type="button" onClick={startNewChat}>
            Start a new chat
          </button>
        </section>
      );
    }

    return (
      <ChatWindow
        messages={messages}
        onSend={sendMessage}
        isTyping={isTyping}
      />
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            🧠
          </div>

          <div>
            <h1>AI/ML Learning Assistant</h1>
            <p>Your personal AI and machine learning tutor</p>
          </div>
        </div>

        <div className="connection-status">
          <span className="status-dot" />
          <span>Frontend demo</span>
        </div>
      </header>

      <div className="app-layout">
        <Sidebar
          activeSection={activeSection}
          onNavigate={setActiveSection}
        />

        <main className="main-content">{renderSection()}</main>

        <LearningPanel
          onSelectTopic={(topic) => {
            if (isTyping) return;
            setActiveSection("chat");
            sendMessage(topic);
          }}
          onNewChat={startNewChat}
          questionCount={
            messages.filter((message) => message.role === "user").length
          }
        />
      </div>
    </div>
  );
}
