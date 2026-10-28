
import React from "react";

const navigationItems = [
  { id: "chat", icon: "💬", label: "AI Chat" },
  { id: "topics", icon: "📚", label: "Learning Topics" },
  { id: "progress", icon: "📈", label: "My Progress" },
  { id: "settings", icon: "⚙️", label: "Settings" },
];

export default function Sidebar({
  activeSection = "chat",
  onNavigate = () => {},
}) {
  return (
    <aside className="sidebar">
      <p className="sidebar-title">Workspace</p>

      {navigationItems.map((item) => (
        <button
          key={item.id}
          type="button"
          className={activeSection === item.id ? "active" : ""}
          onClick={() => onNavigate(item.id)}
        >
          <span aria-hidden="true">{item.icon}</span>
          {item.label}
        </button>
      ))}

      <div className="sidebar-footer">
        <div>AI/ML Learning Assistant</div>
        <div>Learn by understanding</div>
      </div>
    </aside>
  );
}
