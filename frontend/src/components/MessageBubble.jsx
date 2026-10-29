
import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function MessageBubble({ role, content }) {
  const isUser = role === "user";

  // Some model answers place code fences in the middle of a paragraph.
  // Move fences onto separate lines so Markdown can recognize code blocks.
  const formattedContent = isUser
    ? content
    : content.replace(
        /[ \t]*```([a-zA-Z0-9_+-]*)[ \t]*/g,
        "\n```$1\n"
      );

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? "👤" : "🤖"}
      </div>

      <div className="message-bubble">
        {isUser ? (
          content.split("\n").map((line, index) => (
            <React.Fragment key={index}>
              {line || "\u00A0"}
              <br />
            </React.Fragment>
          ))
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {formattedContent}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}
