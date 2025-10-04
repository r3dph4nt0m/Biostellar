import React, { useState, useEffect, useRef } from 'react';

const ChatInterface = ({ userId, onConnectionChange }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Connect to WebSocket
    const connectWebSocket = () => {
      const ws = new WebSocket(`ws://localhost:8000/ws/${userId}`);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        onConnectionChange(true);
        addMessage('system', 'Connected to chat server');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'typing') {
            setIsTyping(true);
          } else if (data.type === 'ai_message') {
            setIsTyping(false);
            addMessage('ai', data.message, data.timestamp);
          } else if (data.error) {
            setIsTyping(false);
            addMessage('error', data.error);
          } else {
            // Handle other message types
            addMessage('system', data.message || event.data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          addMessage('error', 'Error parsing server message');
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        onConnectionChange(false);
        addMessage('system', 'Disconnected from chat server');
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        onConnectionChange(false);
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [userId, onConnectionChange]);

  const addMessage = (type, content, timestamp = null) => {
    const message = {
      id: Date.now() + Math.random(),
      type,
      content,
      timestamp: timestamp || new Date().toISOString()
    };
    setMessages(prev => [...prev, message]);
  };

  const sendMessage = () => {
    if (!inputMessage.trim() || !isConnected) return;

    const message = inputMessage.trim();
    setInputMessage('');

    // Add user message to UI immediately for better UX
    addMessage('user', message);

    // Send message via WebSocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        message: message,
        timestamp: new Date().toISOString()
      }));
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="w-full max-w-4xl h-[600px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 bg-slate-50 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[70%] px-4 py-3 rounded-2xl ${
              message.type === 'user' 
                ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white rounded-br-md' 
                : message.type === 'ai'
                ? 'bg-white text-gray-800 border border-gray-200 rounded-bl-md shadow-sm'
                : message.type === 'system'
                ? 'bg-gray-100 text-gray-600 text-center text-sm rounded-lg max-w-[50%] mx-auto'
                : 'bg-red-50 text-red-600 border border-red-200 text-center text-sm rounded-lg max-w-[50%] mx-auto'
            }`}>
              <div className="break-words leading-relaxed">{message.content}</div>
              <div className={`text-xs opacity-70 mt-1 ${
                message.type === 'user' ? 'text-right' : 'text-left'
              }`}>
                {formatTime(message.timestamp)}
              </div>
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 border border-gray-200 rounded-2xl rounded-bl-md shadow-sm px-4 py-3">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-typing"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-typing" style={{animationDelay: '0.2s'}}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-typing" style={{animationDelay: '0.4s'}}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input Area */}
      <div className="p-6 bg-white border-t border-gray-200">
        <div className="flex gap-3 items-end">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isConnected ? "Type your message..." : "Connecting..."}
            disabled={!isConnected}
            className="flex-1 resize-none border-2 border-gray-200 rounded-xl px-4 py-3 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-200 min-h-[44px] max-h-32 disabled:bg-gray-50 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors"
            rows="1"
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || !isConnected}
            className="bg-gradient-to-r from-primary-500 to-secondary-500 text-white px-6 py-3 rounded-xl font-semibold hover:from-primary-600 hover:to-secondary-600 disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0 min-w-[80px]"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
