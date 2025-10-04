import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';

function App() {
  const [userId] = useState(() => `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [isConnected, setIsConnected] = useState(false);

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-primary-600 via-secondary-600 to-primary-800">
      <header className="bg-white/10 backdrop-blur-lg px-8 py-4 shadow-lg">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-white">Biostellar AI Chat</h1>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full transition-all duration-300 ${
              isConnected 
                ? 'bg-green-400 shadow-lg shadow-green-400/50' 
                : 'bg-red-400 shadow-lg shadow-red-400/50'
            }`}></div>
            <span className="text-white text-sm font-medium">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>
      <main className="flex-1 flex justify-center items-center p-8">
        <ChatInterface 
          userId={userId} 
          onConnectionChange={setIsConnected}
        />
      </main>
    </div>
  );
}

export default App;
