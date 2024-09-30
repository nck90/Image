import React from 'react';
import './App.css';
import ImageComparison from './components/ImageComparison';

const App: React.FC = () => {
  return (
    <div className="App">
      <header className="App-header">
        <h1>이미지 유사도 분석 서비스</h1>
      </header>
      <main>
        <ImageComparison />
      </main>
    </div>
  );
};

export default App;