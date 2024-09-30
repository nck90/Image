import React, { useState } from 'react';
import axios from 'axios';
import './ImageComparison.css';

const ImageComparison: React.FC = () => {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleImage1Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setImage1(e.target.files[0]);
    }
  };

  const handleImage2Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setImage2(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!image1 || !image2) {
      setError('두 개의 이미지를 모두 선택해주세요.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);

    try {
      const response = await axios.post('/api/compare', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error: any) {
      setError('오류가 발생했습니다: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="comparison-container">
      <h2>이미지 유사도 비교</h2>
      <form onSubmit={handleSubmit} className="form-section">
        <div className="input-group">
          <label htmlFor="image1">이미지 1:</label>
          <input type="file" id="image1" accept="image/*" onChange={handleImage1Change} required />
        </div>
        <div className="input-group">
          <label htmlFor="image2">이미지 2:</label>
          <input type="file" id="image2" accept="image/*" onChange={handleImage2Change} required />
        </div>
        <button type="submit" className="compare-btn" disabled={loading}>
          {loading ? '비교 중...' : '비교하기'}
        </button>
      </form>

      {loading && <p className="loading">이미지를 비교하는 중입니다...</p>}

      {error && <p className="error-message">{error}</p>}

      {result && (
        <div className="result-section">
          <h3>유사도 결과</h3>
          <p><strong>픽셀 유사도:</strong> {result.pixel_similarity.toFixed(2)}%</p>
          <p><strong>SSIM 유사도:</strong> {result.ssim_similarity.toFixed(2)}%</p>
          <p><strong>pHash 유사도:</strong> {result.phash_similarity.toFixed(2)}%</p>
          <p><strong>종합 유사도:</strong> {result.total_similarity.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
};

export default ImageComparison;