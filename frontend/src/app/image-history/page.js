'use client';

import React, { useEffect, useState } from 'react';
import Layout from '../../components/Layout';

export default function ImageHistoryPage() {
  const [uploads, setUploads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchHistory = async () => {
      setLoading(true);
      setError('');
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setError('Authentication required. Please log in.');
          setLoading(false);
          return;
        }
        const res = await fetch('http://localhost:8000/patients/me/history', {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });

        if (!res.ok) {
          let errorMsg = 'Failed to fetch history.';
          try {
            const errJson = await res.json();
            errorMsg = errJson.detail || errorMsg;
          } catch {}
          setError(errorMsg);
          setLoading(false);
          return;
        }

        const data = await res.json();
        // Assuming backend returns: { histories: [ { image, date, ... }, ... ] }
        setUploads(data.histories || []);
      } catch (err) {
        setError('An error occurred while fetching history.');
      }
      setLoading(false);
    };

    fetchHistory();
  }, []);

  return (
    <Layout>
      <div className="page-wrapper">
        <div className="w-full max-w-6xl">
          <h1 className="page-title">Image Upload History</h1>
          {loading && <div>Loading...</div>}
          {error && <div className="text-red-600">{error}</div>}
          <div className="image-grid">
            {uploads.length === 0 && !loading && !error && (
              <div>No uploads found.</div>
            )}
            {uploads.map((img, idx) => (
              <div key={img.image_id || idx} className="image-card">
                <img src={img.image} alt={`Upload ${idx + 1}`} className="image-thumb" />
                <p className="image-caption">
                  Uploaded on: {Array.isArray(img.date) ? img.date[0] : img.date}
                </p>
                {/* You can add more details as needed */}
              </div>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
}