"use client";

import React, { useState } from 'react';
import axios from 'axios';

const ImageUploader = ({ onUploadComplete }) => {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('image', image);

    try {
      // 👉 Imgur example
      const response = await axios.post('https://api.imgur.com/3/image', formData, {
        headers: {
          Authorization: 'Client-ID 6352b06a8a2c101',
        },
      });

      const url = response.data.data.link;
      setUploadedUrl(url);
      onUploadComplete && onUploadComplete(url);

    } catch (error) {
      console.error('Upload failed:', error);
      alert('Image upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {previewUrl && <img src={previewUrl} alt="Preview" width={200} style={{ marginTop: '10px' }} />}
      <button onClick={handleUpload} disabled={!image || loading} style={{ marginTop: '10px' }}>
        {loading ? 'Uploading...' : 'Upload Image'}
      </button>
      {uploadedUrl && (
        <div style={{ marginTop: '10px' }}>
          <p>Image uploaded:</p>
          <a href={uploadedUrl} target="_blank" rel="noopener noreferrer">{uploadedUrl}</a>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;

