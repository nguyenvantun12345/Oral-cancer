'use client';

import Layout from '../components/Layout';
import ImageUploader from '../components/ImageUploader';
import './globals.css'; // Import your CSS file

const Home = () => {
  const handleUploadComplete = (url) => {
    console.log('Uploaded image URL:', url);
    // You can send the URL to your backend for analysis if needed
  };

  return (
    <Layout>
      <div className="center-container">
        <h1>Upload an Image for Oral Cancer Diagnosis</h1>
        <ImageUploader onUploadComplete={handleUploadComplete} />
      </div>
    </Layout>
  );
};

export default Home;
