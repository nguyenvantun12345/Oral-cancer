'use client';

import Layout from '../../components/Layout';

const About = () => {
  return (
    <Layout>
      <div className="page-wrapper">
        <div className="card">
          <h1 className="page-title">About This Project</h1>
          <p className="page-text">
            This web application helps in the early diagnosis of oral cancer using advanced machine learning techniques.
            Users can upload images, and our system will analyze them using a deep learning model powered by CBAM and
            VAE. This project is designed to assist medical professionals and raise awareness about early detection.
          </p>
        </div>
      </div>
    </Layout>
  );
};

export default About;
