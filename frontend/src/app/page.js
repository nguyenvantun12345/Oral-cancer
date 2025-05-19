'use client';

import Layout from '../components/Layout';
import ImageUploader from '../components/ImageUploader';
import './globals.css'; // Import your CSS file

import React from 'react';
import OralCancerPredictor from '../components/OralCancerPredictor';

// Assuming you might have Header and Footer components in separate files too
// import YourHeaderComponent from './YourHeaderComponent';
// import YourFooterComponent from './YourFooterComponent';

// App component to render the ObjectRecognizer.
// This is your main application entry point or a page component.
export default function App() {
  return (
    // It's common to have a root div for the entire app or page
    // Global styles like background color or min-height can be applied here
    // or in your main CSS/Tailwind setup.
    <Layout>
      <div className="min-h-screen bg-gray-100">
        <main>
          <OralCancerPredictor />
        </main>

        {/* <YourFooterComponent /> */}
      </div>
    </Layout>
  );
}

