'use client';

import Layout from '../../components/Layout';

const dummyUploads = [
  {
    id: 1,
    url: 'https://via.placeholder.com/300x200?text=Image+1',
    date: '2025-05-10',
  },
  {
    id: 2,
    url: 'https://via.placeholder.com/300x200?text=Image+2',
    date: '2025-05-12',
  },
  {
    id: 3,
    url: 'https://via.placeholder.com/300x200?text=Image+3',
    date: '2025-05-14',
  },
];

export default function ImageHistoryPage() {
  return (
    <Layout>
      <div className="page-wrapper">
        <div className="w-full max-w-6xl">
          <h1 className="page-title">Image Upload History</h1>
          <div className="image-grid">
            {dummyUploads.map((img) => (
              <div key={img.id} className="image-card">
                <img src={img.url} alt={`Upload ${img.id}`} className="image-thumb" />
                <p className="image-caption">Uploaded on: {img.date}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
}
