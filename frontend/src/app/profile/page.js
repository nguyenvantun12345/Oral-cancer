'use client';

import { useState } from 'react';
import Layout from '../../components/Layout';

const initialUser = {
  name: 'John Doe',
  email: 'john.doe@example.com',
  phone: '123-456-7890',
};

export default function ProfilePage() {
  const [user, setUser] = useState(initialUser);
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState(user);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSave = () => {
    setUser(formData);
    setEditMode(false);

    // TODO: Send updated data to backend (FastAPI + MongoDB)
    // fetch('/api/update-profile', { method: 'POST', body: JSON.stringify(formData) })
  };

  const handleCancel = () => {
    setFormData(user);
    setEditMode(false);
  };

  return (
    <Layout>
      <div className="page-wrapper">
        <div className="profile-card">
          <h1 className="page-title">Your Profile</h1>

          {editMode ? (
            <form className="profile-form">
              <label htmlFor="name">Name</label>
              <input name="name" value={formData.name} onChange={handleChange} />

              <label htmlFor="email">Email</label>
              <input name="email" type="email" value={formData.email} onChange={handleChange} />

              <label htmlFor="phone">Phone</label>
              <input name="phone" type="tel" value={formData.phone} onChange={handleChange} />

              <div className="profile-buttons">
                <button type="button" className="save-btn" onClick={handleSave}>Save</button>
                <button type="button" className="cancel-btn" onClick={handleCancel}>Cancel</button>
              </div>
            </form>
          ) : (
            <div>
              <p><strong>Name:</strong> {user.name}</p>
              <p><strong>Email:</strong> {user.email}</p>
              <p><strong>Phone:</strong> {user.phone}</p>
              <button className="edit-btn mt-4" onClick={() => setEditMode(true)}>Edit Profile</button>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
