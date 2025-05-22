'use client';

import { useState, useEffect } from 'react';
import Layout from '../../components/Layout';

export default function ProfilePage() {
  const [user, setUser] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState({ name: '', email: '', phone: '', birthdate: '' });
  const [error, setError] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setError('You must be logged in.');
      return;
    }

    fetch('http://localhost:8000/auth/me', {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(async (res) => {
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Failed to load profile');
        }
        return res.json();
      })
      .then((data) => {
        setUser(data);
        setFormData({
          name: data.name || '',
          email: data.email || '',
          phone: data.phone || '',
          birthdate: data.birthdate || '',  // add birthdate field
        });
      })
      .catch((err) => setError(err.message));
  }, []);

  const handleChange = (e) => {
    setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSave = async () => {
    setError(null);
    const token = localStorage.getItem('token');
    try {
      const res = await fetch('http://localhost:8000/patients/me', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(formData)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to update profile');
      }

      const updated = await res.json();
      setUser(updated);
      setFormData({
        name: updated.name || '',
        email: updated.email || '',
        phone: updated.phone || '',
        birthdate: updated.birthdate || '',
      });
      setEditMode(false);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleCancel = () => {
    setFormData(user);
    setEditMode(false);
    setError(null);
  };

  if (!user) {
    return (
      <Layout>
        <div className="p-4">{error ? error : 'Loading profile...'}</div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="page-wrapper">
        <div className="profile-card">
          <h1 className="page-title">Your Profile</h1>

          {error && <p className="text-red-500">{error}</p>}

          {editMode ? (
            <form onSubmit={(e) => e.preventDefault()} className="profile-form">
              <label htmlFor="name">Name</label>
              <input name="name" value={formData.name} onChange={handleChange} />

              <label htmlFor="email">Email</label>
              <input name="email" type="email" value={formData.email} onChange={handleChange} />

              <label htmlFor="phone">Phone</label>
              <input name="phone" type="tel" value={formData.phone} onChange={handleChange} />

              <label htmlFor="birthdate">Birthdate</label>
              <input
                name="birthdate"
                type="date"
                value={formData.birthdate ? formData.birthdate.split('T')[0] : ''}
                onChange={handleChange}
              />

              <div className="profile-buttons mt-4">
                <button type="button" className="save-btn" onClick={handleSave}>Save</button>
                <button type="button" className="cancel-btn" onClick={handleCancel}>Cancel</button>
              </div>
            </form>
          ) : (
            <div>
              <p><strong>Name:</strong> {user.name}</p>
              <p><strong>Email:</strong> {user.email}</p>
              <p><strong>Phone:</strong> {user.phone}</p>
              <p><strong>Birthdate:</strong> {user.birthdate ? new Date(user.birthdate).toLocaleDateString() : 'N/A'}</p>
              <button className="edit-btn mt-4" onClick={() => setEditMode(true)}>Edit Profile</button>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
