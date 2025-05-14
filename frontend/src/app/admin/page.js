// app/admin/page.js
'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Layout from '../../components/Layout';

export default function AdminPage() {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem('token');
    const role = localStorage.getItem('role'); // Assuming the role is stored in localStorage
    if (!token || role !== 'admin') {
      router.push('/login'); // Redirect to login if not authenticated or not an admin
    } else {
      fetchPatients(token);
    }
  }, []);

  const fetchPatients = async (token) => {
    try {
      const response = await fetch('/api/admin/patients', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) {
        throw new Error('Failed to fetch patients');
      }
      const data = await response.json();
      setPatients(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePatient = async (patientId) => {
    const token = localStorage.getItem('token');
    try {
      const response = await fetch(`/api/admin/patients/${patientId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) {
        throw new Error('Failed to delete patient');
      }
      setPatients(patients.filter((patient) => patient.id !== patientId));  // Remove from state
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <Layout>
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Admin - Manage Patients</h1>
      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">{error}</p>}
      {!loading && patients.length === 0 && <p>No patients found.</p>}
      <table className="min-w-full border-collapse">
        <thead>
          <tr>
            <th className="px-4 py-2 border">Name</th>
            <th className="px-4 py-2 border">Email</th>
            <th className="px-4 py-2 border">Phone</th>
            <th className="px-4 py-2 border">Actions</th>
          </tr>
        </thead>
        <tbody>
          {patients.map((patient) => (
            <tr key={patient.id}>
              <td className="px-4 py-2 border">{patient.name}</td>
              <td className="px-4 py-2 border">{patient.email}</td>
              <td className="px-4 py-2 border">{patient.phone}</td>
              <td className="px-4 py-2 border">
                <button onClick={() => handleDeletePatient(patient.id)} className="text-red-500">Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    </Layout>
  );
}
