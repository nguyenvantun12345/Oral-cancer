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
      const response = await fetch('http://localhost:8000/admin/patients/bulk-read?limit=100&skip=0', {
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
      const response = await fetch('http://localhost:8000/admin/patients/bulk-delete', {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_ids: [patientId],
          wait_seconds: 0,
          max_delete: 15,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to delete patient');
      }

      const result = await response.json();

      // Remove the deleted patient from the local state
      setPatients(patients.filter((patient) => patient.user_id !== patientId));
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
              <tr key={patient.user_id}>
                <td className="px-4 py-2 border">{patient.name}</td>
                <td className="px-4 py-2 border">{patient.email}</td>
                <td className="px-4 py-2 border">{patient.phone}</td>
                <td className="px-4 py-2 border">
                  <button
                    onClick={() => handleDeletePatient(patient.user_id)}
                    className="text-red-500"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Layout>
  );
}
