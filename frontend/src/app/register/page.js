'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Layout from '../../components/Layout';

const RegisterPage = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const router = useRouter();

  const handleRegister = async (e) => {
    e.preventDefault();

    // Call your backend register API here (FastAPI endpoint)
    try {
      const response = await fetch('http://your-api-url/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: username,
          email: email,
          password: password,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        // Successful registration, redirect to login
        router.push('/login');
      } else {
        setErrorMessage(data.detail || 'An error occurred');
      }
    } catch (error) {
      setErrorMessage('Registration failed. Please try again later.');
    }
  };

  return (
    <Layout>
    <div className="min-h-screen flex justify-center items-center bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-lg max-w-sm w-full">
        <h2 className="text-2xl font-semibold mb-6 text-center">Register</h2>
        {errorMessage && (
          <div className="bg-red-200 text-red-800 p-2 mb-4 rounded">{errorMessage}</div>
        )}
        <form onSubmit={handleRegister}>
          <div className="mb-4">
            <label htmlFor="username" className="block text-sm font-semibold text-gray-700">Username</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded mt-2"
              required
            />
          </div>
          <div className="mb-4">
            <label htmlFor="email" className="block text-sm font-semibold text-gray-700">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded mt-2"
              required
            />
          </div>
          <div className="mb-4">
            <label htmlFor="password" className="block text-sm font-semibold text-gray-700">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded mt-2"
              required
            />
          </div>
          <button type="submit" className="w-full py-2 bg-blue-600 text-white rounded mt-4">
            Register
          </button>
        </form>
        <div className="mt-4 text-center">
          <span className="text-sm text-gray-600">Already have an account? </span>
          <a href="/login" className="text-blue-600">Login</a>
        </div>
      </div>
    </div>
    </Layout>
  );
};

export default RegisterPage;
