'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Layout from '@/components/Layout';

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const router = useRouter();

  const handleLogin = async (e) => {
    e.preventDefault();

    // Call your backend login API here (FastAPI endpoint)
    try {
      const response = await fetch('http://your-api-url/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: username,
          password: password,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        // Store the access token in localStorage (for example)
        localStorage.setItem('token', data.access_token);
        // Redirect to the profile page or home after successful login
        router.push('/profile');
      } else {
        setErrorMessage(data.detail || 'An error occurred');
      }
    } catch (error) {
      setErrorMessage('Login failed. Please try again later.');
    }
  };

  return (
    <Layout>
    <div className="min-h-screen flex justify-center items-center bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-lg max-w-sm w-full">
        <h2 className="text-2xl font-semibold mb-6 text-center">Login</h2>
        {errorMessage && (
          <div className="bg-red-200 text-red-800 p-2 mb-4 rounded">{errorMessage}</div>
        )}
        <form onSubmit={handleLogin}>
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
            Login
          </button>
        </form>
        <div className="mt-4 text-center">
          <span className="text-sm text-gray-600">Don't have an account? </span>
          <a href="/register" className="text-blue-600">Register</a>
        </div>
      </div>
    </div>
    </Layout>
  );
};

export default LoginPage;
