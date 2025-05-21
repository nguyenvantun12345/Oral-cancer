'use client';

import { useState,useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Layout from '@/components/Layout';

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  // Check for token expiration on mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const decoded = jwt.decode(token);
        if (decoded && decoded.exp) {
          const expiration = new Date(decoded.exp * 1000);
          if (expiration < new Date()) {
            // Token expired, clear storage
            localStorage.clear();
            router.refresh();
          }
        }
      } catch (error) {
        localStorage.clear();
        router.refresh();
      }
    }
  }, [router]);

  const handleLogin = async (e) => {
    e.preventDefault();

    // Basic validation
    if (!username.trim() || !password.trim()) {
      setErrorMessage('Username and password are required');
      return;
    }

    try {
      // Show loading state
      setErrorMessage('');
      setLoading(true);
      
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: username.trim(),
          password: password,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        // Store the token and user data
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        localStorage.setItem('user_id', data.user_id);
        localStorage.setItem('username', data.username);
        localStorage.setItem('role', data.role);
        // Optionally set up token refresh interval if desired
        // const expiration = data.expires_in * 1000;
        // setTimeout(() => {
        //   // You can implement a refresh here if you have a refresh endpoint
        // }, expiration - 300000); // Refresh 5 minutes before expiration
        // Redirect to the home page
        router.push('/');
      } else {
        setErrorMessage(data.detail || 'Invalid username or password');
        setLoading(false);
      }
    } catch (error) {
      setErrorMessage('Login failed. Please try again later.');
      setLoading(false);
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
