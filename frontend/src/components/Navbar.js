'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';

export default function Navbar() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Simulated auth check — replace with real JWT/localStorage/etc.
  useEffect(() => {
    const token = localStorage.getItem('token');
    setIsLoggedIn(!!token);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsLoggedIn(false);
    // Redirect or reload if needed
  };

  return (
    <nav className="w-full px-4 py-3 bg-gray-800 text-white flex justify-between items-center">
      <h1 className="text-xl font-bold">Oral Cancer Diagnosis</h1>
      <div className="space-x-4">
        <Link href="/">Home</Link>
        {isLoggedIn ? (
          <>
            <Link href="/image-history">History</Link>
            <Link href="/profile">Profile</Link>
            <button onClick={handleLogout}>Logout</button>
          </>
        ) : (
          <>
            <Link href="/login">Login</Link>
            <Link href="/register">Register</Link>
          </>
        )}
        <Link href="/about">About</Link>
        <Link href="/contact">Contact</Link>
      </div>
    </nav>
  );
}
