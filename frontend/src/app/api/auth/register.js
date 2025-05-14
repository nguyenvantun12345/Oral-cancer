// pages/api/auth/register.js

import axios from 'axios';

export default async function handler(req, res) {
  if (req.method === 'POST') {
    try {
      const response = await axios.post('http://localhost:8000/auth/register', req.body);
      res.status(200).json(response.data);
    } catch (err) {
      res.status(400).json({ error: 'Registration failed' });
    }
  } else {
    res.status(405).json({ error: 'Method Not Allowed' });
  }
}
