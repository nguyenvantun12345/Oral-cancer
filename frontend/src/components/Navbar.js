import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="w-full px-4 py-3 bg-gray-800 text-white flex justify-between items-center">
      <h1 className="text-xl font-bold">Oral Cancer Diagnosis</h1>
      <div className="space-x-4">
        <Link href="/">Home</Link>
        <Link href="/image-history">History</Link>
        <Link href="/profile">Profile</Link>
        <Link href="/about">About</Link>
        <Link href="/contact">Contact</Link>
      </div>
    </nav>
  );
}
