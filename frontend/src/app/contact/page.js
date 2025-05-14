import Layout from '../../components/Layout';

// Example: app/contact/page.js

export default function ContactPage() {
  return (
    <Layout>
        <div className="page-container">
        <div className="card">
            <h1 className="page-title">Contact Us</h1>
            <form className="space-y-4">
            <div>
                <label className="input-label">Name</label>
                <input type="text" className="text-input" placeholder="Your name" />
            </div>
            <div>
                <label className="input-label">Email</label>
                <input type="email" className="text-input" placeholder="you@example.com" />
            </div>
            <div>
                <label className="input-label">Message</label>
                <textarea rows="4" className="text-input" placeholder="Your message..." />
            </div>
            <button type="submit" className="submit-btn">Send Message</button>
            </form>
        </div>
        </div>
    </Layout>
  );
}
