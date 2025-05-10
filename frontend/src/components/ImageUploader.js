import { useState } from 'react';
import axios from 'axios';

const ImageUploader = () => {
  const [imageUrl, setImageUrl] = useState(null);
  const [imageFile, setImageFile] = useState(null);

  // Handle file selection
  const handleFileChange = (event) => {
    setImageFile(event.target.files[0]);
  };

  // Upload to Imgur
  const handleImageUpload = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append('image', imageFile);

    try {
      const response = await axios.post('https://api.imgur.com/3/image', formData, {
        headers: {
          Authorization: `Client-ID 6352b06a8a2c101`, // Replace with your Imgur Client ID
        },
      });

      const imageUrl = response.data.data.link;
      setImageUrl(imageUrl); // Store the URL in state or send to backend
      console.log('Image uploaded successfully:', imageUrl);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleImageUpload}>Upload Image</button>
      {imageUrl && <img src={imageUrl} alt="Uploaded Image" />}
    </div>
  );
};

export default ImageUploader;
