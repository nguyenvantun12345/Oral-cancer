import React, { useState, useCallback } from 'react';

// SVG Camera Icon Component
// This could also be in its own file (e.g., CameraIcon.js) if you prefer further modularity
const CameraIcon = ({ className }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
  >
    <path d="M4.5 4.5a3 3 0 00-3 3v9a3 3 0 003 3h15a3 3 0 003-3v-9a3 3 0 00-3-3h-1.5l-1.218-1.624A3 3 0 0015.532 2H8.468a3 3 0 00-2.25 1.125L5.002 4.5H4.5zM12 17a5 5 0 110-10 5 5 0 010 10z" />
    <path d="M12 15a3 3 0 110-6 3 3 0 010 6z" />
  </svg>
);

// Main Object Recognizer Component
const ObjectRecognizer = () => {
  // State for the uploaded image file
  const [uploadedImage, setUploadedImage] = useState(null);
  // State for the image preview URL
  const [imagePreview, setImagePreview] = useState('');
  // State for minimum confidence input
  const [minConfidence, setMinConfidence] = useState(80);
  // State for maximum objects input
  const [maxObjects, setMaxObjects] = useState(20);
  // State to indicate if a file is being dragged over
  const [isDragging, setIsDragging] = useState(false);

  // Handle file selection from input or drag-and-drop
  const handleFileChange = (file) => {
    if (file && file.type.startsWith('image/')) {
      setUploadedImage(file);
      // Create a preview URL for the image
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      // Basic error handling for non-image files
      console.error("Please upload an image file.");
      // Here you could set an error message to display to the user
      setUploadedImage(null);
      setImagePreview('');
    }
  };

  // Handle file drop event
  const onDrop = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      handleFileChange(event.dataTransfer.files[0]);
      event.dataTransfer.clearData();
    }
  }, []); // Empty dependency array as handleFileChange is stable

  // Handle drag over event
  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  }, []);

  // Handle drag leave event
  const onDragLeave = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  }, []);

  // Handle click on the drop zone to trigger file input
  const onDropZoneClick = () => {
    // Programmatically click the hidden file input
    document.getElementById('fileInput').click();
  };

  return (
    // The main container for the object recognizer section
    // Note: The 'min-h-screen' and 'bg-gray-100' might be better placed in App.js
    // or a layout component if this ObjectRecognizer is part of a larger page.
    // For this example, we'll keep it here to match the previous structure.
    <div className="flex flex-col items-center justify-center p-4 font-sans">
      {/* Main content container */}
      <div className="w-full max-w-3xl p-6 md:p-10">
        {/* Section Title */}
        <h1 className="text-4xl font-bold text-center text-gray-800 mb-4">Objects</h1>
        {/* Section Subtitle */}
        <p className="text-center text-gray-600 mb-2">
          Identify objects in your image by using our Object Recognizer.
        </p>
        <p className="text-center text-gray-600 mb-8">
          Vary the detection confidence and the number of objects that you want to detect below.
        </p>

        {/* Image Upload Area */}
        <div
          className={`relative flex flex-col items-center justify-center w-full h-64 md:h-80 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-200 ease-in-out
            ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
            ${imagePreview ? 'border-green-500' : ''}
          `}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={onDropZoneClick} // Allow click to upload
        >
          {/* Hidden file input */}
          <input
            type="file"
            id="fileInput"
            className="hidden"
            accept="image/*"
            onChange={(e) => handleFileChange(e.target.files[0])}
          />
          {imagePreview ? (
            // Display image preview if an image is uploaded
            <img
              src={imagePreview}
              alt="Uploaded preview"
              className="object-contain w-full h-full rounded-lg"
            />
          ) : (
            // Display upload prompt if no image is uploaded
            <div className="text-center">
              <CameraIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 font-semibold">Drop an image here</p>
              <p className="text-gray-400 text-sm">or</p>
              <p className="text-blue-500 hover:text-blue-600 font-medium">click to browse</p>
            </div>
          )}
        </div>

        {/* Input fields for confidence and max objects */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Minimum Confidence Input */}
          <div className="flex flex-col">
            <label htmlFor="minConfidence" className="mb-1 text-sm font-medium text-gray-700">
              Minimum confidence:
            </label>
            <div className="flex items-center">
              <input
                type="number"
                id="minConfidence"
                value={minConfidence}
                onChange={(e) => setMinConfidence(parseInt(e.target.value, 10) || 0)}
                className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow"
                min="0"
                max="100"
              />
              <span className="ml-2 text-gray-600">%</span>
            </div>
          </div>

          {/* Maximum Objects Input */}
          <div className="flex flex-col">
            <label htmlFor="maxObjects" className="mb-1 text-sm font-medium text-gray-700">
              Maximum objects:
            </label>
            <input
              type="number"
              id="maxObjects"
              value={maxObjects}
              onChange={(e) => setMaxObjects(parseInt(e.target.value, 10) || 0)}
              className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow"
              min="1"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectRecognizer; // Export the component to be used in other files
