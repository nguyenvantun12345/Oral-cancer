import React, { useState, useCallback } from 'react';

// IMPORTANT: Replace with your actual Cloudinary details
// For security, ideally handle uploads on your backend with signed uploads
// rather than exposing these directly in frontend code.
const CLOUD_NAME = 'dd5krn2yq'
const UPLOAD_PRESET = 'unsigned'; // e.g., 'unsigned_upload_preset'

// SVG Camera Icon Component (can be in its own file)
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

// Main Oral Cancer Predictor Component
const OralCancerPredictor = () => {
  // State for the uploaded image file
  const [uploadedImage, setUploadedImage] = useState(null);
  // State for the image preview URL
  const [imagePreview, setImagePreview] = useState('');
  // State to indicate if a file is being dragged over
  const [isDragging, setIsDragging] = useState(false);
  // State for the prediction result
  const [prediction, setPrediction] = useState(null); // e.g., 75 (for 75%)
  // State for loading indicator during "diagnosis"
  const [isLoading, setIsLoading] = useState(false);
  // State for error messages
  const [errorMessage, setErrorMessage] = useState('');

  // Memoized handleFileChange
  const handleFileChange = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      setUploadedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      // Reset previous results when a new file is chosen
      setPrediction(null);
      setIsLoading(false);
      setErrorMessage('');
    } else {
      console.error("Please upload an image file.");
      setUploadedImage(null);
      setImagePreview('');
      setPrediction(null);
      setIsLoading(false);
      setErrorMessage("Please upload a valid image file (png, jpg, jpeg, gif).");
    }
  }, []); // Dependencies: setUploadedImage, setImagePreview, setPrediction, setIsLoading, setErrorMessage are stable

  // Handle file drop event
  const onDrop = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      handleFileChange(event.dataTransfer.files[0]);
      event.dataTransfer.clearData();
    }
  }, [handleFileChange]);

  // Handle drag over event
  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!isDragging) setIsDragging(true);
  }, [isDragging]);

  // Handle drag leave event
  const onDragLeave = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    // Check if the drag is leaving to outside the component or to a child element
    // This helps prevent flickering when dragging over child elements.
    if (event.relatedTarget && event.currentTarget.contains(event.relatedTarget)) {
      return;
    }
    setIsDragging(false);
  }, []);

  // Handle click on the drop zone to trigger file input
  const onDropZoneClick = () => {
    // Only trigger if not already loading or prediction is not shown
    if (!isLoading && prediction === null) {
        document.getElementById('fileInput').click();
    }
  };

  // Handle the "diagnosis" submission
  const handleSubmitDiagnosis = async () => {
    if (!uploadedImage) {
      setErrorMessage("Please upload an image first.");
      return;
    }

    if (!CLOUD_NAME || !UPLOAD_PRESET) {
        setErrorMessage("Cloudinary configuration missing. Cannot upload.");
        return;
    }

    // --- Retrieve JWT from localStorage ---
    const jwtToken = localStorage.getItem('token');

    if (!jwtToken) {
        setErrorMessage("Authentication required. Please log in.");
        // Optional: Here you might want to redirect the user to the login page
        console.error("JWT token not found in localStorage.");
        setIsLoading(false); // Ensure loading is off
        return; // Stop the function execution
    }
    // --- End Retrieve JWT ---


    setIsLoading(true);
    setErrorMessage('');
    setPrediction(null);

    const CLOUDINARY_URL = `https://api.cloudinary.com/v1_1/${CLOUD_NAME}/image/upload`;

    try {
      // Step 1: Upload image to Cloudinary
      const formData = new FormData();
      formData.append('file', uploadedImage);
      formData.append('upload_preset', UPLOAD_PRESET);

      console.log("Uploading image to Cloudinary...");
      const cloudinaryResponse = await fetch(CLOUDINARY_URL, {
        method: 'POST',
        body: formData,
      });

      if (!cloudinaryResponse.ok) {
        const errorText = await cloudinaryResponse.text();
        throw new Error(`Cloudinary upload failed: ${cloudinaryResponse.status} ${cloudinaryResponse.statusText} - ${errorText}`);
      }

      const cloudinaryData = await cloudinaryResponse.json();
      const imageUrl = cloudinaryData.secure_url; // Use secure_url for HTTPS

      console.log("Image uploaded to Cloudinary:", imageUrl);

      // Step 2: Send image URL to your backend API with JWT
      console.log("Sending image URL to backend API...");
      const apiResponse = await fetch('http://localhost:8000/patients/me/history', { // Make sure this path is correct relative to your frontend base URL
        method: 'POST',
        headers: {
          // --- Add the Authorization header with the token ---
          'Authorization': `Bearer ${jwtToken}`,
          // --- End Add Authorization header ---
        },
        body: JSON.stringify({ image: imageUrl }),
      });

      // --- Handle 401/403 Unauthorized responses ---
      if (apiResponse.status === 401 || apiResponse.status === 403) {
           // This means the token was missing, invalid, or expired,
           // or the user (based on the token) doesn't have permission.
           let errorDetail = 'Authentication failed. Please log in again.';
           try {
              const errorJson = await apiResponse.json();
              errorDetail = errorJson.detail || errorDetail;
           } catch (e) {
              // Ignore JSON parsing error if response is not JSON
           }
           setErrorMessage(errorDetail);
           // IMPORTANT: Trigger your application's global logout process here
           // e.g., remove token from localStorage, redirect to login page
           console.error("Authentication Error:", apiResponse.status, errorDetail);
           setIsLoading(false); // Ensure loading is off
           return; // Stop the function execution
      }
      // --- End Handle 401/403 Unauthorized responses ---


      if (!apiResponse.ok) {
         // Handle other non-401/403 API errors
         let errorDetail = 'Backend diagnosis request failed.';
         try {
            const errorJson = await apiResponse.json();
            errorDetail = errorJson.detail || JSON.stringify(errorJson);
         } catch (e) {
            errorDetail = apiResponse.statusText || errorDetail;
         }
         throw new Error(`API Error: ${apiResponse.status} - ${errorDetail}`);
      }

      const apiData = await apiResponse.json();

      // Assuming your backend returns an object with 'diagnosis_score'
      if (apiData && typeof apiData.diagnosis_score === 'number') {
        setPrediction(apiData.diagnosis_score);
        console.log("Diagnosis score received:", apiData.diagnosis_score);
      } else {
        // Backend returned OK status but unexpected data
        throw new Error("Invalid response format from API. Expected diagnosis_score.");
      }

    } catch (err) {
      console.error("Diagnosis process error:", err);
       // Only set the error message if it wasn't already set by 401/403 handling
      if (!errorMessage) {
        setErrorMessage(`Error: ${err.message || "An unknown error occurred."}`);
      }
      setPrediction(null); // Ensure prediction is null on error
    } finally {
      setIsLoading(false); // Ensure loading is off
    }
  };

  return (
    <div className="flex flex-col items-center justify-center p-4 font-sans min-h-screen bg-gray-50">
      <div className="w-full max-w-2xl p-6 md:p-10 bg-white shadow-xl rounded-lg">
        <h1 className="text-3xl md:text-4xl font-bold text-center text-gray-800 mb-3">
          Oral Lesion Image Analysis
        </h1>
        <p className="text-center text-gray-600 mb-2">
          Upload an image of an oral lesion to get a predicted likelihood percentage.
        </p>
        <p className="text-center text-red-600 text-sm mb-8 px-4">
          <span className="font-semibold">Disclaimer:</span> This tool uses a machine learning model for simulated prediction and is <span className="font-semibold">not a substitute for professional medical advice, diagnosis, or treatment.</span> Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        </p>

        {/* Image Upload Area */}
        <div
          className={`relative flex flex-col items-center justify-center w-full h-64 md:h-80 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-200 ease-in-out
            ${isDragging ? 'border-blue-600 bg-blue-50' : 'border-gray-400 hover:border-gray-500'}
            ${imagePreview && !isDragging ? 'border-green-500' : ''}
          `}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={onDropZoneClick}
        >
          <input
            type="file"
            id="fileInput"
            className="hidden"
            accept="image/*"
            onChange={(e) => handleFileChange(e.target.files[0])}
             // Disable input if loading or prediction is shown or error exists to prevent multiple uploads/interactions
            disabled={isLoading || prediction !== null || errorMessage !== ''}
          />
          {imagePreview ? (
            <img
              src={imagePreview}
              alt="Uploaded preview"
              className="object-contain w-full h-full rounded-lg p-1"
            />
          ) : (
            <div className="text-center p-5">
              <CameraIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 font-semibold">Drop an image here</p>
              <p className="text-gray-500 text-sm">or</p>
              <p className="text-blue-600 hover:text-blue-700 font-medium">click to browse files</p>
            </div>
          )}
        </div>

         {/* Display error messages (consolidated) */}
        {errorMessage && (
            <div className="mt-4 text-center p-3 bg-red-100 text-red-700 border border-red-200 rounded-lg">
              <p>{errorMessage}</p>
            </div>
        )}


        {/* Action Button: Analyze Image */}
        {uploadedImage && !isLoading && prediction === null && !errorMessage && ( // Show button only after upload, before loading/result/error
          <div className="mt-8 text-center">
            <button
              onClick={handleSubmitDiagnosis}
              className="px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-150 ease-in-out disabled:opacity-50"
              disabled={isLoading}
            >
              Analyze Image
            </button>
          </div>
        )}

        {/* Loading State during "diagnosis" */}
        {isLoading && (
          <div className="mt-8 text-center">
            <div className="inline-flex items-center">
              <svg className="animate-spin -ml-1 mr-3 h-6 w-6 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-lg text-gray-700">Analyzing image, please wait...</p>
            </div>
          </div>
        )}


        {/* Prediction Result Display */}
        {prediction !== null && !isLoading && !errorMessage && (
          <div className="mt-8 text-center p-6 bg-green-50 border-2 border-green-400 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold text-gray-800 mb-2">Prediction Result</h2>
            <p className="text-5xl font-bold text-green-600 my-3">{prediction}%</p>
            <p className="text-md text-gray-700">Predicted Likelihood of Malignancy</p>
            <p className="text-xs text-gray-500 mt-4 px-2">
              This result is based on a simulated analysis. Always consult with a qualified healthcare professional for any medical concerns or for an accurate diagnosis.
            </p>
            {/* Optional: Add a "Analyze another image" button */}
             <button
                onClick={() => {
                    setUploadedImage(null);
                    setImagePreview('');
                    setPrediction(null);
                    setErrorMessage(''); // Clear error when starting over
                }}
                className="mt-4 px-6 py-2 bg-gray-200 text-gray-800 font-semibold rounded-lg hover:bg-gray-300 transition duration-150 ease-in-out"
             >
                Upload Another Image
             </button>
          </div>
        )}

      </div>
    </div>
  );
};

export default OralCancerPredictor;