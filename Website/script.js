<<<<<<< HEAD
document.getElementById("uploadbutton").addEventListener("click", async () => {
    const fileInput = document.getElementById("drop");
    const formData = new FormData();

    if (fileInput.files.length > 0) {
        formData.append("file", fileInput.files[0]);  // Append the selected file to FormData

        try {
            // Send a POST request to the FastAPI backend at the '/predict' endpoint
            const response = await fetch("http://127.0.0.1:8001/predict", {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                document.getElementById("resultText").innerText = data.prediction;  // Display the prediction
            } else {
                document.getElementById("resultText").innerText = "Error occurred during prediction.";
            }
        } catch (error) {
            console.error("Error:", error);
            document.getElementById("resultText").innerText = "Failed to connect to the server.";
        }
    } else {
        alert("Please select a file first.");  // Notify user if no file is selected
    }
=======
document.addEventListener('DOMContentLoaded', function() {
    let selectedFile; // Initialize selectedFile here

    // Handle image file input
    document.getElementById('drop').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            selectedFile = file;

            reader.onload = function(e) {
                const imagePreview = document.getElementById('imagePreview');
                const uploadContainer = document.querySelector('.upload-container');

                // Create an img element
                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = 'Uploaded Image';
                img.style.maxWidth = '720px'; // Make sure the image fits within the container
                img.style.height = 'auto'; // Maintain aspect ratio

                // Clear previous content and add new image
                imagePreview.innerHTML = '';
                imagePreview.appendChild(img);

                // Show the upload container
                if (uploadContainer) {
                    uploadContainer.style.display = 'block'; // Show the container
                } else {
                    console.error('Element with class "upload-container" not found');
                }
            };

            reader.readAsDataURL(file);
        } else {
            console.error('No file selected or file is invalid');
        }
    });

    // Handle upload button click
    document.getElementById('uploadbutton').addEventListener('click', function() {
        if (selectedFile) {
            const formData = new FormData();
            formData.append('file', selectedFile);

            fetch('http://127.0.0.1:8001/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                document.getElementById('resultText').innerText = data.prediction || 'No prediction available';
                alert('File Uploaded successfully');
                // Optionally handle the response data here
            })
            .catch(error => {
                console.error('Error uploading file', error);
                alert("Error uploading file.");
            });
        } else {
            alert("No file selected.");
        }

        // Placeholder for future form/photo submission handling
        alert('Upload button clicked');
    });
>>>>>>> b9d97d6b0532cae77c3683898fb4d7136c33e1b3
});



