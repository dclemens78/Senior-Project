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
                img.style.maxWidth = '900px'; // Make sure the image fits within the container
                img.style.maxHeight = '900px'; // Maintain aspect ratio

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

    });
});