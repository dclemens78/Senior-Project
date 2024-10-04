document.getElementById('uploadbutton').addEventListener('click', function() {
    let form = document.getElementById('uploadForm');
    let formData = new FormData(form);

    // Make the POST request to Flask using fetch API
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())  // Parse the JSON response
    .then(data => {
        // Display the uploaded image in the preview section
        let imgFile = form.filepath.files[0];
        document.getElementById('imagePreview').innerHTML = `<img src="${URL.createObjectURL(imgFile)}" alt="Uploaded Image" width="200" />`;

        // Display the result from the AI model
        document.getElementById('resultText').innerText = data.result;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('resultText').innerText = 'An error occurred. Please try again.';
document.addEventListener('DOMContentLoaded', function() {
    // Handle image file input
    document.getElementById('drop').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            selectedFile = file;
            
            reader.onload = function(e) {
                const imagePreview = document.getElementById('imagePreview');
                const uploadContainer = document.querySelector('.upload-container');
                
                if (imagePreview) {
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
                        uploadContainer.classList.add('show');
                    } else {
                        console.error('Element with class "upload-container" not found');
                    }
                } else {
                    console.error('Element with ID "imagePreview" not found');
                }
            };
            
            reader.readAsDataURL(file);
        } else {
            console.error('No file selected or file is invalid');
        }
    });

    // Handle upload button click (placeholder)
    document.getElementById('uploadbutton').addEventListener('click', function() {
        if (selectedFile){
            const formData = new FormData();
            formData.addend('file', selectedFile);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                alert('File Uploaded successfully');
            })
            .catch(error => {
                console.error('Error uploading file', error);
                alert("Error uploading file.");
            });
        } else{
            alert("No file selected.");
        }
        // Placeholder for future form/photo submission handling
        alert('Upload button clicked');
        
>>>>>>> c50c3cc86301ab18b12473a5f30106b0c6832d4f:Website/script.js
    });
});
