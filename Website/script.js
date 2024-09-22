document.addEventListener('DOMContentLoaded', function() {
    // Handle image file input
    document.getElementById('drop').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            
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
        // Placeholder for future form/photo submission handling
        alert('Upload button clicked');
        
    });
});