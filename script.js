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
    });
});
