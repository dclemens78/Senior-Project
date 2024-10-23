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
});
