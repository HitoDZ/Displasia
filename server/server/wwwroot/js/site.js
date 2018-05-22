// Write your JavaScript code.
function uploadFile() {
    //  var blobFile = $('#filechooser').files[0];
    var blobFile = document.getElementById("filechooser").files[0];
    var formData = new FormData();
    formData.append("fileToUpload", blobFile);
    //debugger;

    $.ajax({
        url: "http://localhost:5000/Home/Rest",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            console.log(response); // .. do something
        },
        error: function(jqXHR, textStatus, errorMessage) {
            console.log(errorMessage); // Optional
        }
    });
}