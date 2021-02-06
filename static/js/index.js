
// Loading scene
function load_page() {
  $('#loading').css('display','block');
  $('#loading').css('overflow','hidden');
}

$('#upload').click(function(input) {
  if (document.getElementById("input-images").files.length != 0){
    load_page();
  }
});

$('#stitching').click(function() {
  load_page();
});

// End loading -----------------------------------

// Check dragzone have files

function readURL(input) {
  if (input.files && input.files[0]) {
    var name = "";
    var gotFiles = document.getElementById("input-images").files;

    for (let i = 0; i < gotFiles.length; i++) {
      name = name + gotFiles[i]['name'] + " ";
    }

    $('#dropzone').text(name);
    $('.drop-container').css('background-color','#32e0c4');
  }
}

$('#raw-btn').click(function() {
  $('#raw').css('display','block');
  $('#panorama').css('display','none');
  $('#raw-btn').css('background-color','#00adb5');
  $('#crop-btn').css('background-color','#222831');
});

$('#crop-btn').click(function() {
  $('#panorama').css('display','block');
  $('#raw').css('display','none');
  $('#crop-btn').css('background-color','#00adb5');
  $('#raw-btn').css('background-color','#222831');
});

$(".stitch").click(function() {
    $('html,body').animate({
        scrollTop: $("#main").offset().top},
        'slow');
});

// -------------------------------------------------

// // Get the modal
// var modal = $(".modal");
//
// // Get the <span> element that closes the modal
// var span_reg = document.getElementsByClassName("close")[0];
//
// // When the user clicks the button, open the modal
// function openModal() {
//   modal.fadeIn();
// }
//
// // When the user clicks on <span> (x), close the modal
// span_reg.onclick = function() {
//   modal.fadeOut();
// }
//
// // When the user clicks anywhere outside of the modal, close it
// window.onclick = function(event) {
//   if (event.target == modal) {
//     modal.fadeOut();
//   }
// }
// End Modal -----------------------------
