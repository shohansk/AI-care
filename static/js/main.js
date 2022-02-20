$(document).ready(function () {

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });


    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        
        $(this).hide();
        $('.loader').show();

        $.fn.multiline = function(text){
            this.text(text);
            this.html(this.html().replace(/\n/g,'<br/>'));
            return this;
        }

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
  
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').multiline(' result of prediction : \n' + data).css({fontSize: '18px'});
                console.log('Success!');
            },
        });
    });

});