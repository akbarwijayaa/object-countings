# Object Countings - Materials Detection

#### Quick Overview
Project Name : **Materials Detection - Object Countings**
<br>
Environment : **`quick_p1`**
<br>
Algorithm : **[YOLOV5]( https://github.com/ultralytics/yolov5) from Ultralytics**
<br>
Weight Type : **`cage_wheel_track_single.pt`**
<br>
Current Model : **hyps high + AdamW optimizer model**

<br>

# <div align="left"><h3>API Documentation</h3></div>

# Base URL

```bash
http://ai.quick.com
```
Seluruh jenis request menuju server AI menggunakan base URL tersebut.

<br>

# Endpoints
- ##  Get API Info
    Endpoint ini digunakan untuk mendapatkan informasi bahwa API telah aktif. Method yang digunakan adalah **`GET`**
    <br>

    **Endpoint**
    ```bash
    GET   /v1/materials_detection
    ```

    **Response**
    ```json
    QUICK MATERIALS DETECTION
    ```
    <br>

- ## Materials Detection
    Endpoint ini digunakan untuk mendeteksi dan menghitung objek dalam gambar. Method yang digunakan adalah **`POST`**
    <br>

    **Endpoint**
    ```bash
    POST   /v1/materials_detection
    ```
    <br>

    **Request Body** `(form-data)` :
    * **`image`** _(file, required)_ : file gambar yang akan dideteksi.
    * **`obj_type`** _(string, required)_ : "cage_wheel_track_single" 

    <br>

    **Example Request using CURL**
    ```bash
    curl --request POST 'http://ai.quick.com/v1/materials_detection' \
    --header 'Host: ai.quick.com' \
    --header 'Content-Type: multipart/form-data; boundary=--------------------------641642483064404002925119' \
    --form 'image=@"/path/To/yourFolder/image.jpg"' \
    --form 'obj_type="cage_wheel_track_single"'
    ```
    <br>


    **Example Response**
    ```bash

    {
        "code": 200,       # kode status HTTP
        "data": [          # hasil deteksi
            {
                "height": 786,   # tinggi bounding box objek (pixel)
                "idx": 1,        # index perhitungan
                "radius": 92.0,  # diameter titik tengah bounding box
                "width": 230,    # lebar bounding box objek (pixel)
                "xcenter": 180,  # letak absis titik x-data
                "ycenter": 454   # letak ordinat titik y-data
            },
            {
                "height": 766,
                "idx": 2,
                "radius": 164.0,
                "width": 410,
                "xcenter": 278,
                "ycenter": 453
            },
            {
                "height": 776,
                "idx": 3,
                "radius": 93.0,
                "width": 234,
                "xcenter": 640,
                "ycenter": 471
            },

            ---

        ],
        "id_counting": 874,     # ID dari hasil deteksi material
        "img_height": 3840,     # tinggi gambar dalam pxel
        "img_width": 2160,      # lebar gambar dalam pixel
        "message": "Successfully",      
        "success": true,
        "time": "2.64s"         # waktu response API 
    }
        
    ```

<br>

- ## Update Data 
    Endpoint ini digunakan untuk menginput data real dari hasil perhitungan manual oleh user. Data ini digunakan untuk evaluasi projek, dengan menghitung jumlah _false positive_ dan _false negative_ dari jumlah real object. Method yang digunakan adalah **`PUT`**
    <br>

    **Endpoint**
    ```shell
    PUT   /counting/<id_counting>
    ```
    <br>

    **Params** :
    * **`quantity`**  _(required)_ : total objek real hasil perhitungan manual

    <br>

    **Example Request from Shell, using CURL**
    ```bash
    curl --request PUT http://ai.quick.com/counting/874 \    # '874' merupakan id _counting
    --header "Content-Type: application/json" \
    --data '{"quantity": 320}' 
    ```
    <br>

    **Example Response**
    ```json
    {
        'success': True   
    }
    ```
    <br>


# Error Handling
Object Countings API menggunakan standar HTTP status code sebagai indikasi sukses/gagal sebuah request.
* **200** _OK_

* **400** _bad request_

* **403** _forbidden_

* **404** _notfound_

* **405** _method not allowed_

* **408** _request timeout_

* **500** _internal server error_

* **502** _bad gateway_

* **503** _service unavailable_

* **504** _gateway timeout_