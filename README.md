# Project Summary
Project Name : **Materials Detection - Object Countings**
<br>
Environment : **`quick_p1 --> include library`**
<br>
Algorithm : **[YOLOV5]( https://github.com/ultralytics/yolov5) from Ultralytics**
<br>
Current Model : **hyps high + AdamW optimizer model**
<br>
Folder for Development : /home/serverai/Project/Quick_deeplearning/yolov5_dev_training/
<br>
Folder Trial and Error Model : /home/serverai/Project/Quick_deeplearning/yolov5_dev_training/final_model/

<br>

# Quickstart
Clone repository ini, kemudian install seluruh dependensi yang terdapat dalam `requirements.txt`.
```bash
$ git clone http://gitlab.quick.com/artificial-intelligence/object-countings.git     #clone
$ cd object-countings
$ pip install -r requirements.txt       #install
```
<br>

# Dataset
Dataset yang kami gunakan adalah dataset custom, sesuai dengan dengan model yang diharapkan. Sejauh ini, kami menggunakan website `app.roboflow.com` untuk mengolah custom dataset.
<img title="Tampilan website roboflow" src="/data/images/doc-roboflow.png">
Kumpulan dataset custom kami simpan di `/home/serverai/Project/Quick_deeplearning/yolov5_dev_training/dataset` (Jupyter notebook).

Untuk informasi lebih lanjut, silahkan kunjungi [disini.](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)

# Training and Testing
Source code untuk menjalankan proses training, validating, dan detection dapat dilihat pada `/notebook/2_Test_Custom_Yolo_V5_rev2.ipynb`

**Training Script**
```python
!python train.py --img 640 --batch-size 16 --epochs 300 --data '/yolov5/data/data.yaml' --weights 'yolov5s.pt'
```
**Testing Script**
```python
!python detect.py --weights '/path/of/weight.pt' --img 640 --conf 0.5 --source '/path/of/image/test' --hide-labels --line-thickness 2 
```
**Validation Script**
```python
!python val.py --img 640 --data '/yolov5/data/data.yaml' --weights '/yolov5/runs/train/exp/weights/best.pt' 
```
<br>

### Logging with Comet ML
Comet ML merupakan salah satu platform yang digunakan sebagai pegelola dan analisa terhadap eksperimen machine learning. Comet ML dapat memberikan insight terkait performa model dalam eksperimen machine learning melalui tracking dan visualisasi hasil eksperimen. Python notebook untuk logging training terdapat dalam `/notebook/YOLOv5_with_CometML.ipynb`

**Using Comet ML**

```python
%pip install comet_ml --quiet   # install Comet ML

import comet_ml
comet_ml.init(project_name='object-countings')  # integrasikan session dengan akun Comet ML
```
_Lihat dokumentasi asli YOLOv5 dengan CometML [disini](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#comet-logging-and-visualization-new)._

<br>

# Model Optimization
### Hyperparameter Tuning
Hyperparameter, dalam machine learning, merupakan parameter yang nilainya tidak ditentukan secara langsung oleh model. Untuk itu perlu dilakukan penyetelan (tuning) sehingga mencapai kinerja model yang optimal. YOLOv5 menyediakan beberapa opsi pengaturan hyperparameter yang terdapat dalam folder `/data/hyps`.
<br>

```yaml
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Hyperparameters for low-augmentation COCO training from scratch
# python train.py --batch 64 --cfg yolov5n6.yaml --weights '' --data coco.yaml --img 640 --epochs 300 --linear
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.01  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
```
_Konfigurasi parameter default YOLOv5, lihat [dokumentasi asli Ultralytics](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution/)_

<br>

### Add Model (Development)
Untuk penambahan model baru,
1. Masuk ke <a href = /home/serverai/Project/Quick_deeplearning/yolov5_dev_training/final_model/>folder trial dan error model </a>
    ```
    └── final_model
        ├── self_training
        └── vastai
    ```
2. _Paste_-kan file `best.pt` ke dalam folder tersebut. Subfolder `self-training` untuk model yang di training **selain** di cloud.
3. Ubah nama file `best.pt` dengan nama unik sesuai model dan dataset yang digunakan saat training.

<br>

### Change Current Model as Weight (Development)
Model terbaik yang terdapat dalam folder `final_model` diaplikasikan sebagai _weight_ untuk deteksi, dan dapat diubah sesuai kebutuhan.
1. Copy model dari folder `final_model`, yang akan dijadikan sebagai weight. Paste ke dalam repo ini, dalam folder `/weight_type`.
2. Replace file `cage_wheel_track_single.pt` yang lama, dan rename model terbaru dengan nama yang sama di dalam folder (jangan lupa backup terlebih dahulu).

<br>

### Materials Detection API
Projek ini menggunakan API yang berjalan di atas host http://ai.quick.com, dengan 3 endpoints utama sebagai berikut:
* Get API Info 
  ```bash 
  GET   /v1/materials_detection
  ```
* Material Detection
  ```bash 
  POST   /v1/materials_detection
  ```
* Update Data on Database
  ```bash 
  PUT   /counting/<id_counting>
  ```

Lihat **[`README.md`](http://gitlab.quick.com/artificial-intelligence/object-countings/-/blob/development/README.md?ref_type=heads)** untuk dokumentasi lengkap mengenai penggunaan API Projek Object Countings.

<br>

# Source Code Explanation
## > API Program 
**File Location : `app.py`** <br> Terdapat dua fungsi yang menjadi inti proses dari API, yaitu `predict()`, dan `update_data()`
- #### `predict()` function
    Fungsi predict() akan dipanggil ketika terjadi HTTP request ke URL dengan metode 'POST' atau 'GET'
    ```python
    @app.route(DETECTION_URL, methods=["POST", "GET"])
    def predict():

        ...

    ```
    <br>

    Baris kode di berikut menjalankan fungsi `detect()` yang diambil dari modul `prediction.py`.
    ```python
    ...
        # mengambil file gambar dan sting dari request api "image" dan "obj_type"
        if request.method != "POST":
            return "QUICK MATERIALS DETECTION"
        image_file = request.files.getlist("image")
        for img in image_file:
            image = img.read()
            with open(temp_file, 'wb')  as outfile:
                outfile.write(image)
            if request.form.get('obj_type'):
                obj_type = request.form.get('obj_type')
                if os.path.exists(join(weight_path, obj_type+ '.pt')):
                    weight_path = join(weight_path, obj_type+ '.pt')
            
            # fungsi detect(), dari modul prediction.py, diaplikasikan untuk tiap gambar yang berasal dari request "image"
            # hasil deteksi disimpan ke dalam JSON
            result = detect(source=temp_file, weights=weight_path)
            result.to_json(temp_json, orient='records')
            f = open(temp_json)
            data_json = json.load(f)
            
            # menyimpan gambar asli dari request
            saveImgReal(img_temp=temp_file, labels_temp=temp_json, bankimg_path = saveImgReal_path)
            img_path = save_img(temp_file, data_json, bankimg_path)

    ...
    ```
    <br>

    Baris kode berikut digunakan untuk menyimpan hasil deteksi ke dalam database, yang akan digunakan sebagai bahan evaluasi performa model object detection. 

    ```python
        ...

        # mendapatkan alamat IP user yang melakukan request API
        ip_addr = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            # menambahkan data ke dalam tabel 'analytics_services_api'
            append_data(
                table_name = 'analytics_services_api',
                model = obj_type,
                id_api = 3,
                ip_address = ip_addr,
                request_date = datetime.now(),
                url_api = "http://ai.quick.com/v1/materials_detection",
                response = data,
                response_time = round((time.time() - start_time) * 100 )
            )
            data['data'] = data_json
            # menambahkan data ke dalam tabel 'analytics_counting_object'
            last_id = append_data(
                table_name = 'analytics_counting_object',
                model = obj_type,
                ip_address = ip_addr,
                request_date = datetime.now(),
                pred_transact = len(data_json),
                response_time = round((time.time() - start_time) * 100 ),
                img_path = img_path
            )
            # menambahkan kunci 'id_counting' ke dalam 'data[]', diambil dari value 'last_id'
            data['id_counting'] = last_id
            
            
        ...


    ```

<br>

- #### `update_data()` function
    Fungsi update_data() digunakan untuk memperbarui data dalam database dan berjalan di bawah metode request HTTP 'PUT'.
    ```python
    ...

    @app.route('/counting/<id_counting>', methods=['PUT'])
    def update_data(id_counting):
        # mengambil nilai 'quantity' dan melakukan koneksi ke database untuk pembaruan data
        quantity = request.json.get('quantity')
        conn, cursor = connect_db()
        query = f'UPDATE mb.analytics_counting_object SET real_transact = {quantity} WHERE id_counting = {id_counting};'
        cursor.execute(query)
        conn.commit()
        cursor.close()
        conn.close()
        result = {
            'success': True
        }
        
        # mengembalikan respon berupa bool (success : true/false) 
        return result

    ...
    ```
    Fungsi ini bertanggung jawab untuk memperbarui nilai kolom `real_transact` dari entri dalam tabel `analytics_counting_object` berdasarkan `id_counting` yang diberikan dalam permintaan 'PUT' yang masuk.

<br>

## > Prediction Module
**File Location : `prediction.py`** <br> Terdapat tiga fungsi dalam modul ini, yaitu `sorting_index()`, `detect()`, dan `get_img_size()`
- #### `sorting_index()` function
    Fungsi `sorting_index()` digunakan untuk melakukan pengurutan index terkecil dari hasil deteksi yang dimuat dalam list `data`, di mana index tersebut diperoleh dari nilai koordinat x dan y objek-objek yang berhasil dideteksi oleh model. Fungsi ini mengembalikan list baru berisi data yang sudah urut, disimpan dalam list `new_list`.

    ```python 
    ...
    # data = [x,y,w,h]
    def sorting_index(data):
        df_duplicate = [i for i in data]
        list_idx = []
        new_list = []
        while len(df_duplicate) > 0:
            #parse data
            min_xy = [x+y for x,y,w,h in data]
            x_data = [x for x,y,w,h in data]
            y_data = [y for  x,y,w,h in data]
            h_data = [h for  x,y,w,h in data]
            #cek data di min_xy sudah ada di list_idx atau belum
            list_val = [(idx, val) for idx, val in enumerate(min_xy) if idx not in list_idx]
            if len(list_val) >0:
                #Mencari nilai terkecil dari list_val
                min_idx_val = min([val for idx, val in list_val])
                start_row = int([idx for idx,val in list_val if val==min_idx_val][0])
                list_idx.append(start_row)
                new_list.append(data[start_row])
                del df_duplicate[0]

                y_start = y_data[start_row] - int((h_data[start_row])/2)
                y_range = y_data[start_row] + int((h_data[start_row])/2)

                in_range = [(idx, val) for idx,val in enumerate(y_data) if val in range(y_start, y_range) if idx != start_row]
                for i in range(0, len(in_range)):
                    min_val = [(x_data[idx], y_data[idx]) for idx, val in in_range if idx not in list_idx]
                    if len(min_val) >0:
                        min_val = min(min_val)
                        next_column = int([idx for idx, val in in_range if (x_data[idx], y_data[idx]) == min_val][0])
                        list_idx.append(next_column)
                        new_list.append(data[next_column])
                        del df_duplicate[0] 
                    else:
                        pass     
            else:
                pass
        return new_list
    ...
    ```
    <br>

- #### `detect()` function
    Fungsi `detect()` merupakan fungsi utama karena memuat inti proses deteksi objek menggunakan YOLOv5. Model diambil dari argumen `weights` yang value nya merupakan path menuju file `weights_type.pt`. Output yang dihasilkan dari fungsi ini adalah sebuah DataFrame dengan kolom `["xcenter", "ycenter", "width", "height"]`.
    <br>
    Baris kode dalam fungsi detect berikut merupakan inisialisasi argumen-argumen untuk menjalankan object detection.

    ```python
    def detect(weights,  # model.pt path(s)
            source,  # file/dir/URL/glob, 0 for webcam
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.5,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            # obj_double="cage_wheel_track",
            ):


        ... #rest of code

    ```
    <br>

    Pada baris berikut, berisi kode untuk melakukan beberapa setting dan penyiapan input untuk proses loading inferensi model.
    ```python 
    ...

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        # Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    ...

    ```
    <br>

    Proses inferensi model dan prediksi dimuat dalam baris kode berikut. Proses inferensi menggunakan tiga argumen yaitu 'im', 'augment', dan 'visualize'. Outputnya akhir disimpan dalam list `xywh = []`
    ```python

    ...

        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            
            pred = model(im, augment=augment, visualize=visualize)
        # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # names_class = det[:, -1].unique()
                    xywh = []
                    for *xyxy, conf, cls in reversed(det):
                        xywh.append((int(xyxy[0]+ (xyxy[2]-xyxy[0])/2), int(xyxy[1]+ (xyxy[3]-xyxy[1])/2),int(xyxy[2]-xyxy[0]),int(xyxy[3]-xyxy[1])))
                else:
                    xywh = None
                    class_list = None

    ...

    ```
    <br>

    Step terakhir dalam fungsi `detect()` adalah mengaplikasikan fungsi `sorting_index(data)` di mana argumen 'data' diambil dari output prediksi sebelumnya, yaitu list `xywh`. Dari step ini didapatkan output DataFrame dengan kolom `["xcenter", "ycenter", "width", "height"]` yang disimpan dalam variabel `df`.
    ```python

    ...

        if xywh is not None:
            sorted_idx = sorting_index(xywh)
            df = (pd.DataFrame(sorted_idx)).rename(columns={0:'xcenter', 1: 'ycenter', 2:'width', 3:'height'}) # create df from data, without count column
            df['idx']= df.index+1
            radius = [min(data[2], data[3]) // 2.5 for data in df.values]
            df['radius'] = radius
            return df
        else:
            column_names = ["xcenter", "ycenter", "width", "height"]
            df = (pd.DataFrame(columns = column_names))
            return df

    ...

    ```

    <br>

- #### `get_img_size()` function
    Fungsi `get_img_size()` bertujuan untuk mendapatkan dimensi suatu gambar yang dimuat dalam `img_path`. Fungsi ini mengembalikan output berupa ukuran gambar dalam format lebar x tinggi `(width,height)`.

    ```python
    ...

    def get_img_size(img_path):
        im = cv2.imread(img_path)
        im_sz = im.shape
        return im_sz[1], im_sz[0] # width & height
    ```

<br>

# Testing Program
Selalu lakukan testing program langsung menggunakans serverai. Lakukan ssh ke server ai dengan `serverai@192.168.168.195`. Gunakan environment yang sesuai dengan penjelasan diatas. Running program python seperti biasa, `python app.py`. Pastikan saat itu port tidak terpakai oleh aplikasi lain. Jika program sudah berjalan, lakukan pengujian dengan mengirimkan gambar sample delam api.

_Lihat dokumentasi api selengkapnya [disini](http://ai.quick.com/documentation/object-countings/)_