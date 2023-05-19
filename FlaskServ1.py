from flask import Flask, render_template, request, send_from_directory, make_response
import os
import time
from lama.bin.predict import *
from toCube import *
import datetime
import base64
from PIL import Image
from io import BytesIO
from cv2 import dnn_superres

app = Flask(__name__, template_folder='C:/Users/Hi-tech/PycharmProjects/pythonProject8')

@app.route('/')
def lama_upload():
    return render_template('1.php')

@app.route('/image/<p1>/')
def uploaded_file_test(p1):
    return send_from_directory(p1 + "/9_result_on_floor/", "result.png")

@app.route('/upload/image', methods=['POST'])
def upload_image():

    now = datetime.datetime.now()
    project_dir = os.path.abspath(os.path.dirname(__file__))
    log1 = now.strftime("%d-%m-%Y_%H_%M_%S.%f")
    log = project_dir + "/" + log1
    os.mkdir(log)
    dict_dir = ["5_image_and_mask", "6_croped_and_resize", "7_lama_result", "8_upsc_result", "9_result_on_floor"]
    for folder in dict_dir:
        if not os.path.isdir(log + "/" + folder):
            os.mkdir(log + "/" + folder)
    b64_string = request.form['image1']
    img_data = b64_string.split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    img.save(log + "/" + dict_dir[0] + "/image.png")
    b64_string = request.form['image2']
    img_data = b64_string.split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    img.save(log + "/" + dict_dir[0] + "/image_mask.png")

    def cut_area(in_path,resize_path):

        mask = cv2.imread(in_path + "/image_mask.png")
        orig = cv2.imread(in_path + "/image.png")
        arr_mask = np.array(mask)
        height, width, channels = arr_mask.shape

        white = np.array([255, 255, 255])
        black = np.array([0, 0, 0])

        for strl in range(len(arr_mask)):
            for px in range(len(arr_mask[strl])):
                if not np.array_equal(arr_mask[strl][px], white) and not np.array_equal(arr_mask[strl][px], black):
                    arr_mask[strl][px] = white
        cv2.imwrite(in_path + "/image_mask.png", arr_mask)

        Xmin = [2500, 0]
        Xmax = [0, 0]
        Ymin = [0, 2500]
        Ymax = [0, 0]

        for strl in range(len(arr_mask)):
            for px in range(len(arr_mask[strl])):
                if arr_mask[strl][px][0] == 255 and arr_mask[strl][px][1] == 255 and arr_mask[strl][px][2] == 255:
                    if Xmin[0] > px:
                        Xmin = [px, strl]
                    if Xmax[0] < px:
                        Xmax = [px, strl]
                    if Ymin[1] > strl:
                        Ymin = [px, strl]
                    if Ymax[1] < strl:
                        Ymax = [px, strl]

        sqXYlt = [Xmin[0], Ymin[1]]
        sqXYrb = [Xmax[0], Ymax[1]]

        sqCUT = [Xmax[0] - Xmin[0], Ymax[1] - Ymin[1]]

        sqREsize = [int((sqCUT[0]) // 2), int((sqCUT[1]) // 2)]
        sqRESlt = [sqXYlt[0] - sqREsize[0], sqXYlt[1] - sqREsize[1]]
        sqRESrb = [sqXYrb[0] + sqREsize[0], sqXYrb[1] + sqREsize[1]]

        if sqRESlt[0] < 0:
            sqRESlt[0] = 0
        if sqRESlt[1] < 0:
            sqRESlt[1] = 0
        if sqRESrb[0] > width - 1:
            sqRESrb[0] = width - 1
        if sqRESrb[1] > height - 1:
            sqRESrb[1] = height - 1

        crop_img = orig[sqRESlt[1]:sqRESrb[1], sqRESlt[0]:sqRESrb[0]]
        crop_mask = arr_mask[sqRESlt[1]:sqRESrb[1], sqRESlt[0]:sqRESrb[0]]

        _, width, _ = crop_mask.shape

        if width > 1500:
            resized_down = cv2.resize(crop_img, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)
            resized_down_mask = cv2.resize(crop_mask, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(resize_path + "/image" + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$4$" + ".png", resized_down)
            cv2.imwrite(resize_path + "/image" + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$4$" + "_mask.png", resized_down_mask)
        elif width > 750:
            resized_down = cv2.resize(crop_img, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
            resized_down_mask = cv2.resize(crop_mask, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(resize_path + "/image" + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$2$" + ".png", resized_down)
            cv2.imwrite(resize_path + "/image" + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$2$" + "_mask.png", resized_down_mask)
        else:
            cv2.imwrite(resize_path + "/image" + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$0$" + ".png",crop_img)
            cv2.imwrite(resize_path + "/image" + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$0$" + "_mask.png",crop_mask)

    def upsc(in_path,out_path,modelpathx4, modelpathx2, modeltype):
        list_of_files = os.listdir(in_path)
        list_of_images = [file for file in list_of_files]
        sr = dnn_superres.DnnSuperResImpl_create()
        if os.path.isfile(in_path + "/" + list_of_images[0]):
            image = cv2.imread(in_path + "/" + list_of_images[0])
            if list_of_images[0].split('$')[-2] == "2":
                sr.readModel(modelpathx2)
                sr.setModel(modeltype, 2)
                result = sr.upsample(image)
                cv2.imwrite(out_path + "/" + list_of_images[0], result)
            elif list_of_images[0].split('$')[-2] == "4":
                sr.readModel(modelpathx4)
                sr.setModel(modeltype, 4)
                result = sr.upsample(image)
                cv2.imwrite(out_path + "/" + list_of_images[0], result)
            else:
                cv2.imwrite(out_path + "/" + list_of_images[0], image)
        print("upsc finished")

    def CopyPaste(in_path_orig,in_path_upsk,out_path):
        list_of_files = os.listdir(in_path_upsk)
        list_of_images = [file for file in list_of_files]
        mask = cv2.imread(in_path_orig + "/" + "image_mask.png")
        orig = cv2.imread(in_path_orig + "/" + "image.png")
        upsc = cv2.imread(in_path_upsk + "/" + list_of_images[0])
        mask_arr = np.array(mask)
        orig_arr = np.array(orig)
        upsc_arr = np.array(upsc)
        white = np.array([255, 255, 255])
        for strl in range(len(mask)):
            for px in range(len(mask[strl])):
                if np.array_equal(mask_arr[strl][px], white):
                    orig_arr[strl][px]=upsc_arr[strl-int(list_of_images[0].split("%")[2])][px - int(list_of_images[0].split("%")[1])]

        cv2.imwrite(out_path + "/result.png", orig_arr)
        print("paste into original - finished")



    image_and_mask_path = log + "/" + dict_dir[0]
    croped_mask_path = log + "/" + dict_dir[1]

    cut_area(image_and_mask_path, croped_mask_path)
    predict_config = OmegaConf.load('./lama/configs/prediction/default.yaml')
    predict_config.indir = log + "/" + dict_dir[1]
    predict_config.outdir = log + "/" + dict_dir[2]
    predict_config.model.path = "C:/Users/Hi-tech/PycharmProjects/pythonProject8/lama/bin/big-lama"
    predict_config.device = "cpu"
    main(predict_config)
    upsc(log+"/"+dict_dir[2],log+"/"+dict_dir[3],"C:/Users/Hi-tech/PycharmProjects/pythonProject8/UpscModels/FSRCNN_x4.pb","C:/Users/Hi-tech/PycharmProjects/pythonProject8/UpscModels/FSRCNN_x2.pb","fsrcnn")
    CopyPaste(log+"/"+dict_dir[0],log+"/"+dict_dir[3],log+"/"+dict_dir[4])
    return '{"rezalt":"http://127.0.0.1:5000/image/' + log1 + '"}'

app.run(debug=True)