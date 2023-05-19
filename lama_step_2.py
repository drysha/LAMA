import datetime
from lama.bin.predict import *
import cv2
from cv2 import dnn_superres

@hydra.main(config_path='./lama/configs/prediction/', config_name='default.yaml')
def step2(predict_config: OmegaConf):
    step1dir = predict_config.step1dir
    dict_dir = ["5_create_mask","6_change_size","7_lama_result","8_upsc_result","9_result_on_floor"]
    now = datetime.datetime.now()
    log = step1dir + "/" + now.strftime("%d-%m-%Y_%H_%M_%S")
    os.mkdir(log)
    for folder in dict_dir:
        if not os.path.isdir(log + "/" + folder):
            os.mkdir(log + "/" + folder)

    def cut_area(path1,path2):
        path_mask = path1 + "/masks/"
        path_original = path1 + "/4_floor/"
        path_cut_area = path2 + "/5_create_mask/"

        dirs = os.listdir(path_mask)
        for item in dirs:
            mask = cv2.imread(path_mask + item)
            orig = cv2.imread(path_original + item.replace("_"+item.split("_")[-1],"") +".png")

            arr_mask = np.array(mask)
            height, width, channels = arr_mask.shape

            Xmin = [2500,0]
            Xmax = [0,0]
            Ymin = [0,2500]
            Ymax = [0,0]




            for strl in range(len(arr_mask)):
                for px in range(len(arr_mask[strl])):
                    if arr_mask[strl][px][0] == 255 and arr_mask[strl][px][1] == 255 and arr_mask[strl][px][2] == 255:
                        if Xmin[0] > px:
                            Xmin = [px,strl]
                        if Xmax[0] < px:
                            Xmax = [px,strl]
                        if Ymin[1] > strl:
                            Ymin = [px,strl]
                        if Ymax[1] < strl:
                            Ymax = [px,strl]



            sqXYlt = [Xmin[0],Ymin[1]]
            sqXYrb = [Xmax[0],Ymax[1]]

            sqCUT = [Xmax[0]-Xmin[0],Ymax[1]-Ymin[1]]

            sqREsize = [int((sqCUT[0]*0.4)//2),int((sqCUT[1]*0.4)//2)]
            sqRESlt = [sqXYlt[0]-sqREsize[0],sqXYlt[1]-sqREsize[1]]
            sqRESrb = [sqXYrb[0]+sqREsize[0],sqXYrb[1]+sqREsize[1]]

            if sqRESlt[0]<0:
                sqRESlt[0] = 0
            if sqRESlt[1]<0:
                sqRESlt[1] = 0
            if sqRESrb[0] > width-1:
                sqRESrb[0] = width-1
            if sqRESrb[1] > height-1:
                sqRESrb[1] = height-1

            crop_img = orig[sqRESlt[1]:sqRESrb[1], sqRESlt[0]:sqRESrb[0]]
            crop_mask = mask[sqRESlt[1]:sqRESrb[1], sqRESlt[0]:sqRESrb[0]]

            _, width, _ = crop_mask.shape


            if width > 1500:
                cv2.imwrite(path_cut_area + item.replace("_"+item.split("_")[-1],"") + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1])+ "%" + "$4$" + ".png",crop_img)
                cv2.imwrite(path_cut_area + item.replace("_"+item.split("_")[-1],"") + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1])+ "%" + "$4$" + "_" + item.split("_")[-1],crop_mask)
            elif width > 750:
                cv2.imwrite(path_cut_area + item.replace("_"+item.split("_")[-1],"") + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$2$" + ".png",crop_img)
                cv2.imwrite(path_cut_area + item.replace("_"+item.split("_")[-1],"") + "%" + str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$2$"+ "_" + item.split("_")[-1], crop_mask)
            else:
                cv2.imwrite(path_cut_area + item.replace("_"+item.split("_")[-1],"") + "%" +str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$0$" + ".png",crop_img)
                cv2.imwrite(path_cut_area + item.replace("_"+item.split("_")[-1],"") + "%" +str(sqRESlt[0]) + "%" + str(sqRESlt[1]) + "%" + "$0$" + "_" + item.split("_")[-1], crop_mask)

    def resize(path1):
        path = path1 + "/5_create_mask/"
        OUTpath = path1 + "/6_change_size/"
        dirs = os.listdir(path)
        for item in dirs:
            if os.path.isfile(path + item):
                im = cv2.imread(path + item)
                height, width, _ = im.shape
                f, e = os.path.splitext(path + item)
                Res = OUTpath + item
                if width > 1500:
                    resized_down = cv2.resize(im, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)
                    res_img = resized_down
                elif width > 750:
                    resized_down = cv2.resize(im, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
                    res_img = resized_down
                else:
                    res_img = im

            if "mask" in item:
                res_arr = np.array(res_img)
                white = np.array([255, 255, 255])
                black = np.array([0, 0, 0])

                for strl in range(len(res_arr)):
                    for px in range(len(res_arr[strl])):
                        if not np.array_equal(res_arr[strl][px], white) and not np.array_equal(res_arr[strl][px],black):
                            res_arr[strl][px] = white

                cv2.imwrite(Res, res_arr)
            else:
                cv2.imwrite(Res, res_img)

        print("resize - finished")

    def upsc(path1,modelpathx4, modelpathx2, modeltype):
        impath = path1 + "/7_lama_result/"
        outpath = path1 + "/8_upsc_result/"
        sr = dnn_superres.DnnSuperResImpl_create()
        dirs = os.listdir(impath)
        for item in dirs:
            if os.path.isfile(impath + item):
                image = cv2.imread(impath + item)
                if item.split('$')[-2] == "2":
                    sr.readModel(modelpathx2)
                    sr.setModel(modeltype, 2)
                    result = sr.upsample(image)
                    cv2.imwrite(outpath + item, result)
                elif item.split('$')[-2] == "4":
                    sr.readModel(modelpathx4)
                    sr.setModel(modeltype, 4)
                    result = sr.upsample(image)
                    cv2.imwrite(outpath + item, result)
                else:
                    cv2.imwrite(outpath + item, image)
        print("upsc finished")

    def CopyPaste(path1,path2):
        path_result = path2 + "/9_result_on_floor/"
        path_mask_floor = path2 + "/5_create_mask/"
        path_upsc = path2 + "/8_upsc_result/"
        path_orig = path1 + "/4_floor/"
        dirs = os.listdir(path_mask_floor)
        for item in dirs:
            if "mask" in item:
                if os.path.isfile(path_mask_floor + item):
                    mask = cv2.imread(path_mask_floor + item)
                    orig = cv2.imread(path_orig + item.split("%")[0] + ".png")
                    upsc = cv2.imread(path_upsc + item)
                    mask_arr = np.array(mask)
                    orig_arr = np.array(orig)
                    upsc_arr = np.array(upsc)
                    white = np.array([255, 255, 255])
                    res = orig
                    for strl in range(len(mask)):
                        for px in range(len(mask[strl])):
                            if np.array_equal(mask_arr[strl][px], white):
                                orig_arr[strl + int(item.split("%")[-2])][px + int(item.split("%")[-3])] = \
                                upsc_arr[strl][px]

                    cv2.imwrite(path_result + item.split("_")[0] + ".png", orig_arr)
        print("paste into original - finished")



    cut_area(step1dir,log)
    resize(log)
    predict_config.indir = log+"/6_change_size/"
    predict_config.outdir = log+"/7_lama_result/"
    main(predict_config)
    upsc(log,predict_config.upsc.x4model,predict_config.upsc.x2model,predict_config.upsc.model_name)
    CopyPaste(step1dir,log)
