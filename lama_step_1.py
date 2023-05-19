import datetime
from lama.bin.predict import *
import cv2
import numpy
from numpy import pi
from PIL import Image
import shutil

project_dir = os.path.abspath(os.path.dirname(__file__))

dict_dir = ["1_original","2_to_cube", "3_cut_parts","4_floor","masks"]
now = datetime.datetime.now()
log = project_dir + "/" + now.strftime("%d-%m-%Y_%H_%M_%S")
os.mkdir(log)
for folder in dict_dir:
    if not os.path.isdir(log + "/" + folder):
        os.mkdir(log + "/" + folder)

@hydra.main(config_path='./lama/configs/prediction/', config_name='default.yaml')
def copy(predict_config: OmegaConf):
    source_dir = predict_config.origdir
    target_dir = log + "/1_original/"
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.copy(os.path.join(source_dir, file_name), target_dir)
    print("copy - finished")


def generate_mapping_data(image_width):
    # Output texture is 4 by 3 format (required by cubemap generators)
    out_tex_width = image_width
    out_tex_height = image_width * 3 // 4
    # Each cube face covers a quarter of the output image width
    face_edge_size = out_tex_width / 4

    # Create our numpy arrays
    out_mapping = numpy.zeros((out_tex_height, out_tex_width, 2), dtype="f4")
    xyz = numpy.zeros((out_tex_height * out_tex_width // 2, 3), dtype="f4")
    vals = numpy.zeros((out_tex_height * out_tex_width // 2, 3), dtype="i4")

    start, end = 0, 0
    # Here we are computing a "horizontal cross" made of cube map faces, since that's the format
    # most software need to generate a cubemap file.
    # This algorithm will go by pixel columns.
    # Interested pixels per column are mostly between face_edge_size and face_edge_size * 2
    # which will cover "the body" of the cross.
    # Apart from those, the pixels per column when rendering "the arms" of the cross, are going
    # to cover the whole column, and so from 0 to face_edge_size * 3.
    pix_column_range_1 = numpy.arange(0, face_edge_size * 3)
    pix_column_range_2 = numpy.arange(face_edge_size, face_edge_size * 2)
    # For each pixel column of the output texture ( so from 0 to width excluded)
    for col_idx in range(out_tex_width):
        face = int(col_idx / face_edge_size)
        # Each column will compute pixels in the "central strip" of the texture
        # Except for face 1 where we compute the "arms of the cross" which range
        # will interest every pixel of the column.
        col_pix_range = pix_column_range_1 if face == 1 else pix_column_range_2

        end += len(col_pix_range)
        vals[start:end, 0] = col_pix_range
        vals[start:end, 1] = col_idx
        vals[start:end, 2] = face
        start = end

    # Top/bottom are special conditions
    # Create references to values of each dimension of vals
    col_pix_range, col_idx, face = vals.T
    # Set each face index value where column pixel range less than cube face size, to 4.
    # That's because in the horizontal cross, face 4 it's the only face having this condition.
    face[col_pix_range < face_edge_size] = 4  # top
    # Equivalently, every face value where the column pixel range is greater or equal 2 times the
    # size of a cube face, belongs to face 5.
    face[col_pix_range >= 2 * face_edge_size] = 5  # bottom

    # Convert from texel coordinates to 3D points on cube
    a = 2.0 * col_idx / face_edge_size
    b = 2.0 * col_pix_range / face_edge_size
    one_arr = numpy.ones(len(a))
    for k in range(6):
        # Here face_idx points to the values of vals where face equals to the current face index k
        face_idx = face == k

        # Create helper arrays with same dimension as face_idx
        one_arr_idx = one_arr[face_idx]
        a_idx = a[face_idx]
        b_idx = b[face_idx]

        if k == 0:  # X-positive face mapping
            vals_to_use = [one_arr_idx, a_idx - 1.0, 3.0 - b_idx]
        elif k == 1:  # Y-positive face mapping
            vals_to_use = [3.0 - a_idx, one_arr_idx, 3.0 - b_idx]
        elif k == 2:  # X-negative face mapping
            vals_to_use = [-one_arr_idx, 5.0 - a_idx, 3.0 - b_idx]
        elif k == 3:  # Y-negative face mapping
            vals_to_use = [a_idx - 7.0, -one_arr_idx, 3.0 - b_idx]
        elif k == 4:  # Z-positive face mapping
            vals_to_use = [3.0 - a_idx, b_idx - 1.0, one_arr_idx]
        elif k == 5:  # bottom face mapping
            vals_to_use = [3.0 - a_idx, 5.0 - b_idx, -one_arr_idx]
            print(len(vals_to_use[0]))
        # Assign computed values for the dimesions of xyz pointed by face_idx
        xyz[face_idx] = numpy.array(vals_to_use).T

    # Convert to phi and theta
    x, y, z = xyz.T
    phi = numpy.arctan2(y, x)
    # Be r the vector from the sphere origin to the point on the cube
    # here r_proj_xy=r*sin(theta) and z=r*cos(theta), that's why below theta=pi/2 - arctg(z,r_proj_xy)
    r_proj_xy = numpy.sqrt(x ** 2 + y ** 2)
    theta = pi / 2 - numpy.arctan2(z, r_proj_xy)

    # From polar to input texel coordinates
    # Note: The input texture is assumed to be sized
    # width = 4.0 * face_edge_size
    # height = 2.0 * face_edge_size
    uf = 4.0 * face_edge_size * phi / (2.0 * pi) % out_tex_width
    vf = 2.0 * face_edge_size * theta / pi

    out_mapping[col_pix_range, col_idx, 0] = uf
    out_mapping[col_pix_range, col_idx, 1] = vf

    return out_mapping[:, :, 0], out_mapping[:, :, 1]


def create_cube_and_split():
    impath = log + "/1_original/"
    outpath = log + "/3_cut_parts/"
    OutCubePath = log + "/2_to_cube/"
    floorpath = log + "/4_floor/"
    dirs = os.listdir(impath)
    for foto in dirs:
        if os.path.isfile(impath + foto):
            imgIn = Image.open(impath + foto)
            inSize = imgIn.size

            map_x_32, map_y_32 = generate_mapping_data(inSize[0])
            cubemap = cv2.remap(numpy.array(imgIn), map_x_32, map_y_32, cv2.INTER_LINEAR)

            imgOut = Image.fromarray(cubemap)
            imgOut.save(OutCubePath + foto.split(".")[0] + "_cube.png")
            x = imgOut.size[0] / 4
            y = imgOut.size[1] / 3

            imgBottom = imgOut.crop((x, y * 2, x * 2, y * 3))
            imgTop = imgOut.crop((x, 0, x * 2, y))
            imgLeft = imgOut.crop((0, y, x, y * 2))
            imgRight = imgOut.crop((x * 2, y, x * 3, y * 2))
            imgBack = imgOut.crop((x * 3, y, x * 4, y * 2))
            imgFront = imgOut.crop((x, y, x * 2, y * 2))

            imgBottom.save(outpath + foto.split(".")[0] + "_bottom.png")
            imgTop.save(outpath + foto.split(".")[0] + "_top.png")
            imgLeft.save(outpath + foto.split(".")[0] + "_left.png")
            imgRight.save(outpath + foto.split(".")[0] + "_right.png")
            imgBack.save(outpath + foto.split(".")[0] + "_back.png")
            imgFront.save(outpath + foto.split(".")[0] + "_front.png")
            imgBottom.save(floorpath + foto.split(".")[0] + ".png")
    print("2_cube, 3_cut_parts, 4_floor - finished")

copy()
create_cube_and_split()





