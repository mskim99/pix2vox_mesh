import shutil
import glob
import os

############################################################################################################################################
# Volume Processing
############################################################################################################################################

# File Copy & Paste
'''
path = "J:\\DK_Data_Process\\i_1-2_Slices_Cropped_Volume\\23_24_spine\\Axial"
for i in range(1, 56):
    if not os.path.isdir(path + '\\f_' + str(i).zfill(3)):
        continue
    axial_file_list = glob.glob(path + '\\f_' + str(i).zfill(3) + '\\*')
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/renderings.txt', 'w')
    for j, file_name in enumerate(axial_file_list):
        shutil.copy(file_name, 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/a' + str(j).zfill(3) + '.png')
        f.write('a' + str(j).zfill(3) + '.png\n')
    print('Axial Female ' + str(i) + ' ended')

path = "J:\\DK_Data_Process\\i_1-2_Slices_Cropped_Volume\\23_24_spine\\Coronal"
for i in range(1, 56):
    if not os.path.isdir(path + '\\f_' + str(i).zfill(3)):
        continue
    coronal_file_list = glob.glob(path + '\\f_' + str(i).zfill(3) + '\\*')
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/renderings.txt', 'a')
    for j, file_name in enumerate(coronal_file_list):
        shutil.copy(file_name, 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/c' + str(j).zfill(3) + '.png')
        f.write('c' + str(j).zfill(3) + '.png\n')
    print('Coronal Female ' + str(i) + ' ended')

path = "J:\\DK_Data_Process\\i_1-2_Slices_Cropped_Volume\\23_24_spine\\Sagittal"
for i in range(1, 56):
    if not os.path.isdir(path + '\\f_' + str(i).zfill(3)):
        continue
    sagittal_file_list = glob.glob(path + '\\f_' + str(i).zfill(3) + '\\*')
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/renderings.txt', 'a')
    for j, file_name in enumerate(sagittal_file_list):
        shutil.copy(file_name, 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/s' + str(j).zfill(3) + '.png')
        f.write('s' + str(j).zfill(3) + '.png\n')
    print('Sagittal Female ' + str(i) + ' ended')

path = "J:\\DK_Data_Process\\i_1-2_Slices_Cropped_Volume\\23_24_spine\\Axial"
for i in range(1, 58):
    if not os.path.isdir(path + '\\m_' + str(i).zfill(3)):
        continue
    axial_file_list = glob.glob(path + '\\m_' + str(i).zfill(3) + '\\*')
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/renderings.txt', 'w')
    for j, file_name in enumerate(axial_file_list):
        shutil.copy(file_name, 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/a' + str(j).zfill(3) + '.png')
        f.write('a' + str(j).zfill(3) + '.png\n')
    print('Axial Male ' + str(i) + ' ended')

path = "J:\\DK_Data_Process\\i_1-2_Slices_Cropped_Volume\\23_24_spine\\Coronal"
for i in range(1, 58):
    if not os.path.isdir(path + '\\m_' + str(i).zfill(3)):
        continue
    coronal_file_list = glob.glob(path + '\\m_' + str(i).zfill(3) + '\\*')
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/renderings.txt', 'a')
    for j, file_name in enumerate(coronal_file_list):
        shutil.copy(file_name, 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/c' + str(j).zfill(3) + '.png')
        f.write('c' + str(j).zfill(3) + '.png\n')
    print('Coronal Male ' + str(i) + ' ended')

path = "J:\\DK_Data_Process\\i_1-2_Slices_Cropped_Volume\\23_24_spine\\Sagittal"
for i in range(1, 58):
    if not os.path.isdir(path + '\\m_' + str(i).zfill(3)):
        continue
    sagittal_file_list = glob.glob(path + '\\m_' + str(i).zfill(3) + '\\*')
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/renderings.txt', 'a')
    for j, file_name in enumerate(sagittal_file_list):
        shutil.copy(file_name, 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/s' + str(j).zfill(3) + '.png')
        f.write('s' + str(j).zfill(3) + '.png\n')
    print('Sagittal Male ' + str(i) + ' ended')
    '''
# Check File Existance
'''
for i in range(1, 58):
    path = 'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/*'
    files = glob.glob(path)
    print('Female ' + str(i) + '  : ' + str(files.__len__()))

for i in range(1, 58):
    path = 'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/*'
    files = glob.glob(path)
    print('Male ' + str(i) + '  : ' + str(files.__len__()))
    '''

# Store img number labeled by 'a', 'c', 's'
'''
for i in range(1, 58):
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/views.txt', 'w')

    path = 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/a*'
    files = glob.glob(path)
    f.write(str(files.__len__()) + ' ')

    path = 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/c*'
    files = glob.glob(path)
    f.write(str(files.__len__()) + ' ')

    path = 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/s*'
    files = glob.glob(path)
    f.write(str(files.__len__()) + ' ')

for i in range(1, 58):
    f = open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/views.txt', 'w')

    path = 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/a*'
    files = glob.glob(path)
    f.write(str(files.__len__()) + ' ')

    path = 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/c*'
    files = glob.glob(path)
    f.write(str(files.__len__()) + ' ')

    path = 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc128_23_24/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/s*'
    files = glob.glob(path)
    f.write(str(files.__len__()) + ' ')
    '''

############################################################################################################################################
# X-ray Image Processing
############################################################################################################################################

for i in range (49, 58):
    '''
    # Woman
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Axial/f_' + str(i).zfill(3) + '_Axial_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Coronal/f_' + str(i).zfill(3) + '_Coronal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Sagittal/f_' + str(i).zfill(3) + '_Sagittal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/02.png')
    
    # Man
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Axial/m_' + str(i).zfill(3) + '_Axial_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Coronal/m_' + str(i).zfill(3) + '_Coronal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Sagittal/m_' + str(i).zfill(3) + '_Sagittal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/02.png')
    '''
    # Woman
    '''
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/00.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_ct/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/01.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_ct/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/03.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_ct/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/02.png')
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/04.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_ct/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/03.png')
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/05.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_ct/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/04.png')
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/06.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_ct/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/05.png')
                '''
    # Man
    '''
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Axial/m_' + str(i).zfill(3) + '_Axial_cropped_xray.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_xray/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Coronal/m_' + str(i).zfill(3) + '_Coronal_cropped_xray.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_xray/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('J:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Sagittal/m_' + str(i).zfill(3) + '_Sagittal_cropped_xray.png',
                'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_xray/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/02.png')
                '''

for i in range (1, 58):
    '''
    if os.path.exists('J:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_128/f_' + str(i).zfill(3) + '_vrt_23_24.binvox'):
        shutil.copy('J:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_128/f_' + str(i).zfill(3) + '_vrt_23_24.binvox',
        '''
    if os.path.exists('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_256/gtv_m_' + str(i).zfill(7) + '_a.binvox'):
        shutil.copy('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_256/gtv_m_' + str(i).zfill(7) + '_a.binvox',
        'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc256/KISTI_Vox/00000024/m_' + str(i).zfill(7) + '/model.binvox')
        print(str(i) + ' exists')

# Rendering Metadata
'''
for i in range (1, 58):
    # shutil.copy('J:/renderings.txt', 'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_sc128_23_24/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/views.txt')
    shutil.copy('J:/renderings.txt', 'J:/Program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_23_24_ct/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/renderings.txt')
    '''






