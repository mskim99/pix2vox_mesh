import numpy as np

with open('J:/Program/Pix2Vox-mesh/datasets/KISTI_volume_CT_mesh_sc128/KISTI_Vox_BD/00000024/f_0000001/model.obj', 'r') as f:
    vertices = []
    lines = f.readlines()
    num = 0
    for line in lines:
        strings = line.split()
        if len(strings) == 4 and strings[0] == 'v':
            vertices.append(float(strings[1]))
            vertices.append(float(strings[2]))
            vertices.append(float(strings[3]))
            num += 1

vertices = np.array(vertices)
print(vertices.shape)