import matplotlib.pyplot as plt
import src.utilities.reading_images as ons
import base64
import mysql.connector
import h5py
import numpy as np
import matplotlib as mp



def returnHeatmap(id_patient):
    hd_b_l_mlo = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_benign/0_L_MLO.hdf5",'r')
    hd_b_l_cc = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_benign/0_L_CC.hdf5",'r')
    hd_b_r_mlo = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_benign/0_R_MLO.hdf5",'r')
    hd_b_r_cc = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_benign/0_R_CC.hdf5",'r')
    hd_m_l_mlo = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_malignant/0_L_MLO.hdf5",'r')
    hd_m_l_cc = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_malignant/0_L_CC.hdf5",'r')
    hd_m_r_mlo = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_malignant/0_R_MLO.hdf5",'r')
    hd_m_r_cc = h5py.File("C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_" + id_patient + "/heatmap_malignant/0_R_CC.hdf5",'r')
    array1 = np.array(hd_b_l_mlo['image']).T
    array2 = np.array(hd_b_l_cc['image']).T
    array3 = np.array(hd_b_r_mlo['image']).T
    array4 = np.array(hd_b_r_cc['image']).T
    array5 = np.array(hd_m_l_mlo['image']).T
    array6 = np.array(hd_m_l_cc['image']).T
    array7 = np.array(hd_m_r_mlo['image']).T
    array8 = np.array(hd_m_r_cc['image']).T
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_l_mlo.png', array1)
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_l_cc.png', array2)
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_r_mlo.png', array3)
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_r_cc.png', array4)
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_l_mlo.png', array5)
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_l_cc.png', array6)
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_r_mlo.png', array7)
    mp.image.imsave('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_r_cc.png', array8)
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_l_mlo.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_b_l_mlo = str(b_l_mlo, 'utf-8')
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_l_cc.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_b_l_cc = str(b_l_mlo, 'utf-8')
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_r_mlo.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_b_r_mlo = str(b_l_mlo, 'utf-8')
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/b_r_cc.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_b_r_cc = str(b_l_mlo, 'utf-8')
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_l_mlo.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_m_l_mlo = str(b_l_mlo, 'utf-8')
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_l_cc.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_m_l_cc = str(b_l_mlo, 'utf-8')
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_r_mlo.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_m_r_mlo = str(b_l_mlo, 'utf-8')
    with open('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient +'/m_r_cc.png', "rb") as image_file1:
        b_l_mlo = base64.b64encode(image_file1.read())
        ch_m_r_cc = str(b_l_mlo, 'utf-8')

    sql = "INSERT INTO heatmap (b_l_mlo,b_l_cc,b_r_mlo,b_r_cc,patient_id,m_l_mlo,m_l_cc,m_r_mlo,m_r_cc) VALUES (%s, %s, %s, %s,%s, %s, %s, %s, %s)"
    val = (ch_b_l_mlo, str(ch_b_l_cc), str(ch_b_r_mlo), str(ch_b_r_cc), str(id_patient), str(ch_m_l_mlo),str(ch_m_l_cc),str(ch_m_r_mlo),str(ch_m_r_cc))
    print("bokbok")
    import src.utilities.IP as ip
    db_connect = mysql.connector.connect(
        host=ip.ip_serveur,
        user="amir",
        passwd="amir",
        database="pinktie",

    )

    '''db_connect = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            passwd="",
            database="pinktie",

        )'''
    mycursor = db_connect.cursor()
    mycursor.execute(sql, val)
    db_connect.commit()











