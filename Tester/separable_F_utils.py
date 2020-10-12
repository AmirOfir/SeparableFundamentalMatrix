import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from separable_F_core  import  h_coordinates,sed_core



def output_results(img_num,F_ELH_R, F8ransac,img1c,img2c,pts1,pts2,inlier_threshold):

    err_8   = symmetric_epipolar_distance(F8ransac, pts1,pts2)
    err_4_r = symmetric_epipolar_distance(F_ELH_R, pts1,pts2)

    err_4_mean    = np.mean(err_4_r[err_4_r < inlier_threshold])
    inliers_4     = sum(err_4_r < inlier_threshold)
    err_8_mean    = np.mean(err_8[err_8 < inlier_threshold])
    inliers_8     = sum(err_8 < inlier_threshold)

    title_ours  ="Image"+str(img_num)+",Separable F:Error is {0:2.2f} for {1} inliers of {2} points".format(err_4_mean,inliers_4,len(err_4_r))
    title_ransac="Image"+str(img_num)+",RANSAC 8pts:Error is {0:2.2} for {1} inliers of {2} points".format(err_8_mean,inliers_8, len(err_8))
    print(title_ours)
    print(title_ransac)

    # pts1i1 = pts1[err_4_r < inlier_threshold].astype(int)
    # pts2i1 = pts2[err_4_r < inlier_threshold].astype(int)
    # display_inliers(F_ELH_R, img1c, img2c, pts1i1[::2], pts2i1[::2],title_ours)

    # pts1i = pts1[err_8 < inlier_threshold].astype(int)
    # pts2i = pts2[err_8 < inlier_threshold].astype(int)
    # display_inliers(F8ransac, img1c, img2c, pts1i[::2], pts2i[::2],title_ransac)



def symmetric_epipolar_distance(F,pts1,pts2):
    x1 = h_coordinates(np.transpose(pts1))
    x2 = h_coordinates(np.transpose(pts2))
    return sed_core(F,x1,x2)

import os 
def get_matched_points(img1_name,img2_name,scale_factor):
    print("Matching points {},{}".format(img1_name, img2_name))
    img1c = cv2.imread(img1_name)  # queryImage
    img2c = cv2.imread(img2_name)  # trainImage
    if (img1c is None):
        img1c = cv2.imread(img1_name.replace('../', ''))
    if (img2c is None):
        img2c = cv2.imread(img2_name.replace('../', ''))

    if scale_factor<1:
        print("Image shape before scaling  {}".format(np.shape(img1c)))
        img1c = cv2.resize(img1c, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        img2c = cv2.resize(img2c, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    print("Image shape after scaling  {}".format(np.shape(img1c)))
    img1=cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2c, cv2.COLOR_BGR2GRAY)

    descriptor = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = descriptor.detectAndCompute(img1,None)
    kp2, desc2 = descriptor.detectAndCompute(img2,None)

    pts1all = np.float32([kp.pt for kp in kp1])
    pts2all = np.float32([kp.pt for kp in kp2])


    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(desc1,desc2,k=2)
    all_matches = sorted(all_matches, key=lambda x: x[0].distance)
    all_matches =all_matches[0:1000]
    matches = []

    for m, n in all_matches:
        if m.distance < 0.8 * n.distance:
            matches.append([m])
    pts1 = np.float32([pts1all[m[0].queryIdx] for m in matches])
    pts2 = np.float32([pts2all[m[0].trainIdx] for m in matches])

    return pts1,pts2,img1c,img2c,img1,img2

def get_image_points_on_line(w1,h1,line2,distance=0.1):
    h_pt, w_pt = np.indices((w1, h1))
    img_points = np.concatenate((h_pt.reshape(-1, 1), w_pt.reshape(-1, 1)), axis=1)
    a, b, c = line2 #/ line2[2]
    d = abs((a * img_points[:, 0] + b * img_points[:, 1] + c)) / (math.sqrt(a * a + b * b))
    line_pts = np.array(img_points[np.where(d < distance), :]).squeeze()
    return line_pts


def showEpipolarLines(F, pts1, img2,pts2=None):
    h1=img2.shape[0]
    w1=img2.shape[1]
    plt.imshow(img2)
    pts1_hom = np.hstack((pts1, np.ones((len(pts1), 1)))).T
    epiline_vec = F.dot(pts1_hom).T
    for line in epiline_vec:
        mp=get_image_points_on_line(w1,h1,line)
        plt.plot(mp[:,0], mp[:,1], 'g')
    if pts2 is not None:
        plt.plot(pts2[:, 0], pts2[:, 1], 'r*')
    # plt.axis('off')
    # plt.show()



def plot_epipolarhomography_lines(dbg_inf,Fgt,img1,img2,title):
    if (not dbg_inf is None):
        line=dbg_inf['top_lines']
        lines_img1=dbg_inf['lines_img1']
        lines_img2=dbg_inf['lines_img2']
        rescale=dbg_inf['rescale']

        Fgt= np.dot(np.diag([1/rescale, 1/rescale, 1]), np.dot(Fgt, np.diag([1/rescale, 1/rescale, 1])))
        img1 = cv2.resize(img1, None, fx=rescale, fy=rescale, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx=rescale, fy=rescale, interpolation=cv2.INTER_AREA)

        plot_matching_lines(Fgt,img1,img2,lines_img1[line['line1_index']], lines_img2[line['line2_index']],
                            line['line_points_1'], line['line_points_2'],title)


def plot_matching_lines(Fgt,img1,img2,line1_dict,line2_dict,mp1,mp2,title):
    img1a = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2a = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.axis('off')
    f, axarr = plt.subplots(1, 2)
    f.suptitle(title)
    axarr[0].axis('off')
    axarr[1].axis('off')

    plt.sca(axarr[0])
    line1 = line1_dict['line_eq_abc']
    h1, w1, _ = img1a.shape
    line_pts=get_image_points_on_line(h1,w1,line1)
    plt.plot(line_pts[:, 0], line_pts[:, 1], 'b')
    for (pt1,pt2) in zip(mp1,mp2):
        pt1 = [int(p) for p in pt1]
        plt.plot(pt1[0], pt1[1],'ro')
    plt.title('1nd Image')
    showEpipolarLines(Fgt.T, mp2.astype(np.int), img1a)

    plt.sca(axarr[1])
    line2 = line2_dict['line_eq_abc']
    h1, w1, _ = img2a.shape
    line_pts=get_image_points_on_line(h1,w1,line2)
    plt.plot(line_pts[:, 0], line_pts[:, 1], 'b')
    for (pt1,pt2) in zip(mp1,mp2):
        pt2 = [int(p) for p in pt2]
        plt.plot(pt2[0], pt2[1],'ro')
    plt.title('2nd Image')
    showEpipolarLines(Fgt, mp1.astype(np.int), img2a)

    plt.show()


def draw_epilines(img1, img2, lines, pts1, pts2):
    c=img1.shape[1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), (0,255,50), 2,cv2.LINE_AA)
        img1 = cv2.circle(img1, tuple(pt1), 5, [255,0,0], -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, [255,0,0], -1)
    return img1, img2



def display_inliers(F,img1,img2,pts1i,pts2i,title="None"):
    lines1 = cv2.computeCorrespondEpilines(pts2i.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = draw_epilines(img1, img2, lines1, pts1i, pts2i)

    lines2 = cv2.computeCorrespondEpilines(pts1i.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = draw_epilines(img2, img1, lines2, pts2i, pts1i)

    plt.axis('off')
    f, axarr = plt.subplots(1, 2)
    f.suptitle(title)
    axarr[0].imshow(img5)
    axarr[0].axis('off')
    axarr[1].imshow(img3)
    axarr[1].axis('off')
    plt.show()





