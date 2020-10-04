import numpy as np
import cv2
import math
import scipy.spatial.distance
import random


### Copyrights: Gil Ben-Artzi, 2019
#
# Find fundamental matrix by sampling epipolar line homography
# theoretical and practical improvement up to orders of magnitudes
# of the required iterations, comparing to standard RANSAC, depending on the
# actual inliers ratio.
#


def separable_F(im_size_h_org,im_size_w_org,
                pts1org, pts2org,
                inlier_ratio    =0.4,
                inlier_threshold=3):
    return separable_F_core(im_size_h_org,im_size_w_org,
                            pts1org,pts2org,
                            inlier_ratio,inlier_threshold)

def separable_F_core(im_size_h_org,im_size_w_org, pts1org, pts2org,inlier_ratio,inlier_threshold,
                hough_rescale=-1,
                num_matching_pts_to_use=150,
                pixel_res=4,
                min_hough_points=4,
                theta_res=180,
                max_distance_pts_line=3,
                top_line_retries=2,
                min_shared_points=4): # line info

    result_F,result_dbg=[],[]
    # try:
        # auto resize
    if hough_rescale==-1:
        hough_rescale=(2*len(pts1org))/(im_size_h_org)
        #print(hough_rescale)
    if hough_rescale>=1: # only subsample
        hough_rescale=1

    top_lines_by_inliers=[]
    lines1=[]
    lines2=[]
    hough_rescale         = hough_rescale * 2 # for the first time
    max_distance_pts_line = max_distance_pts_line * 0.5
    # we sample a small subset of features to use in the hough transform, if our sample is too sparse, increase it
    for _ in range(top_line_retries):
        # rescale points and image size for fast line detection
        hough_rescale         = hough_rescale * 0.5
        max_distance_pts_line = max_distance_pts_line * 2
        pts1 = pts1org * hough_rescale
        pts2 = pts2org * hough_rescale
        im_size_h = int(round(im_size_h_org * hough_rescale)) + 3
        im_size_w = int(round(im_size_w_org * hough_rescale)) + 3

        lines_img1, lines_img2 = get_hough_lines(np.array([im_size_h,im_size_w]),
                                                 pts1,
                                                 pts2,
                                                 min_hough_points, pixel_res, theta_res,max_distance_pts_line,num_matching_pts_to_use)

        if len(lines_img1)!=0 and len(lines_img2) !=0:
            top_lines_by_inliers,lines1,lines2 = get_top_matching_lines(pts1, pts2, lines_img1, lines_img2,min_shared_points,inlier_ratio)

        if len(top_lines_by_inliers)>=2: # We found the lines
            break

        print("Sep F 4pts:Rescale again")


        
    if  len(top_lines_by_inliers)>0: # We have at least one line
        F_R,F_L,line_ix = sep_F_ransac_lines(top_lines_by_inliers, pts1, pts2, inlier_ratio,
                                                      inlier_threshold)
        diag = np.diag([hough_rescale, hough_rescale, 1])
        result_F = [np.dot(diag, np.dot(F, diag)) for F in [F_R, F_L]]

        result_dbg = {'top_lines': top_lines_by_inliers[line_ix], 'lines_img1': lines1, 'lines_img2': lines2,
                      'rescale': hough_rescale}

    # except Exception as e:
    #     print(e)


    return  result_F,result_dbg




def sampson_error(F,x1,x2):
    Fx1 = np.dot(F,x1)
    Fx2 = np.dot(F, x2)
    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    numerator = (np.diag(np.dot(x1.T, np.dot(F, x2)))) ** 2
    return (numerator/denom)



def sed_core(F,x1,x2):
    Fx1 = np.dot(F, x1)
    Fx2 = np.dot(F.T, x2)
    if any((Fx1[0] ** 2 + Fx1[1] ** 2) == 0) or any((Fx2[0] ** 2 + Fx2[1] ** 2) == 0):
        return np.inf
    else:
        denom1 = 1.0 / (Fx1[0] ** 2 + Fx1[1] ** 2)
        denom2 = 1.0 / (Fx2[0] ** 2 + Fx2[1] ** 2)
    numerator = (np.diag(np.dot(x2.T, np.dot(F, x1)))) ** 2
    return (np.sqrt(numerator * denom1) + np.sqrt(numerator * denom2)) / 2
    # note: this is according to zisserman - it is the sum of squared of the distances to the lines
    #return numerator*(denom1+denom2)



def ransac_F_Fast(data, num_iterations, inlier_threshold, line_pts):
    random_samples = [random.sample(list(np.arange(len(data))), k=5) for _ in range(int(num_iterations * len(line_pts)))]
    data_samples   = [data[x, :] for x in random_samples]
    line_samples   = line_pts*num_iterations
    model_samples  = [compute_F(*list(param)) for param in zip(data_samples,line_samples)]
    my_fast_sed    = symmetric_epipolar_distance_Sample(data.T)
    model_errs     = [my_fast_sed(model_s) for model_s in model_samples]
    model_inliers  = [np.sum(model_err<inlier_threshold) for model_err in model_errs]
    model_medians  = [np.median(model_err) for model_err in model_errs]
    best_idx_lmeds = np.argmin(model_medians)
    best_F_lmeds   = compute_F(data[model_errs[best_idx_lmeds]<inlier_threshold])
    best_idx_ransac= np.argmax(model_inliers)
    best_F_ransac  = compute_F(data[model_errs[best_idx_ransac]<inlier_threshold])
    best_F_ransac  =  best_F_ransac if np.shape(best_F_ransac)==(3,3) else np.ones((3,3))
    best_F_lmeds   = best_F_lmeds if  np.shape(best_F_lmeds) ==(3, 3) else np.ones((3,3))
    selected_line_ix=best_idx_ransac%len(line_pts)
    return best_F_ransac, best_F_lmeds,selected_line_ix



def compute_F(data,line_pts=None):
    x1 = data[:,0:3]
    x2 = data[:,3:6]
    if line_pts is not None:
        x1a = line_pts[:,0:3]
        x2a = line_pts[:,3:6]
        x1=np.concatenate((x1a,x1),axis=0)
        x2=np.concatenate((x2a,x2),axis=0)
    F8, _ = cv2.findFundamentalMat(x1, x2, cv2.FM_8POINT)
    if F8 is None:
        F8 = np.ones((3, 3))
    if F8[2, 2] != 0:
        F8=F8 / F8[2, 2]
    return F8





def h_coordinates(pts):
    return np.vstack((pts, np.ones((1, pts.shape[1]))))


def sep_F_ransac_lines(top_inliers_lines, pts1, pts2, inlier_ratio, inlier_threshold):
    max_num_iterations = np.floor(np.log(0.01) / np.log(1 - inlier_ratio ** 5) + 1).astype(np.uint)
    x1n = h_coordinates(np.transpose(pts1))
    x2n = h_coordinates(np.transpose(pts2))
    data = np.vstack((x1n, x2n)).T
    line_a=[]
    for top_line in top_inliers_lines:
        # Selected  points on the line
        line_x1n=h_coordinates(np.transpose(top_line['selected_line_points1']))
        line_x2n=h_coordinates(np.transpose(top_line['selected_line_points2']))
        line = np.vstack((line_x1n, line_x2n)).T
        line_a.append(line)
    Fr, Fl,line_ix   = ransac_F_Fast(data, max_num_iterations, inlier_threshold, line_a)
    return Fr,Fl,line_ix





# select three point with the maximum distance
def select_max_dist(matching_points1):
    X = np.array(matching_points1)
    sq_dist = scipy.spatial.distance.pdist(X)
    pairwise_dists = scipy.spatial.distance.squareform(sq_dist)
    two_points=np.where(pairwise_dists==np.max(pairwise_dists))
    max_dist=pairwise_dists[two_points[0][0],two_points[0][1]]

    third_point_idx=np.abs([pairwise_dists[two_points[1][0],:]-pairwise_dists[two_points[0][0],:]]).argmin()
    min_dist= np.min((pairwise_dists[two_points[0][0],third_point_idx], pairwise_dists[two_points[0][1],third_point_idx]))

    #return the selected points , their maximum distance, and their minimum distance
    return np.array([two_points[0][0],two_points[1][0],third_point_idx]),max_dist,min_dist


def normalize_coordinates(pts):
    return pts / (pts[:, -1].reshape((-1, 1))+1e-10)#avoid divid by zero

def homography_err(data,model):
    pts_src =data[:,0:2]
    pts_dest=data[:,2:4]
    pts_dest_H=np.dot(model,pts_src.T)
    try:
        pts_src_H = np.dot(np.linalg.inv(model), pts_dest.T)
        return np.sqrt(np.sum((pts_src   -normalize_coordinates(pts_src_H.T) *pts_src[:,-1].reshape((-1, 1))) **2+
                              (pts_dest  -normalize_coordinates(pts_dest_H.T)*pts_dest[:,-1].reshape((-1, 1)))**2,axis=1))
    except:
        return np.inf

# Find 1D homography
def line_homography(data):
    """
     input : kx4, k is at least 3
     output: The 2x2 (1d) line homography between the points
    """
    x1=np.transpose(data[:,0:2])
    x2=np.transpose(data[:,2:4])
    num_points=np.shape(x1)[1]
    A = np.zeros((num_points, 4))
    for n  in range(num_points):
        A[n,:]=np.concatenate(([x2[1, n] * x1[:, n], -x2[0,n]*x1[:,n]]))
    U, S, V = np.linalg.svd(A)
    H = V[3].reshape((2, 2))
    return H



def ransac_get_line_inliers(n_iters,line1_pts,line2_pts,inlier_th=0.35):
    data           = np.concatenate((line1_pts,line2_pts),axis=1)
    random_samples = [random.sample(list(np.arange(len(data))), k=3) for _ in range(n_iters)]
    data_samples   = [data[x, :] for x in random_samples]
    model_samples  = [line_homography(x) for x in data_samples]
    model_errs     = [homography_err(data,model_s) for model_s in model_samples]
    model_inliers  = [np.sum(model_err<inlier_th) for model_err in model_errs]
    best_idx_ransac= np.argmax(model_inliers)
    inliers_idx    = np.arange(len(data))[model_errs[best_idx_ransac]<inlier_th]
    return inliers_idx,np.mean(model_errs[best_idx_ransac][inliers_idx])


# return the point which is the nearest point to p3 on the line p1,p2
def project_point_line(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2-x1, y2-y1
    d = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/d
    return x1+a*dx, y1+a*dy



def symmetric_epipolar_distance_Sample(data):
    def sed(F):
        x1=data[0:3,:]
        x2=data[3:6,:]
        return sed_core(F,x1,x2)
    return sed




def set_shared_index_single(pts_lines_mtx):
    def set_index_single(index_match,index_line):
        pts_lines_mtx[index_match, index_line] = 1
    return set_index_single

def set_shared_index_both(pts_lines_mtx):
    def set_index_both(index_match,index_line1,index_line2):
        pts_lines_mtx[index_match, index_line1,index_line2] = 2
    return set_index_both



#sorted_lines,lines_shared_points
def get_top_matching_lines( pts1,pts2,
                            lines_info_img1,
                            lines_info_img2,
                            min_shared_points,
                            inlier_ratio):

# The parrallel option
#     t0=timer()
#     pts_lines         = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))
#     set_ix_1=set_shared_index_1(pts_lines )
#     [set_ix_1(x['matching_index'],x['line_index']) for x in lines_info_img1]
#     set_ix_2=set_shared_index_2(pts_lines)
#     [set_ix_2(x['matching_index'][0],x['line_index']) for x in lines_info_img2 if len(x['matching_index'][0])>0]
#     pts_lines[pts_lines < 2] = 0
#     hough_pts = np.sum(pts_lines, axis=0)
#     print("time is "+str(timer()-t0))


    # t0=timer()
    # pts_line1         = np.zeros((len(pts1), len(lines_info_img1)))
    # pts_line2         = np.zeros((len(pts1), len(lines_info_img2)))
    # set_ix_a=set_shared_index_a(pts_line1)
    # [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img1]
    # set_ix_a=set_shared_index_a(pts_line2)
    # [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img2]
    # print("THE time is " + str(timer() - t0))
    # pts_lines = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))
    # a, b = np.where(pts_line1 > 0)
    # pts_lines[a, b, :] = 1
    # a, b = np.where(pts_line2 > 0)
    # pts_lines[a, :, b] += 1
    # print("THE time is " + str(timer() - t0))
    # # First remove all entries which does not have two matching lines
    # pts_lines[pts_lines < 2] = 0
    # # Sum across points' index, this gives us how many shared points for each pair of lines
    # hough_pts = np.sum(pts_lines, axis=0)
    # # print("the time is 1" + str(timer() - t0))

    # The fastest option
#     t0=timer()
    pts_line1a         = np.zeros((len(pts1), len(lines_info_img1)))
    pts_line2b         = np.zeros((len(pts1), len(lines_info_img2)))
    set_ix_a=set_shared_index_single(pts_line1a)
    [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img1]
    set_ix_a=set_shared_index_single(pts_line2b)
    [set_ix_a(x['matching_index'],x['line_index']) for x in lines_info_img2]
    a1, b1 = np.where(pts_line1a > 0)
    a2, b2 = np.where(pts_line2b > 0)
    pts_lines = np.zeros((len(pts1), len(lines_info_img1), len(lines_info_img2)))
    set_index_both=set_shared_index_both(pts_lines)
    [set_index_both(x[0], b1[np.where(a1==x[0])], x[1]) for x in zip(a2,b2) if x[0] in a1]
    [set_index_both(x[0], x[1], b2[np.where(a2==x[0])]) for x in zip(a1,b1) if x[0] in a2]
    hough_pts = sum(pts_lines)
    # print("the time is 0 " + str(timer() - t0))


    # convert to a list where each entry is 1x3: the number of shared points for each pair of line and their indices
    n_lines1,n_lines2=np.shape(hough_pts)
    r, c      = np.indices((n_lines1, n_lines2))
    dim_lines = n_lines1*n_lines2

    # use voting to find out which lines shares points
    num_shared_points_vote= np.concatenate((
        np.concatenate(
            (hough_pts[r.reshape(dim_lines,1),c.reshape(dim_lines,1)], r.reshape(dim_lines,1)),axis=1),
            c.reshape(dim_lines,1)),axis=1)



    # sort the entries, but delete all non-relevent before to save space and time
    # all entries are at least two (one for each left and one for right line) so we multiply it by two
    mask = num_shared_points_vote<min_shared_points*2
    w = np.where(mask)
    num_shared_points_vote_a = np.delete(num_shared_points_vote, w[:len(num_shared_points_vote)], axis=0)
    num_shared_points_vote = np.delete(num_shared_points_vote, (w), axis=0)
    sorted_lines          = sorted(np.array(num_shared_points_vote), key=lambda X: X[0], reverse=True)



    # For each matching points on the matching lines,
    # project the shared points to be exactly on the line
    # start with the lines that shared the highest number of points, so we can do top-N
    # return a list index by the lines (k,j) with the projected points themself
    num_line_ransac_iterations =int((np.log(0.01) / np.log(1 - inlier_ratio ** 3)))+1
    top_lines = []
    # Go over the top lines with the most number of shared points, project the points, store by the matching indices of the pair of lines
    num_sorted_lines=np.min((len(sorted_lines), 50))
    for n in range(num_sorted_lines):
        k=int(sorted_lines[n][1])
        j=int(sorted_lines[n][2])

        # some points might be too near each other
        arr_idx = np.intersect1d(lines_info_img1[k]['matching_index'][0], lines_info_img2[j]['matching_index'][0])

        (x1a,y1a,x2a,y2a)=lines_info_img1[k]['edge_points']
        (x1b,y1b,x2b,y2b)=lines_info_img2[j]['edge_points']
        matching_points1=[]
        matching_points2=[]

        arr_shared_pts=np.concatenate((pts1[arr_idx,:],pts2[arr_idx,:]),axis=1)
        # project the points
        for pt in arr_shared_pts:

            nearest_pt_on_line1=project_point_line(np.array((x1a, y1a)).astype(float), np.array((x2a, y2a)).astype(float), np.array((pt[0], pt[1])).astype(float))
            nearest_pt_on_line2=project_point_line(np.array((x1b, y1b)).astype(float), np.array((x2b, y2b)).astype(float), np.array((pt[2], pt[3])).astype(float))

            matching_points1.append(nearest_pt_on_line1)
            matching_points2.append(nearest_pt_on_line2)


        matching_points1=np.array(matching_points1)
        matching_points2=np.array(matching_points2)

        # we need at least four unique points
        _,unique_idx_1=np.unique(matching_points1.astype(int),axis=0,return_index=True)
        _,unique_idx_2=np.unique(matching_points2.astype(int),axis=0,return_index=True)
        unique_idx=np.intersect1d(unique_idx_1,unique_idx_2)

        if len(unique_idx)<4:
            continue

        matching_points1=matching_points1[unique_idx, :]
        matching_points2=matching_points2[unique_idx, :]

        # Find inliers, inlier_idx_homography - index of inliers of all the line points
        inlier_idx_homography,hom_average_err = ransac_get_line_inliers(int(num_line_ransac_iterations),matching_points1,matching_points2)
        if len(inlier_idx_homography)<4:
            continue

        inlier_points = np.array(matching_points1)[inlier_idx_homography]
        inlier_selected_idx, inlier_selected_max_dist, inlier_selected_min_dist = select_max_dist(inlier_points)
        inlier_selected_idx, inlier_selected_max_dist, inlier_selected_min_dist = select_max_dist(inlier_points)


        mp1 = np.array(matching_points1)[inlier_idx_homography][inlier_selected_idx]
        mp2 = np.array(matching_points2)[inlier_idx_homography][inlier_selected_idx]

        # should also take the ones with the most distant points
        top_lines.append(
            {'num_inliers'  :len(inlier_idx_homography),
             'line_points_1':np.array(matching_points1)[inlier_idx_homography],
             'line_points_2':np.array(matching_points2)[inlier_idx_homography],
             'line1_index':k,
             'line2_index':j,
             'inlier_selected_index':inlier_selected_idx,
             'selected_line_points1':mp1,
             'selected_line_points2':mp2,
             'max_dist':inlier_selected_max_dist,
             'min_dist':inlier_selected_min_dist,
             'homg_err':hom_average_err})




    if len(top_lines)<2:
        return [],[],[]
    top_lines.sort(key=lambda X: X['min_dist'],reverse=True) # max_dist
    top_two_lines=[None,None]
    top_two_lines[0]=top_lines[0]
    a=lines_info_img1[top_lines[0]['line1_index']]['line_eq_abc_norm'][0]
    b=lines_info_img1[top_lines[0]['line1_index']]['line_eq_abc_norm'][1]
    line_eqs= [lines_info_img1[x['line1_index']]['line_eq_abc_norm'] for x in top_lines[1:20]]
    all_angles=[ np.arccos(np.min((a*x[0]+b*x[1],1)))*180 / np.pi for x in line_eqs]
    max_angle_ix=np.argmax(np.min(np.array((np.array(all_angles),180-np.array(all_angles))),axis=0))
    top_two_lines[1] = top_lines[max_angle_ix]
    return top_two_lines,lines_info_img1,lines_info_img2

def find_intersection_points(rho,theta,im_size):
    a = np.cos(theta)
    b = np.sin(theta)
    if a != 0:
        x_0=rho/a # y=0
        x_1 = (rho - (b * im_size[1])) / a  # y= max image size
    else:
        x_0=x_1=-1000

    if b != 0:
        y_0=rho/b # x=0
        y_1=(rho-(a*im_size[0]))/b # x= max image size
    else:
        y_0=y_1=-1000

    p0=[x_0,0] if (x_0>=0 and  x_0<=im_size[0]) else []
    p1=[0,y_0] if (y_0>=0 and  y_0<=im_size[1]) else []
    p2=[x_1,im_size[1]] if (x_1>=0  and x_1<=im_size[0]) else []
    p3=[im_size[0],y_1] if (y_1>=0 and  y_1<=im_size[1]) else []

    return []+p0+p1+p2+p3




def line_info(pts,points_intersection,max_distance,line_index):
    x1, y1, x2, y2 = points_intersection
    line_eq = np.cross([x1, y1, 1], [x2, y2, 1])
    if (abs(line_eq[2])) < np.finfo(float).eps :  # too small to divide, solve with least square
        points = [(x1, y1), (x2, y2)]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        # y= ax +c
        a, c = np.linalg.lstsq(A, y_coords)[0]
        # c should be zero
        c = 0
        b = -1
        line_eq=np.array([a,b,c])
    else:
        line_eq = line_eq / line_eq[2]
        (a, b, c) = line_eq

    d = abs((a * pts[:, 0] + b * pts[:, 1] + c)) / (math.sqrt(a * a + b * b))
    return {'matching_index': np.where(d < max_distance),
            'line_eq_abc' : line_eq,
            'line_eq_abc_norm': line_eq/np.sqrt((line_eq[0] ** 2 + line_eq[1] ** 2)+1e-17), # normalize coor. for angles between lines
            'edge_points' : (x1, y1, x2, y2),
            'max_distance': max_distance,
            'line_index'  : line_index}


def get_hough_lines(img_size,pts1,pts2,min_hough_points,pixel_res,theta_res,max_distance,num_matching_pts_to_use):
    bw_img1 = np.zeros(img_size).astype(np.uint8)
    pts1r = np.round(pts1[0:num_matching_pts_to_use, :]).astype(int)
    bw_img1[pts1r[:, 1], pts1r[:, 0]] = 255
    bw_img2 = np.zeros(img_size).astype(np.uint8)
    pts2r = np.round(pts2[0:num_matching_pts_to_use, :]).astype(int)
    bw_img2[pts2r[:, 1], pts2r[:, 0]] = 255


    lines1 = cv2.HoughLines(bw_img1, pixel_res, np.pi / theta_res, min_hough_points)
    lines2 = cv2.HoughLines(bw_img2, pixel_res, np.pi / theta_res, min_hough_points)


    if lines1 is None or lines2 is None:
        return [],[]

    line_index=0
    D_lines_1=[]
    for l in lines1:
        rho, theta = l[0]
        p_intersect = find_intersection_points(rho,theta,img_size)
        if len(p_intersect)==4:
            D_lines_1.append(line_info(pts1,p_intersect,max_distance,line_index))
            line_index=line_index+1

    D_lines_2 = []
    line_index=0
    for l in lines2:
        rho, theta = l[0]
        p_intersect = find_intersection_points(rho,theta,img_size)
        if len(p_intersect)==4:
            D_lines_2.append(line_info(pts2,p_intersect,max_distance,line_index))
            line_index=line_index+1

    return D_lines_1,D_lines_2


