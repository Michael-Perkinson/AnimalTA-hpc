import math
import random
from tkinter import *
from tkinter import ttk
from AnimalTA.A_General_tools import Diverse_functions, UserMessages, User_help, Class_Lecteur, Function_draw_arenas, Class_loading_Frame, Color_settings
from AnimalTA.D_Tracking_process import Treat_simgle_image
from  AnimalTA.E_Post_tracking.b_Analyses import Functions_Analyses_Speed
import decord
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
import scipy.interpolate as interp



def Compute_silhouette(Vid, Ind, Ind_coos, after_track=True):
    Copy_Coos=Ind_coos.copy()
    Area=Vid.Identities[Ind][0]
    #We take 100 of random contours to create the silhouette:
    sample_size = 100

    #We first do a very small subsample to have an idea of the border width we want:
    possible_frames = np.where(Copy_Coos[:, 0] > -1000)[0].tolist()
    frames=random.sample(possible_frames, min(len(possible_frames), sample_size))
    all_cnts=np.array(get_cnts(Vid, frames, Copy_Coos))

    if len(all_cnts)>0:
        # Calculate the average of the first elements
        average_area = np.mean(all_cnts[:, 0])
        sd_area= np.std(all_cnts[:, 0])
        average_perimeter = np.mean(all_cnts[:, 1])
        sd_perimeter = np.std(all_cnts[:, 1])
        average_width = np.mean(all_cnts[:, 2])
        sd_width = np.std(all_cnts[:, 2])
        average_height = np.mean(all_cnts[:, 3])
        sd_height = np.std(all_cnts[:, 3])
        average_skel_len = np.nanmean(all_cnts[:, 4])
        sd_skel_len = np.nanstd(all_cnts[:, 4])

        N=len(all_cnts[:, 0])
        Nskel=np.sum(~np.isnan(all_cnts[:, 4]))

        return([average_area,sd_area,average_perimeter,sd_perimeter,average_width,sd_width,average_height,sd_height,average_skel_len,sd_skel_len,N,Nskel])
    else:
        return (["NA","NA","NA","NA","NA","NA","NA","NA","NA","NA","NA","NA"])





def get_cnts(Vid, frames, Coos):
    frames.sort()
    Which_part = 0
    start = Vid.Cropped[1][0]  # Video beginning (after crop)
    end = Vid.Cropped[1][1]  # Video end (after crop)

    if Vid.Cropped[0]:
        if len(Vid.Fusion) > 1:  # If the video results from concatenated videos
            Which_part = [index for index, Fu_inf in enumerate(Vid.Fusion) if Fu_inf[0] <= start][-1]

    #!!!!!!!!!!!!!!!! adapt to immg sequence!!!!!!!!!!!!!!
    capture = decord.VideoReader(Vid.Fusion[Which_part][1])  # Open video
    capture.seek(0)
    Prem_image_to_show = capture[start - Vid.Fusion[Which_part][0]].asnumpy()  # Take the first image
    del capture

    if Vid.Track[1][6][0]:
        type="fixed"
    else:
        type="variable"

    if type == "fixed":
        mask, or_bright, Arenas, Prem_image_to_show = Treat_simgle_image.Prepare_Vid(Vid, Prem_image_to_show,
                                                                                     type, portion=False,
                                                                                     arena_interest=None)
    elif type == "variable":
        mask, or_bright, Arenas, Main_Arenas_image, Main_Arenas_Bimage, Prem_image_to_show = Treat_simgle_image.Prepare_Vid(Vid,Prem_image_to_show,type,portion=False,arena_interest=None)

    all_cnts=[]
    Which_part = None
    first = True
    nb_comb=0
    for frame in frames:
        new_Which_part = [index for index, Fu_inf in enumerate(Vid.Fusion) if Fu_inf[0] <= frame][-1]
        if (Which_part is None) or (new_Which_part!=Which_part):
            Which_part=new_Which_part
            cap=cv2.VideoCapture(Vid.Fusion[Which_part][1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - Vid.Fusion[Which_part][0] + Vid.Cropped[1][0])
        ret, img=cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if Vid.Rotation == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif Vid.Rotation == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif Vid.Rotation == 3:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if Vid.Cropped_sp[0]:
            img = img[Vid.Cropped_sp[1][0]:Vid.Cropped_sp[1][2],
                    Vid.Cropped_sp[1][1]:Vid.Cropped_sp[1][3]]


        cnts, cnts_full, or_img = Treat_simgle_image.Image_modif(Vid, img, Prem_image_to_show, mask, or_bright, approx=False)
        if len(cnts)>0:
            #We associate the point with the closest contour:
            Center = Coos[frame, :]
            all_dists=[]
            for cnt in cnts:
                dist=cv2.pointPolygonTest(cnt, Center, True)
                if dist>0:
                    dist=0
                else:
                    dist=-dist

                all_dists.append(dist)

            cnt_ID = all_dists.index(min(all_dists))


            if (all_dists.index(min(all_dists))/ Vid.Scale[0]) < Vid.Track[1][5]:
                #Then we calculate the contour characteristics
                cnt=cnts[cnt_ID]
                cnt_full=cnts_full[cnt_ID]


                Area = cv2.contourArea(cnt) * (1 / float(Vid.Scale[0])) ** 2
                Perimeter = cv2.arcLength(cnt, True) / Vid.Scale[0]
                ((x, y), (width, height), angle) = cv2.minAreaRect(cnt)
                Width=max([width,height])
                Height = min([width, height])

                #We extract the length of the skeleton:
                x, y, w, h = cv2.boundingRect(cnt)
                cnt_img = np.zeros((h + 2, w + 2), dtype=np.uint8)
                contour_shifted = cnt - [x - 1, y - 1]
                cv2.drawContours(cnt_img, [contour_shifted], -1, (255), thickness=cv2.FILLED)
                skeleton = skeletonize(cnt_img).astype(np.uint8)
                kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
                neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)  # Convolution to count neighbors
                endpoints = np.argwhere(neighbors == 11)  # 11 corresponds to 1 neighbor
                saved_indices=None
                if len(endpoints) > 1:
                    cost_array = np.where(skeleton > 0, 1, np.inf)  # Set infinite cost for non-skeleton pixels
                    max_path_length = 0
                    for i in range(len(endpoints)):
                        for j in range(i + 1, len(endpoints)):
                            p1, p2 = endpoints[i], endpoints[j]
                            indices, _ = route_through_array(cost_array, tuple(p1), tuple(p2), fully_connected=True)
                            if len(indices) > max_path_length:
                                max_path_length = len(indices)
                                saved_indices=indices


                    indices=saved_indices
                    skel_length=0
                    cnt_img2 = np.zeros((h + 2, w + 2), dtype=np.uint8)

                    contour_shifted2 = cnt_full - [x - 1, y - 1]
                    cv2.drawContours(cnt_img2, [contour_shifted2], -1, (255), thickness=1)
                    cnt_img2 = cv2.cvtColor(cnt_img2, cv2.COLOR_GRAY2BGR)


                    all_dists = []
                    all_pos = []
                    all_Xs = []

                    points = contour_shifted2.reshape(-1, 2)
                    x = points[:, 0]
                    y = points[:, 1]

                    for i in range(0, len(indices)+1):
                        # Get current and previous points
                        if i == 0 or i == len(indices):
                            #We first look for the very first point
                            if i ==0:
                                first_pt=np.array(indices[0])
                                next_pt=np.array(indices[min(len(indices)-1,5)])
                            else:
                                first_pt=np.array(indices[len(indices)-4])
                                next_pt=np.array(indices[len(indices)-1])

                            slope = (first_pt[0] - next_pt[0]) / (first_pt[1] - next_pt[1])
                            intercept= next_pt[0] - slope*next_pt[1]


                            cv2.circle(cnt_img2, [int(indices[0][1]),int(indices[0][0])], 3, (50, 100, 250))
                            cv2.circle(cnt_img2, [int(indices[1][1]), int(indices[1][0])], 3, (50, 100, 250))
                            for pt in contour_shifted2:
                                cv2.circle(cnt_img2, [pt[0][0],pt[0][1]], 1, (50, 100, 50))

                            try:
                                cv2.line(cnt_img2,[0,int(intercept)],[w+2,int(slope*(w+2)+intercept)],(50,100,250))
                            except:
                                pass

                            if math.isinf(slope):
                                distances = np.abs(x - next_pt[1])
                            else:
                                distances = np.abs(y - (slope * x + intercept))
                                # Find points with distances close to zero

                            line_points_idx = np.where(distances < 20)[0]  # Adjust the tolerance as needed

                            found_points=points[line_points_idx]
                            xf = found_points[:, 0]
                            yf = found_points[:, 1]
                            distances = np.sqrt(np.power(first_pt[1]-xf,2)+np.power(first_pt[0]-yf,2))

                            if len(distances)>0:
                                x_to_add=np.min(distances)

                                if i ==0:
                                    all_Xs = [0]
                                    all_dists = [0]
                                    all_pos=["left"]
                                    skel_length=x_to_add
                                else:
                                    all_Xs = all_Xs + [skel_length + x_to_add]
                                    all_dists = all_dists + [0]
                                    all_pos = all_pos + ["left"]

                        else:
                            prev_point = np.array(indices[i - 1])
                            curr_point = np.array(indices[i])

                            pprev_point=np.array(indices[max(i - 4,0)])
                            after_point = np.array(indices[min(i + 4,len(indices)-1)])

                            slopeold=(pprev_point[0]-after_point[0])/(pprev_point[1]-after_point[1])

                            slope=(-1 / slopeold)
                            intercept=(curr_point[0] + (1 / slopeold)*curr_point[1])

                            # Calculate the distance of each point from the line
                            distances = np.abs(y - (slope * x + intercept))
                            # Find points with distances close to zero
                            line_points_idx = np.where(distances < 5)[0]  # Adjust the tolerance as needed
                            found_points=points[line_points_idx]

                            cross_product = (after_point[1] - pprev_point[1]) * (found_points[:, 1] - pprev_point[0]) - (after_point[0] - pprev_point[0]) * (found_points[:, 0] - pprev_point[1])
                            # Determine the position of points relative to the line
                            positions = np.where(cross_product > 0, 'left', 'right')

                            xf = found_points[:, 0]
                            yf = found_points[:, 1]
                            distances = np.sqrt(np.power(curr_point[1]-xf,2)+np.power(curr_point[0]-yf,2))


                            if 'left' in positions:
                                left_mask = positions == 'left'
                                left_dist = np.min(distances[left_mask])
                                all_dists = all_dists+[left_dist]
                                all_pos=all_pos+["left"]
                                all_Xs = all_Xs + [skel_length + x_to_add]

                            if 'right' in positions:
                                right_mask = positions == 'right'
                                right_dist = np.min(distances[right_mask])
                                all_dists = all_dists + [right_dist]
                                all_pos = all_pos + ["right"]
                                all_Xs=all_Xs+[skel_length + x_to_add]

                            try:
                                cv2.line(cnt_img2,[0,int(intercept)],[w+2,int(slope*(w+2)+intercept)],(200,0,50))
                            except:
                                pass
                            cnt_img2[found_points[:, 1], found_points[:, 0]] = (150, 0, 200)
                            skel_length += math.sqrt(math.pow((prev_point[0] - curr_point[0]), 2) + math.pow((prev_point[1] - curr_point[1]),
                                                                                        2))

                    sil_width = max(all_dists)*2


                    #
                    # all_dists=np.array(all_dists)
                    # all_Xs = np.array(all_Xs)
                    # all_pos = np.array(all_pos)
                    #
                    # left_distances = -all_dists[np.where(all_pos == "left")[0]]
                    # right_distances = all_dists[np.where(all_pos == "right")[0]]
                    #
                    #
                    # AllXs_left=all_Xs[np.where(all_pos == "left")[0]]
                    # AllXs_right = all_Xs[np.where(all_pos == "right")[0]][::-1]
                    #
                    # # Combine the Y coordinates
                    # all_Ys = np.concatenate((left_distances, right_distances[::-1]))
                    # all_Xs = np.concatenate((AllXs_left, AllXs_right))
                    #
                    # new_contour = np.array([[[x, y]] for x, y in zip(all_Xs, all_Ys)], dtype=np.int32)
                    #
                    # resul_img = np.zeros(( int(sil_width + 2), int(max(all_Xs) + 2)), dtype=np.uint8)
                    # resul_img=cv2.drawContours(resul_img,[new_contour],-1,255,-1)
                    #
                    # new_contour = resample_contour(new_contour, 100)
                    #
                    # if first:
                    #     first = False
                    #     contour_combined = new_contour
                    #     contour_comb_len=int(max(all_Xs))
                    #
                    # cur_cnt_len=int(max(all_Xs))
                    #
                    # width = np.max(new_contour[:, 0])
                    # swapped= new_contour.copy()
                    # swapped[:, 0] = width - swapped[:, 0]
                    #
                    # all_effects=[]
                    # all_changes=[]
                    # ncnt = new_contour.reshape(-1, 2)
                    # ncnt_swapped = swapped.reshape(-1, 2)  # Convert to Nx2 array
                    # cmbncnt = contour_combined.reshape(-1, 2)  # Convert to Nx2 array
                    #
                    # for swap in [0,1]:
                    #     if swap:
                    #         test_cnt=ncnt_swapped.copy()
                    #     else:
                    #         test_cnt=ncnt.copy()
                    #
                    #     for X in range(-max(int(contour_comb_len/2),int(cur_cnt_len/2)),max(int(contour_comb_len/2),int(cur_cnt_len/2))):
                    #         test_cnt_t=test_cnt+np.array([X, 0])
                    #         distances=(np.linalg.norm(test_cnt_t[:, np.newaxis, :] - cmbncnt[np.newaxis, :, :], axis=2))
                    #         filtered_sum = np.sum(distances)
                    #         all_effects.append(filtered_sum)
                    #         all_changes.append([swap,X])
                    #
                    # translation=all_effects.index(min(all_effects))
                    #
                    # if all_changes[translation][0]==1:
                    #     new_contour=swapped
                    #
                    # aligned_contour=new_contour + np.array([all_changes[translation][1], 0])
                    # aligned_contour = reorder_to_leftmost(aligned_contour)
                    #
                    # # Ensure clockwise order
                    # if not is_clockwise(aligned_contour):
                    #     aligned_contour = aligned_contour[::-1]  # Reverse order
                    #
                    # contour_to_show_comb = contour_combined.copy()
                    # contour_to_show_curr = aligned_contour.copy()
                    #
                    # resul_img = np.zeros(( int(sil_width + 2), int(max(all_Xs) + 2)), dtype=np.uint8)
                    # resul_img = cv2.drawContours(resul_img, [contour_to_show_curr], -1, 150, -1)
                    # resul_img=cv2.drawContours(resul_img,[contour_to_show_comb],-1,255,1)
                    # #
                    # cv2.imshow("Etire", cv2.resize(resul_img, [resul_img.shape[1] * 5, resul_img.shape[0] * 5]))
                    # cv2.waitKey()
                    #
                    # contour_combined=np.round(((contour_combined*nb_comb)+aligned_contour)/(nb_comb+1)).astype(int)
                    #
                    # nb_comb += 1

                    skel_length=skel_length / Vid.Scale[0]

                else:
                    skel_length=np.nan

                all_cnts.append([Area,Perimeter,Width,Height, skel_length])


    '''
    x, y, w, h = cv2.boundingRect(contour_combined)
    canvas = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour_combined - [x, y]
    cv2.drawContours(canvas, [shifted_contour], -1, 255, -1)

    cv2.imshow("Contour combined", cv2.resize(canvas, [canvas.shape[1] * 5, canvas.shape[0] * 5]))
    cv2.waitKey()
    '''

    return (all_cnts)



def calculate_y_coordinate_distances(contour1, contour2):
    sum_distances = 0
    # Create a dictionary to store y coordinates for each x coordinate

    ###Not working
    contour2_dict = {pt[0]: [pt[1], ] for pt in contour2[:, 0, :]}

    for pt in contour1[:, 0, :]:
        x = pt[0]
        y1 = pt[1]
        if x in contour2_dict:
            y2 = contour2_dict[x]
            y_distance = abs(y1 - y2)
            sum_distances+=y_distance

    return sum_distances

    # Example usage:
    # contour1 = np.array([[[0, 1]], [[1, 2]], [[2, 3]]])
    # contour2 = np.array([[[0, 4]], [[1, 5]], [[2, 2]]])
    # distances = calculate_y_coordinate_distances(contour1, contour2)
    # print(distances)


def resample_contour(contour, num_points):
    # Reshape the contour to Nx2 (flattening)
    contour = contour.reshape(-1, 2)

    # Generate a linear space for the current contour indices
    indices = np.linspace(0, 1, len(contour))

    # Interpolate each coordinate axis (x and y) separately
    interpolator_x = interp.interp1d(indices, contour[:, 0], kind='linear', fill_value="extrapolate")
    interpolator_y = interp.interp1d(indices, contour[:, 1], kind='linear', fill_value="extrapolate")

    # Generate new indices for the resampled contour
    new_indices = np.linspace(0, 1, num_points)

    # Resample the contour by interpolating both x and y coordinates
    resampled_contour = np.column_stack((interpolator_x(new_indices), interpolator_y(new_indices)))
    resampled_contour=resampled_contour.astype(int)

 # Ensure the contour starts from the leftmost point
    resampled_contour = reorder_to_leftmost(resampled_contour)

    # Ensure clockwise order
    if not is_clockwise(resampled_contour):
        resampled_contour = resampled_contour[::-1]  # Reverse order

    return resampled_contour

def reorder_to_leftmost(contour):
    """Reorders contour points so that it starts from the leftmost point."""
    min_index = np.argmin(contour[:, 0])  # Find the index of the leftmost point
    return np.roll(contour, -min_index, axis=0)  # Shift points to start from the leftmost

def is_clockwise(contour):
    """Checks if a contour is oriented clockwise using the signed area method."""
    # Compute the signed area (shoelace formula)
    x = contour[:, 0]
    y = contour[:, 1]
    area = np.sum((x[:-1] * y[1:]) - (x[1:] * y[:-1]))
    return area < 0  # Negative area means clockwise

