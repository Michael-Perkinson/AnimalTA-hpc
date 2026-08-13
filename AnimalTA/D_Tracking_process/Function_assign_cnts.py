from AnimalTA.D_Tracking_process import security_settings_track
import csv
import time
import cv2
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from skimage.graph import route_through_array
import sys
import datetime as _dt
import queue

def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[assign {ts}] {msg}", file=sys.stderr, flush=True)


def _build_dist_table(old_positions, new_centroids, scale, threshold, sentinel):
    """Build distance table using cdist then apply threshold sentinel."""
    old_arr = np.array(old_positions, dtype=np.float64)
    new_arr = np.array(new_centroids, dtype=np.float64)
    dists = cdist(old_arr, new_arr) / float(scale)
    dists[dists >= threshold] = sentinel
    return dists.tolist()

def Treat_cnts_fixed(Vid, Arenas, start, end, prev_row, Extracted_cnts, Th_extract_cnts, To_save, portion, one_every, AD, use_Kalman=False, head_tail=False, worker_state=None):
    kalmans=[]

    all_NA = [False] * len(Arenas)  # Value = True if there is only "NA" in the first frame
    last_HD_pos = [[["NA", "NA"], ["NA", "NA"]] for a in Arenas]

    _assign_total_s = 0.0
    _assign_frames = 0
    _REPORT_EVERY = 500

    # We write the first frame:
    all_rows = [["Frame", "Time"]]

    #Individual_measurements=[]
    Measured = []
    for ID_AR in range(len(Arenas)):
        Measured.append([])
        #Individual_measurements.append([])
        for ID_Ind in range(Vid.Track[1][6][ID_AR]):
            if head_tail and Vid.Track[1][6][ID_AR] == 1:
                all_rows[0].append("X_Arena" + str(ID_AR) + "_Ind" + str(ID_Ind)+"_Head")
                all_rows[0].append("Y_Arena" + str(ID_AR) + "_Ind" + str(ID_Ind)+"_Head")
                all_rows[0].append("X_Arena" + str(ID_AR) + "_Ind" + str(ID_Ind)+"_Tail")
                all_rows[0].append("Y_Arena" + str(ID_AR) + "_Ind" + str(ID_Ind)+"_Tail")
            else:
                all_rows[0].append("X_Arena" + str(ID_AR) + "_Ind" + str(ID_Ind))
                all_rows[0].append("Y_Arena" + str(ID_AR) + "_Ind" + str(ID_Ind))
            #Individual_measurements[ID_AR].append([])
            Measured[ID_AR].append(0)

            if use_Kalman:
                kalman_fil = cv2.KalmanFilter(4, 2)
                kalman_fil.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                kalman_fil.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                kalman_fil.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.5
                kalmans.append([kalman_fil,["NA","NA"]])


    with open(To_save, 'w', newline='', encoding="utf-8") as file:
        if worker_state is not None:
            worker_state.mark_output_opened()

        writer = csv.writer(file, delimiter=";")

        while Th_extract_cnts.is_alive() or Extracted_cnts.qsize()>0:#While we are still loading images or there are some extracted images that have not been associated yet
            if security_settings_track.stop_threads:
                break

            try:
                frame, kept_cnts = Extracted_cnts.get(timeout=0.1)
            except queue.Empty:
                continue
            if worker_state is not None:
                worker_state.increment("assigned_frames")
            AD.set(frame)
            # Once all the contours are filtered, we associate them to arenas (i.e. in which arenas are they)
            new_row = []#The nest row that will be saved in the csv file
            new_row.append(int(frame/one_every))#Frame number (after frame rate modification)
            new_row.append(round((frame/one_every)/Vid.Frame_rate[1],2))#Time since the biginning of the video

            new_row_to_write = []#The nest row that will be saved in the csv file
            new_row_to_write.append(int(frame/one_every))#Frame number (after frame rate modification)
            new_row_to_write.append(round((frame/one_every)/Vid.Frame_rate[1],2))#Time since the biginning of the video

            for Are in range(len(Arenas)):
                Cnt_ID_HD="NA"
                #Positions = ["NA"]*Vid.Track[1][6][Are]
                Ar_cnts = []
                for cnt in range(len(kept_cnts)):
                    cnt_M = cv2.moments(kept_cnts[cnt])
                    if cnt_M["m00"] > 0:
                        cX = cnt_M["m10"] / cnt_M["m00"]
                        cY = cnt_M["m01"] / cnt_M["m00"]
                    else:
                        cX = cnt_M["m10"]
                        cY = cnt_M["m01"]
                    result = cv2.pointPolygonTest(Arenas[Are], (cX, cY), False)  # Is the center of the contour inside the arena?
                    if result >= 0:
                        Ar_cnts.append(
                            [kept_cnts[cnt], (cX, cY)])  # If yes, we save this contour and add it as part of the arena

                if prev_row == None and (frame == start or all_NA[Are]):  # If its is the first row or if no positions were ever known
                    if len(Ar_cnts) > 0:  # If there is at least one contour
                        if len(Ar_cnts) >= Vid.Track[1][6][Are]:
                            final_cnts = Ar_cnts[0:(Vid.Track[1][6][Are])]  # If there are enought countours, we take the biggets ones one per expected target
                            #Positions = list(range((Vid.Track[1][6][Are])))

                        else:  # Else if there are less contours than expected targets, we separate the contour into subcontours
                            array = np.vstack([cnt[0] for cnt in Ar_cnts])
                            array = array.reshape(array.shape[0], array.shape[2])
                            kmeans = KMeans(n_clusters=Vid.Track[1][6][Are], random_state=0, n_init=50).fit(array)
                            new_pos = kmeans.cluster_centers_
                            final_cnts = new_pos.tolist()
                            final_cnts = [[[], (nf[0], nf[1])] for nf in final_cnts]

                        if head_tail and Vid.Track[1][6][Are]==1:
                            Cnt_ID_HD = 0
                        final_cnts = [cnt_info[1] for cnt_info in final_cnts]  # Final list of contours
                        all_NA[Are] = False

                    else:  # If we have no information about the target's previous positions and that ther is no visible contour, we fill the row with NAs
                        final_positions = [["NA", "NA"]] * Vid.Track[1][6][Are]
                        final_cnts=final_positions.copy()
                        all_NA[Are] = True

                else:  # If we had info about target's positions
                    if prev_row != None and frame == start:  # If we are working with a portion of video (redo part of the track)
                        last_row = prev_row


                    if len(Ar_cnts) == 0:  # If there are no contours found, fill wih NAs
                        final_cnts = [["NA", "NA"]] * Vid.Track[1][6][Are]

                    else:  # Else, fill with "Not_Yet"
                        final_cnts = [["Not_yet"]] * Vid.Track[1][6][Are]
                        passed_inds = sum(Vid.Track[1][6][0:Are])
                        sentinel = (Vid.shape[0] * Vid.shape[1]) / float(Vid.Scale[0])
                        old_positions = []
                        na_rows = []
                        for ind in range(Vid.Track[1][6][Are]):
                            OldCoos = last_row[(2 + 2 * passed_inds + ind * 2):(4 + 2 * passed_inds + ind * 2)]
                            if OldCoos[0] != "NA":
                                old_positions.append((float(OldCoos[0]), float(OldCoos[1])))
                            else:
                                old_positions.append(None)
                            na_rows.append(OldCoos[0] == "NA")
                        new_centroids = [(float(c[1][0]), float(c[1][1])) for c in Ar_cnts]
                        valid_old = [p for p in old_positions if p is not None]
                        if valid_old:
                            dists_valid = cdist(np.array(valid_old), np.array(new_centroids)) / float(Vid.Scale[0])
                        table_dists = []
                        valid_idx = 0
                        for ind in range(Vid.Track[1][6][Are]):
                            if na_rows[ind]:
                                table_dists.append([0] * len(Ar_cnts))
                            else:
                                row = dists_valid[valid_idx].tolist()
                                row = [d if d < Vid.Track[1][5] else sentinel for d in row]
                                table_dists.append(row)
                                valid_idx += 1

                        _t0 = time.perf_counter()
                        row_ind, col_ind = linear_sum_assignment(table_dists)
                        _assign_total_s += time.perf_counter() - _t0
                        _assign_frames += 1
                        if _assign_frames % _REPORT_EVERY == 0:
                            _tlog("assign step avg over last {} frames: {:.3f}ms/frame".format(
                                _REPORT_EVERY, _assign_total_s / _assign_frames * 1000))

                        to_del = []
                        for ind in range(len(row_ind)):
                            if table_dists[row_ind[ind]][col_ind[ind]] < Vid.Track[1][5]:
                                final_cnts[row_ind[ind]] = Ar_cnts[col_ind[ind]][1]
                                if head_tail and Vid.Track[1][6][Are] == 1:
                                    Cnt_ID_HD = col_ind[ind]
                                #Positions[row_ind[ind]]= col_ind[ind]
                            else:
                                to_del.append(ind)  # We will remove contours in case they are associated with a target too far from them
                        row_ind = np.delete(row_ind, to_del)
                        col_ind = np.delete(col_ind, to_del)

                        need_sep = [1] * len(Ar_cnts)

                        while ["Not_yet"] in final_cnts:  # If not all targets are associated to a contour
                            missing = final_cnts.index(["Not_yet"])
                            row_ind = np.append(row_ind, missing)
                            if min(table_dists[missing]) < Vid.Track[1][5]:  # If there was at least one contour that was close enought to the last target position
                                #We add a very small value to the distance to avoid having 0 division error
                                sorted_list = [[idx, elem+Vid.Track[1][5]/10000, cv2.contourArea(Ar_cnts[idx][0])] for idx, elem in enumerate(table_dists[missing]) if elem < Vid.Track[1][5]]
                                sorted_list.sort(key=lambda x: x[2]/(x[1]**2), reverse=True)
                                need_sep[sorted_list[0][0]] += 1  # The contour closest to the last known position of the target but already associated with another target will be split in X (here we count the number of time we need to split the countour)
                                col_ind = np.append(col_ind, sorted_list[0][0])
                                final_cnts[missing] = ["Waiting_sep"]  # If we found a potential contour, we change the status of the current target position
                            else:
                                col_ind = np.append(col_ind,-1)  # If there was no contour close enough from the last known target position, the position is considered as "NA"
                                final_cnts[missing] = ["NA", "NA"]

                        if ["Waiting_sep"] in final_cnts:  # If there were some contours to be splitted
                            for Cnt in range(len(need_sep)):
                                if need_sep[Cnt] > 1:
                                    inds = [ind for ind, cnt in enumerate(col_ind) if cnt == Cnt]  # Which individuals are sassociated to this contour

                                    array = np.vstack([Ar_cnts[Cnt][0]])
                                    array = array.reshape(array.shape[0], array.shape[2])
                                    if len(array) < need_sep[Cnt]:  # If the number of px to split is smaller than the number of targets, we will consider that some targets are missing
                                        to_split = len(array)
                                    else:
                                        to_split = need_sep[Cnt]

                                    kmeans = KMeans(n_clusters=to_split, random_state=0, n_init=5).fit(array)  # This function split the contours
                                    new_pos = kmeans.cluster_centers_

                                    passed_inds = sum(Vid.Track[1][6][0:Are])
                                    corr_old = []
                                    corr_na = []
                                    for ind in inds:
                                        OldCoos = last_row[(2 + 2 * passed_inds + row_ind[ind] * 2):(4 + 2 * passed_inds + row_ind[ind] * 2)]
                                        if OldCoos[0] != "NA":
                                            corr_old.append((float(OldCoos[0]), float(OldCoos[1])))
                                            corr_na.append(False)
                                        else:
                                            corr_old.append(None)
                                            corr_na.append(True)
                                    corr_new = new_pos
                                    valid_corr_old = [p for p in corr_old if p is not None]
                                    if valid_corr_old:
                                        corr_dists = cdist(np.array(valid_corr_old), np.array(corr_new)) / float(Vid.Scale[0])
                                    table_dists_corr = []
                                    valid_corr_idx = 0
                                    for is_na in corr_na:
                                        if is_na:
                                            table_dists_corr.append([sentinel] * len(corr_new))
                                        else:
                                            table_dists_corr.append(corr_dists[valid_corr_idx].tolist())
                                            valid_corr_idx += 1
                                    row_ind2, col_ind2 = linear_sum_assignment(table_dists_corr)

                                    for i in range(len(row_ind2)):
                                        final_cnts[row_ind[inds[i]]] = [val for val in new_pos[col_ind2[i]]]

                        '''
                        for ind in range(len(final_cnts)):
                            if len(Individual_measurements[Are][ind]) < 1 or (frame - start) > (((end - start) / 1000) * (Measured[Are][ind])):
                                if Positions[ind]!="NA":
                                    surface = cv2.contourArea(Ar_cnts[Positions[ind]][0])
                                    periphery = cv2.arcLength(Ar_cnts[Positions[ind]][0], True)
                                    ((x, y), (width, height), angle) = cv2.minAreaRect(Ar_cnts[Positions[ind]][0])
                                    Individual_measurements[Are][ind].append([frame, surface, periphery, max(width, height),min(width,height)])  # Frame of measurement, Area, periphery, length, width
                                Measured[Are][ind]+=1
                        '''
                        #If some targets are not correctly asociated because split was impossible:
                        if ["Waiting_sep"] in final_cnts:
                            for cnid in range(len(final_cnts)):
                                if final_cnts[cnid]==["Waiting_sep"]:
                                    final_cnts[cnid]=["NA","NA"]



                if head_tail and Vid.Track[1][6][Are] == 1 :
                    if Cnt_ID_HD!="NA":
                        cnt=Ar_cnts[Cnt_ID_HD][0]

                        x, y, w, h = cv2.boundingRect(cnt)
                        cnt_img = np.zeros((h + 2, w + 2), dtype=np.uint8)
                        contour_shifted = cnt - [x - 1, y - 1]
                        cv2.drawContours(cnt_img, [contour_shifted], -1, (255), thickness=cv2.FILLED)
                        skeleton = skeletonize(cnt_img).astype(np.uint8)
                        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
                        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)  # Convolution to count neighbors
                        endpoints = np.argwhere(neighbors == 11)  # 11 corresponds to 1 neighbor

                        if len(endpoints) > 0:
                            if len(endpoints)>2:
                                cost_array = np.where(skeleton > 0, 1, np.inf)  # Set infinite cost for non-skeleton pixels
                                max_path_length = 0
                                for i in range(len(endpoints)):
                                    for j in range(i + 1, len(endpoints)):
                                        p1, p2 = endpoints[i], endpoints[j]
                                        indices, _ = route_through_array(cost_array, tuple(p1), tuple(p2), fully_connected=True)
                                        if len(indices) > max_path_length:
                                            max_path_length = len(indices)
                                            endpoint1, endpoint2 = p1, p2

                            elif len(endpoints)==2:
                                endpoint1, endpoint2 = endpoints[0], endpoints[1]
                            else:
                                endpoint1 = endpoints[0]
                                endpoint2 = np.argwhere(neighbors == np.max(neighbors))[0]

                            endpoint1[1] += x
                            endpoint1[0] += y

                            endpoint2[1] += x
                            endpoint2[0] += y



                        else:
                            endpoint1 = [Ar_cnts[Cnt_ID_HD][1][1],Ar_cnts[Cnt_ID_HD][1][0]]
                            endpoint2 = endpoint1


                        head_coos = (endpoint1[1], endpoint1[0])
                        tail_coos = (endpoint2[1], endpoint2[0])

                        HD_Coos=[head_coos,tail_coos]

                        if last_HD_pos[Are][0][0]=="NA":
                            final_positions_HT = HD_Coos
                            last_HD_pos[Are] = HD_Coos
                        else:
                            dists_table=cdist(last_HD_pos[Are], HD_Coos)
                            order=list(linear_sum_assignment(dists_table)[1])
                            final_positions_HT = [HD_Coos[order[0]],HD_Coos[order[1]]]
                            last_HD_pos[Are] = [HD_Coos[order[0]],HD_Coos[order[1]]]
                    else:
                        final_positions_HT = [["NA","NA"],["NA","NA"]]

                    final_positions_to_write=final_positions_HT


                else:
                    final_positions_to_write=final_cnts.copy()

                final_positions = final_cnts.copy()


                new_row = new_row + [coo for sublist in final_positions for coo in sublist]  # We keep the positions here
                new_row_to_write = new_row_to_write + [coo for sublist in final_positions_to_write for coo in sublist]  # We keep the positions here



            #LAst progress
            if use_Kalman:
                next_Pos = new_row.copy()
                for indPos in range(int((len(new_row)-2)/2)):
                    if new_row[indPos*2+2] != "NA":
                        if kalmans[indPos][1][0]=="NA":
                            kalmans[indPos][1]=[new_row[indPos*2+2], new_row[indPos*2+3]]

                        mp = np.array([[np.float32(new_row[indPos*2+2] - kalmans[indPos][1][0])], [np.float32(new_row[indPos*2 + 3] - kalmans[indPos][1][1])]])

                        kalmans[indPos][0].correct(mp)
                        pred = kalmans[indPos][0].predict()
                        next_Pos[indPos*2+2]=pred[0][0]+kalmans[indPos][1][0]
                        next_Pos[indPos * 2+3] = pred[1][0]+kalmans[indPos][1][1]


                last_row_WNA = next_Pos  # We keep the positions here
            else:
                last_row_WNA = new_row.copy()

            all_rows.append(new_row_to_write)

            # The new row is saved to be used as next last row. If individuals were lost, we keep their last known position
            if frame > start or (portion and prev_row != None):
                for val in range(len(last_row_WNA)):
                    if last_row_WNA[val] == "NA":
                        last_row_WNA[val] = last_row[val]
            last_row = last_row_WNA



            if len(all_rows) >= 300:  # Every 1000 lines, we save the data in a csv file
                for row in all_rows:
                    writer.writerow(row)
                all_rows = []



        # After we finished the tracking, we save the remaining rows in the csv file.
        if not security_settings_track.stop_threads:
            for row in all_rows:
                writer.writerow(row)
                #Vid.Morphometrics = [IM for Ar in Individual_measurements for IM in Ar]
                #Vid.Morphometrics_saved = [IM for Ar in Individual_measurements for IM in Ar]



def Treat_cnts_variable(Vid, Arenas, Main_Arenas_image, Main_Arenas_Bimage, start, end, prev_row, Extracted_cnts, Th_extract_cnts, To_save, portion, one_every, AD, specify_entrance, use_Kalman=False, head_tail=False, worker_state=None):
    delay_lost=5 #How much frames do we wait before considering a target left the entrance area if it is lost
    delay_found=3#How much time of existance do we consider for a target to be real (inside entrance area)
    all_NA = [True] * len(Arenas)  # Value = True if there is only "NA" in the first frame
    Nb_found = [0] * len(Arenas)  # Number of targets who entered and then left the arena
    security_settings_track.ID_kepts = [[] for i in Arenas]
    IDs=[[] for i in Arenas]
    last_row=[[] for i in Arenas]
    kalmans = [[] for i in Arenas]

    all_rows = [["Frame", "Time", "Arena", "Ind", "X", "Y"]]
    list_tmp_rows=[]
    tmp_rows=[]

    with open(To_save, 'w', newline='', encoding="utf-8") as file:
        if worker_state is not None:
            worker_state.mark_output_opened()
        writer = csv.writer(file, delimiter=";")

        while Th_extract_cnts.is_alive() or Extracted_cnts.qsize() > 0:  # While we are still loading images or there are some extracted images that have not been associated yet
            if security_settings_track.stop_threads:
                break
            try:
                frame, kept_cnts = Extracted_cnts.get(timeout=0.1)
            except queue.Empty:
                continue
            if worker_state is not None:
                worker_state.increment("assigned_frames")
            AD.set(frame)
            # Once all the contours are filtered, we associate them to arenas (i.e. in which arenas are they)

            if prev_row != None and frame == start:  # If we are working with a portion of video (redo part of the track)
                last_row = prev_row

            for Are in range(len(Arenas)):
                final_cnts = []
                Ar_cnts = []
                for cnt in range(len(kept_cnts)):
                    cnt_M = cv2.moments(kept_cnts[cnt])
                    if cnt_M["m00"] > 0:
                        cX = int(cnt_M["m10"] / cnt_M["m00"])
                        cY = int(cnt_M["m01"] / cnt_M["m00"])
                    else:
                        cX = int(cnt_M["m10"])
                        cY = int(cnt_M["m01"])
                    if cv2.pointPolygonTest(Arenas[Are], (cX, cY), False) >= 0:# Is the center of the contour inside the arena?
                        Ar_cnts.append([kept_cnts[cnt], (cX, cY)])  # If yes, we save this contour and consider it inside the arena

                if prev_row == None and frame == start:  # If its is the first row or if no positions were ever known
                    if len(Ar_cnts)>0:
                        final_cnts = [[cnt_info[1], Main_Arenas_Bimage[Are][cnt_info[1][1],cnt_info[1][0]]==255,0,delay_found+1, Main_Arenas_image[Are][cnt_info[1][1],cnt_info[1][0]]==255] for cnt_info in Ar_cnts]  # Final list of contours
                        all_NA[Are] = False

                elif prev_row == None and all_NA[Are]:
                    final_cnts = [[cnt_info[1], False, 0, delay_found+1, Main_Arenas_image[Are][cnt_info[1][1],cnt_info[1][0]]==255] for cnt_info in Ar_cnts if Main_Arenas_Bimage[Are][cnt_info[1][1], cnt_info[1][0]] == 255]  # Final list of contours
                    all_NA[Are] = False

                else:  # If we had info about target's positions
                    if len(Ar_cnts) == 0:  # If there are no contours found
                        for last_pos in last_row[Are]:
                            if not last_pos[1] and (last_pos[2]>=delay_lost or last_pos[3]<=delay_found):#Is the target was last seen inside the entrance area and was not lost for too many time:
                                final_cnts.append(["Out",False,0,0,0])#we consider it left
                            else:
                                final_cnts.append([["NA", "NA"],last_pos[1],last_pos[2]+1,last_pos[3], False])#If it was last time seen inside the main arena, we consider it is still inside but just missing

                    else:  # Else, fill with "Not_Yet", which means we need to associate the targets with new positions, second element is None as we don't know wether point is inside main or entrance areas
                        final_cnts = [[["Not_yet"],None,lr[2],lr[3], False] for lr in last_row[Are]]
                        var_sentinel = (Vid.shape[0] * Vid.shape[1]) / float(Vid.Scale[0])
                        table_dists = []  # We make a table that will cross all distances between last known position and current targets' positions
                        inds = [i[0] for i in last_row[Are] if i[0] != "Out"]
                        if len(inds) > 0:
                            var_new_centroids = np.array([(float(c[1][0]), float(c[1][1])) for c in Ar_cnts])
                            var_valid_old = [(float(ind[0]), float(ind[1])) for ind in inds if ind[0] != "NA"]
                            var_na_mask = [ind[0] == "NA" for ind in inds]
                            if var_valid_old:
                                var_dists = cdist(np.array(var_valid_old), var_new_centroids) / float(Vid.Scale[0])
                            var_valid_idx = 0
                            for is_na in var_na_mask:
                                if is_na:
                                    table_dists.append([0] * len(Ar_cnts))
                                else:
                                    row = var_dists[var_valid_idx].tolist()
                                    row = [d if d < Vid.Track[1][5] else var_sentinel for d in row]
                                    table_dists.append(row)
                                    var_valid_idx += 1

                            row_ind, col_ind = linear_sum_assignment(table_dists)

                            to_del = []
                            for ind in range(len(row_ind)):
                                if table_dists[row_ind[ind]][col_ind[ind]] < Vid.Track[1][5]:#If the distance between the points is lower than the threshold fixed by user
                                    final_cnts[row_ind[ind]][0] = Ar_cnts[col_ind[ind]][1]#We save the position in its correct place
                                    final_cnts[row_ind[ind]][1] = Main_Arenas_Bimage[Are][Ar_cnts[col_ind[ind]][1][1],Ar_cnts[col_ind[ind]][1][0]] == 255#We also check if this point is inside the main arena or entrance area
                                    final_cnts[row_ind[ind]][2] = 0
                                    final_cnts[row_ind[ind]][3] += 1
                                    final_cnts[row_ind[ind]][4] = Main_Arenas_image[Are][Ar_cnts[col_ind[ind]][1][1],Ar_cnts[col_ind[ind]][1][0]] == 255
                                else:
                                    to_del.append(ind)  # We will remove contours in case they are associated with a target too far from them
                            row_ind = np.delete(row_ind, to_del)
                            col_ind = np.delete(col_ind, to_del)
                        else:
                            row_ind = []
                            col_ind = []

                        need_sep = [1] * len(Ar_cnts)
                        while ["Not_yet"] in [f[0] for f in final_cnts]:  # If not all targets are associated to a contour
                            missing = [f[0] for f in final_cnts].index(["Not_yet"])
                            if not last_row[Are][missing][1] and (final_cnts[missing][2]>=delay_lost or final_cnts[missing][3]<=delay_found):  # The last time this contour was seens, was it in entrance area?
                                final_cnts[missing]= ["Out",False,0,0,0]
                                row_ind = np.append(row_ind, missing)
                                col_ind = np.append(col_ind, -1) #If it was in entrance area, we consider it is lost
                            elif min(table_dists[missing]) < Vid.Track[1][5]:  # Else, if there was at least one contour that was close enought to the last target position
                                row_ind = np.append(row_ind, missing)
                                sorted_list = [[idx, elem] for idx, elem in enumerate(table_dists[missing]) if elem < Vid.Track[1][5]]
                                sorted_list.sort(key=lambda x: x[1])
                                need_sep[sorted_list[0][0]] += 1  # The contour closest to the last known position of the target but already associated with another target will be split in X (here we count the number of time we need to split the countour)
                                col_ind = np.append(col_ind, sorted_list[0][0])
                                final_cnts[missing][0] = ["Waiting_sep"]  # If we found a potential contour, we change the status of the current target position
                            else:
                                row_ind = np.append(row_ind, missing)
                                col_ind = np.append(col_ind,-1)  # If there was no contour close enough from the last known target position, the position is considered as "NA"
                                final_cnts[missing] = [["NA", "NA"],last_row[Are][missing][1],last_row[Are][missing][2]+1,last_row[Are][missing][3], last_row[Are][missing][4]]

                        col_ind = [x for _, x in sorted(zip(row_ind, col_ind))]
                        row_ind = sorted(row_ind)

                        if ["Waiting_sep"] in [cn[0] for cn in final_cnts]:  # If there were some contours to be separated
                            for Cnt in range(len(need_sep)):
                                if need_sep[Cnt] > 1:
                                    array = np.vstack([Ar_cnts[Cnt][0]])
                                    array = array.reshape(array.shape[0], array.shape[2])

                                    if len(array)<need_sep[Cnt]:#If the number of px to split is smaller than the number of targets, we will consider that some targets are missing
                                        to_split=len(array)
                                    else:
                                        to_split=need_sep[Cnt]

                                    kmeans = KMeans(n_clusters=to_split, random_state=0, n_init=5).fit(array)  # This function split the contours
                                    new_pos = kmeans.cluster_centers_

                                    inds = [ind for ind, cnt in enumerate(col_ind) if cnt == Cnt and final_cnts[ind][0] != "Out"]  # Which individuals are associated to this contour
                                    var_corr_old = []
                                    var_corr_na = []
                                    for ind in inds:
                                        OldCoos = last_row[Are][ind]
                                        if OldCoos[0][0] != "NA":
                                            var_corr_old.append((float(OldCoos[0][0]), float(OldCoos[0][1])))
                                            var_corr_na.append(False)
                                        else:
                                            var_corr_old.append(None)
                                            var_corr_na.append(True)
                                    var_valid_corr = [p for p in var_corr_old if p is not None]
                                    if var_valid_corr:
                                        var_corr_dists = cdist(np.array(var_valid_corr), np.array(new_pos)) / float(Vid.Scale[0])
                                    table_dists_corr = []
                                    var_corr_idx = 0
                                    for is_na in var_corr_na:
                                        if is_na:
                                            table_dists_corr.append([var_sentinel] * len(new_pos))
                                        else:
                                            table_dists_corr.append(var_corr_dists[var_corr_idx].tolist())
                                            var_corr_idx += 1
                                    row_ind2, col_ind2 = linear_sum_assignment(table_dists_corr)
                                    for i in range(len(row_ind2)):
                                        if final_cnts[row_ind[inds[i]]][0] == ["Waiting_sep"]:
                                            final_cnts[row_ind[inds[i]]][2] = last_row[Are][row_ind[inds[i]]][2]+1

                                        final_cnts[row_ind[inds[i]]][0] = [int(val) for val in new_pos[col_ind2[i]]]
                                        final_cnts[row_ind[inds[i]]][1] = Main_Arenas_Bimage[Are][int(new_pos[col_ind2[i]][1]),int(new_pos[col_ind2[i]][0])]==255
                                        final_cnts[row_ind[inds[i]]][4] = Main_Arenas_image[Are][int(new_pos[col_ind2[i]][1]), int(new_pos[col_ind2[i]][0])] == 255

                        #If some targets are not correctly asociated because sptlit was impossible:
                        Missed= [idx for idx,cn in enumerate(final_cnts) if cn[0]== ["Waiting_sep"]]
                        for Miss in Missed:
                            final_cnts[Miss][0]=["NA","NA"]
                            final_cnts[Miss][1] = [False]
                            final_cnts[Miss][2] = last_row[Are][Miss][2] + 1

                        for cnt_remain in [Ar_cnts[i] for i in range(len(Ar_cnts)) if i not in col_ind]:
                            result = Main_Arenas_Bimage[Are][cnt_remain[1][1],cnt_remain[1][0]]==255  # Is the point inside the entrance area?
                            if not result:  # If a contour is not associated with any existing point and is inside the entrance area, we consider it entered the arena
                                final_cnts.append([cnt_remain[1],False,0,0,Main_Arenas_image[Are][cnt_remain[1][1],cnt_remain[1][0]]==255])


                while len(final_cnts)>len(IDs[Are]):
                    Nb_found[Are]+=1
                    IDs[Are].append(Nb_found[Are]-1)
                    last_row[Are].append([["NA","NA"],False,0,0])

                    if use_Kalman:
                        kalman_fil = cv2.KalmanFilter(4, 2)
                        kalman_fil.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                        kalman_fil.transitionMatrix = np.array(
                            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                        kalman_fil.processNoiseCov = np.array(
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.5
                        kalmans[Are].append([kalman_fil, ["NA", "NA"]])

                if not all_NA[Are]:#If at least one target was found
                    for id in reversed(range(len(final_cnts))):
                        if final_cnts[id][0]!="Out":#The individual is inside the arena
                            if specify_entrance and final_cnts[id][1]:
                                all_rows.append([(frame / one_every), round((frame / one_every) / Vid.Frame_rate[1], 2), Are,IDs[Are][id], final_cnts[id][0][0], final_cnts[id][0][1]])
                                if IDs[Are][id] not in security_settings_track.ID_kepts[Are]:
                                    security_settings_track.ID_kepts[Are].append(IDs[Are][id])
                            elif not specify_entrance and final_cnts[id][3]>delay_found and final_cnts[id][0][0]!="NA" and final_cnts[id][4]:
                                if IDs[Are][id] in list_tmp_rows:
                                    pos = list_tmp_rows.index(IDs[Are][id])
                                    for r in tmp_rows[pos]:
                                        all_rows.append(r)
                                    list_tmp_rows.pop(pos)
                                    tmp_rows.pop(pos)
                                all_rows.append([(frame / one_every), round((frame / one_every) / Vid.Frame_rate[1], 2), Are, IDs[Are][id], final_cnts[id][0][0], final_cnts[id][0][1]])

                                if IDs[Are][id] not in security_settings_track.ID_kepts[Are]:
                                    security_settings_track.ID_kepts[Are].append(IDs[Are][id])

                            elif not specify_entrance and final_cnts[id][4] and (final_cnts[id][3]<=delay_found or final_cnts[id][0][0]=="NA"):
                                if IDs[Are][id] not in list_tmp_rows:
                                    list_tmp_rows.append(IDs[Are][id])
                                    tmp_rows.append([[(frame / one_every), round((frame / one_every) / Vid.Frame_rate[1], 2),Are, IDs[Are][id], final_cnts[id][0][0], final_cnts[id][0][1]]])
                                else:
                                    pos=list_tmp_rows.index(IDs[Are][id])
                                    tmp_rows[pos].append([(frame / one_every), round((frame / one_every) / Vid.Frame_rate[1], 2), Are, IDs[Are][id], final_cnts[id][0][0], final_cnts[id][0][1]])

                            if final_cnts[id][0][0]!="NA":#The coordinate is not lost
                                last_row[Are][id] = final_cnts[id]
                                if use_Kalman:
                                    if kalmans[Are][id][1][0] == "NA":
                                        kalmans[Are][id][1] = [final_cnts[id][0][0], final_cnts[id][0][1]]

                                    mp = np.array([[np.float32(final_cnts[id][0][0] - kalmans[Are][id][1][0])],
                                                   [np.float32(final_cnts[id][0][1] - kalmans[Are][id][1][1])]])

                                    kalmans[Are][id][0].correct(mp)
                                    pred = kalmans[Are][id][0].predict()
                                    last_row[Are][id][0] = (pred[0][0] + kalmans[Are][id][1][0],pred[1][0] + kalmans[Are][id][1][1])

                            else:
                                last_row[Are][id][2]+=1

                        elif final_cnts[id][0]=="Out":#the individual left the arena
                            if IDs[Are][id] in list_tmp_rows:
                                pos = list_tmp_rows.index(IDs[Are][id])
                                list_tmp_rows.pop(pos)
                                tmp_rows.pop(pos)
                            IDs[Are].pop(id)
                            last_row[Are].pop(id)
                            if use_Kalman:
                                kalmans[Are].pop(id)




            if len(all_rows) >= 300:  # Every 300 lines, we save the data in a csv file
                for row in all_rows:
                    writer.writerow(row)
                all_rows = []

        # After we finished the tracking, we save the remaining rows in the csv file.
        if not security_settings_track.stop_threads:
            for row in all_rows:
                writer.writerow(row)

