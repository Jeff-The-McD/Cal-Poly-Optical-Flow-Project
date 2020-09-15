from cv2.cv2 import OPTFLOW_USE_INITIAL_FLOW

import Optical_flow_func_Def as fd
import timeit


def main():
    dir_original = 'ORIGINAL'
    dir_opt_flow = 'OPT_FLOW'
    dir_source_video = 'TEST_VIDEOS'
    # creating an inital image for the algorithm
    out_orignal, out_opt_flow = fd.recording_setup_Windows(dir_original, dir_opt_flow)
    cam = fd.cv.VideoCapture(fd.os.path.join(dir_source_video, 'DJI_0038.MOV'))
    # cam = fd.cv.VideoCapture(0)
    ret, prev = cam.read()
    prev = fd.cv.resize(prev, (640, 480))
    prevgray = fd.cv.cvtColor(prev, fd.cv.COLOR_BGR2GRAY)

    fr_count = 0
    while ret:
        loop_start = fd.time.time()

        fr_count += 1
        print(fr_count)
        ret, img = cam.read()  # reads NEXT frame from camera (NEEDED FOR ANY METHOD)
        if ret:
            img = fd.cv.resize(img, (640, 480))

            # Method 1:Flow Amplitude w red method
            # prevgray, gray, lines, new_lines = fd.flow_amplitude_method(img, prevgray)  # using flow amplitude method
            # red_contours = fd.detect_red_circle(img)
            # new_frame = fd.draw_simple_flow_W_red(gray, lines, red_contours)  # drawing results of flow amplitude method
            # decision, info = fd.make_decision_Areas_method(new_frame)  # Grabs the info to decide to turn or not based on the area
#==========================================================================================================================================#

#           Method 2: Get Partial Flow method
#             gray = fd.cv.cvtColor(img, fd.cv.COLOR_BGR2GRAY) # converts img to gray
#             flow = fd.cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.6, 5, 15, 3, 5, 1.2, fd.cv.OPTFLOW_FARNEBACK_GAUSSIAN)
#             prevgray = gray
#             ldx, ldy = fd.get_partial_flow(gray, flow, 1 ,2) # first half
#
#             threshold = 5
#
#             if max(ldx) > threshold:
#                 print('Turn right')
#                 print('Max:')
#                 #fd.print_info(new_frame, max(ldx))
#                 fd.cv.putText(new_frame, "turn right!",
#                               (300, 50),
#                               fd.cv.FONT_HERSHEY_PLAIN,
#                               3, (0, 0, 255))
#
#             ldx, ldy = fd.get_partial_flow(gray,flow,2,2)
#
#             if min(ldx) < threshold * (-1):
#                 print('Turn left!')
#                 #fd.print_info(new_frame, min(ldx))
#                 fd.cv.putText(new_frame, "turn left!",
#                               (50, 50),
#                               fd.cv.FONT_HERSHEY_PLAIN,
#                               3, (0, 0, 255))
#
#
#             new_frame=fd.draw_flow_diff_dens(gray, flow,1,3,ldx,ldy)

#=============================================================================================================
            # method 3 get get amplitude with flow

            gray = fd.cv.cvtColor(img,fd.cv.COLOR_BGR2GRAY)
            flow = fd.cv.calcOpticalFlowFarneback(prevgray,gray,None, 0.5, 10, 20, 5, 7, 1.5, fd.cv.OPTFLOW_FARNEBACK_GAUSSIAN)
            lines = fd.get_grid_coords_w_flow(img, flow , [0,1], [0,1],16)
            new_frame = fd.draw_simple_flow(gray,lines)
            new_lines = fd.get_flow_over_threshold(lines, 'y', 3,3)
            new_frame=fd.draw_simple_flow_over_threshold(new_frame,new_lines)
            decision, info = fd.get_decision_by_flow(new_lines)
            
# =============================================================================================================
            # End Of methods

            # video set up
            out_orignal.write(img)  # recording the original video input
            out_opt_flow.write(new_frame)  # recording the opt flow output

            fd.print_info(new_frame, info)


            fd.cv.imshow("OpticalFlow", new_frame)  # displays image with flow on it, for illustration
            loop_end = fd.time.time()

            print('Frame ' + str(fr_count) + ' Execution Time:' + str((loop_end - loop_start)) + ' Seconds')

            key = fd.cv.waitKey(38)
            if key == ord('q'):
                out_opt_flow.release()
                out_orignal
                break
        else:
            break
    print('landing')

main()