import numpy as np
import cv2
import matplotlib.pyplot as plt

def findCorners(image,grid=(9,5)):
    object_points=[]
    image_points=[]
    object_point = np.zeros((grid[0]*grid[1],3),np.float32)
    object_point[:,:2] = np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
    ret, corners = cv2.findChessboardCorners(image,grid,None)
    if ret:
        object_points.append(object_point)
        image_points.append(corners)

    return object_points,image_points

def cal_undistort(image,objpoints,imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,image.shape[1::-1],None,None)
    dst = cv2.undistort(image,mtx,dist,None,mtx)

    return dst

def get_M_Minv():
    src = np.float32([[(203,720),(585,460),(695,460),(1127,720)]])
    dst = np.float32([[(320,720),(320,0),(960,0),(960,720)]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    return M,Minv

def abs_sobel_thresh(image,orient='x',thresh_min=0,thresh_max=255):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    return binary_output

def hls_select(image,thresh=(0,255)):
    hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    channel = hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 255

    return binary_output

def select_white(image):
    lower = np.array([170,170,170])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image,lower,upper)

    return mask

def select_yellow(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    cv2.imshow("5",hsv)
    lower = np.array([20,60,60])
    upper = np.array([38,174,250])
    mask = cv2.inRange(hsv,lower,upper)

    return mask

def luv_select(image,thresh=(0,255)):
    luv = cv2.cvtColor(image,cv2.COLOR_RGB2LUV)
    channel = luv[:,:,0]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 255

    return binary_output

def lab_select(image,thresh=(0,255)):
    lab = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    channel = lab[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 255

    return binary_output

def find_line(warped):
    histogram = np.sum(warped[warped.shape[0]//2:,:],axis = 0)
    x = [x for x in range(len(histogram))]
    y = [histogram[i] for i in range(len(histogram))]
    plt.plot(x,y)
    plt.show()
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    windows = 9
    window_height = np.int(warped.shape[0]/windows)

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(windows):
        win_y_low = warped.shape[0]-(window+1)*window_height
        win_y_high = warped.shape[0]-window*window_height
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current-margin
        win_xright_high = rightx_current+margin
##good_left_inds表示的是非0 nonzero中做&运算以后非0项的位置
        good_left_inds = ((nonzeroy >= win_y_low)&(nonzeroy < win_y_high)&
                    (nonzerox >= win_xleft_low)&(nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low)&(nonzeroy < win_y_high)&
                    (nonzerox >= win_xright_low)&(nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    return left_fit,right_fit,left_lane_inds,right_lane_inds

def draw_area(undist,warped,Minv,left_fit,right_fit):
    ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((pts_left,pts_right))
    cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))
    newwarp = cv2.warpPerspective(color_warp,Minv,(undist.shape[1],undist.shape[0]))
    result = cv2.addWeighted(undist,1,newwarp,0.4,0)
    return result

    

def calculate_curv_and_pos(warped,left_fit,right_fit):
    ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty +right_fit[2]

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(ploty*ym_per_pix,leftx*xm_per_pix,2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix,rightx*xm_per_pix,2)

    left_curverad = ((1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**1.5
                     )/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5
                     )/np.absolute(2*right_fit_cr[0])

    curvature = ((left_curverad + right_curverad) / 2)

    lane_width = np.mean(rightx - leftx)
    lane_xm_per_pix = 3.7/lane_width

    veh_pos = (( lane_width * lane_xm_per_pix)/2.)
    cen_pos = ((warped.shape[1] * lane_xm_per_pix)/2.)
    distance_from_center = cen_pos - veh_pos
    return curvature,distance_from_center


if __name__ == "__main__":
    image = cv2.imread("E:/Python/lane/data/camera1.jpg")
    image_1 = cv2.imread("E:/Python/lane/data/test2.jpg")
    objpoints,imgpoints = findCorners(image)
    dst_image = cal_undistort(image_1,objpoints,imgpoints)
    M, Minv = get_M_Minv()
    trans_image = cv2.warpPerspective(dst_image,M,image_1.shape[1::-1],flags=cv2.INTER_LINEAR )

    sobel_output = abs_sobel_thresh(image_1,'x',55,100)
    hls_output = hls_select(image_1,thresh = (180,255))
    yellow_output = select_yellow(image_1)
    white_output = select_white(image_1)

    combined = np.zeros_like(white_output)
    combined[((sobel_output == 255) | (hls_output == 255)) | (
        white_output>0) | (yellow_output>0)] = 255
    warped = cv2.warpPerspective(combined,M,image_1.shape[1::-1],flags=cv2.INTER_LINEAR)
    left_fit,right_fit,left_lane_inds,right_lane_inds = find_line(warped)
    curvature,distance_from_center = calculate_curv_and_pos(warped,left_fit,right_fit)
    result = draw_area(image_1,warped,Minv,left_fit,right_fit)

    print(curvature,distance_from_center)
    cv2.imwrite("E:/Python/lane/data/test2_result.jpg",result)
    cv2.imshow("1",result)
    cv2.imshow("5",warped)
    cv2.imshow("2",dst_image)
    cv2.imshow("4",trans_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
