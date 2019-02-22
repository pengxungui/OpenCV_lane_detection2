# OpenCV_lane_detection2
这篇文章是道路检测的升级篇，其中包含了如果计算车道的曲率半径、车道宽度等，可以说这篇文章较上一篇更具有实用价值。这篇文章中使用了摄像机标定、透视变换、滑动窗口等技术。
上一篇文章只是使用了OpenCV简单的几个函数，实现了车道检测。这篇文章是道路检测的升级篇，其中包含了如果计算车道的曲率半径、车道宽度等，可以说这篇文章较上一篇更具有实用价值。这篇文章中使用了摄像机标定、透视变换、滑动窗口等技术。
结果展示


以下为具体方法：
1.摄像机标定
摄像机标定，是实时图像处理中最先需要解决的问题，一般用棋盘法标定。openCV中提供了几个重要的函数用于实现标定。具体过程大家可以自行百度。其中，每个摄像头的都有各自标定对应关系，所以在实际使用中，需要使用自己的摄像头，拍一张棋盘的网格图，然后用OpenCV中的函数进行标定。具体标定代码如下
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

2.透视变换
透视变换，即把拍摄的道路图片变成鸟瞰图。具体代码如下
def get_M_Minv():
    src = np.float32([[(203,720),(585,460),(695,460),(1127,720)]])
    dst = np.float32([[(320,720),(320,0),(960,0),(960,720)]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    return M,Minv
（其中M即表示原始图到鸟瞰图之间的变换关系；Minv表示鸟瞰图到原始图之间的变换关系，在鸟瞰图上处理后得到相应数据，最后还是要表现在原始图上，所以需要反转关系；有变换关系图以后，直接使用函数cv2.warpPerspective（）即可得到鸟瞰图）


3.阀值处理
通过各种阀值处理，并把这个阀值判断结果融合在一起，得到想要的车道信息。代码如下
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

（以上只是部分阀值判断的函数，如果有其他需要，可以增加其他各种阀值判断，然后融合在一起）

4.通过滑动窗口，拟合车道线
通过滑动窗口，得到左右车道线对应的（x，y）坐标，并使用numpy提供的多项式拟合函数polyfit(）实现车道拟合。具体代码如下
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
（以上函数的位置变换比较多，需要逐一理解，如果不懂得地方，欢迎留言交流）
在车道寻找开始，使用了在Y方向的像素分布统计，即像素分布的峰值，极有可能表示的是车道信息。


5.画车道区域，计算曲率和车道宽度
根据车道拟合函数，得到车道对应的坐标，然后画出车道区域。第一，需要使用曲率函数，如果对曲率函数不熟悉的，自行学习；第二，曲率和车道宽度的计算，需要摄像机坐标系和世界坐标系之间的关系，也就是单摄像头测距技术。本文中只是使用了一个既定关系，并不具有普适性。代码如下
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
具体展示可以参照知乎：https://zhuanlan.zhihu.com/p/57405038
