'''
  
  C. Mitelut; github: "catubc"; mitelutco@gmail.com

'''

import time, os
import numpy as np
from multiprocessing import shared_memory
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, RadioButtons

plt.ion()

#################################################
############### PLOTTING CLASS ##################
#################################################
#
class PlotROIs():
    ''' Class that visualizes the BMI ROI readouts as traces
        Input: shared memory

    '''

    #
    def __init__(self,
                 calibration_flag,
                 fname_rois_pixels_and_thresholds,
                 shmem_rois_traces_ensemble1,
                 shmem_rois_traces_ensemble2,
                 shmem_n_ttl,
                 rois_traces_raw_ensemble1_shape,
                 rois_traces_raw_ensemble2_shape,
                 shmem_reward_times,
                 shmem_tone_state,
                 shmem_live_frame,
                 shmem_ensemble_state,
                 bmi_high_threshold,
                 shmem_termination_flag,
                 shmem_live_video_frame,
                 shmem_high_threshold_state,
                 video_width,
                 video_length,
                 shmem_motion_correction_flag,
                 motion_flag,
                 shmem_dynamic_f0_flag,
                 shmem_manual_motion_correction_array,
                 shmem_contingency_degradation,
    ):

        #
        self.shmem_contingency_degradation = shmem_contingency_degradation

        #
        self.shmem_rois_traces_ensemble1 = shmem_rois_traces_ensemble1
        self.shmem_rois_traces_ensemble2 = shmem_rois_traces_ensemble2

        #
        self.rois_traces_raw_shape_ensemble1 = rois_traces_raw_ensemble1_shape
        self.rois_traces_raw_shape_ensemble2 = rois_traces_raw_ensemble2_shape

        #
        self.shmem_motion_correction_flag = shmem_motion_correction_flag

        #
        self.shmem_manual_motion_correction_array = shmem_manual_motion_correction_array

        #
        self.shmem_dynamic_f0_flag = shmem_dynamic_f0_flag

        #
        self.motion_flag = motion_flag

        #
        self.calibration_flag = calibration_flag

        # video image
        self.video_width = video_width
        self.video_length = video_length
        
        #
        self.video_show_downscale_factor = 5

        #
        self.shmem_live_video_frame = shmem_live_video_frame

        #
        self.shmem_high_threshold_state = shmem_high_threshold_state

        #
        self.fname_rois_pixels_and_thresholds = fname_rois_pixels_and_thresholds

        #
        self.sampleRate_2P = 30
        print ("...assuming sampling rate is ", self.sampleRate_2P, "hz")

        #
        self.plotting_window_width = 30

        # No. of frames to use for the live image averaging
        self.live_image_average_n_frames = 5

        # How many frames must go by before updating
        # TODO: not clear why we need 2 of these variables
        self.live_image_update_n_frames = 5

        #
        self.live_image_vmin = 1
        self.live_image_vmax = 8000

        #
        self.show_contours_on_image = False

        #
        self.bmi_high_threshold = bmi_high_threshold

        #
        self.live_image_counter = 0

        #
        self.verbose2 = False

        #
        self.shmem_termination_flag = shmem_termination_flag

        #
        self.shmem_live_frame = shmem_live_frame

        #
        self.shmem_ensemble_state = shmem_ensemble_state

        #
        self.shmem_n_ttl = shmem_n_ttl

        #
        self.shmem_reward_times = shmem_reward_times

        #
        self.shmem_tone_state = shmem_tone_state

        #
        if True: # self.calibration_flag==False:

            #
            self.initialize_rois_traces()

            #
            self.initialize_ROIs_contours()

            #
            self.initialize_ensemble_state()

            #
            self.initialize_ensemble_state_array()

            #
            self.initialize_contingency_degradation()

            #
            self.initialize_tone_state()

            #
            self.initialize_live_image_array()

            #
            self.initialize_motion_correction_flag()

            #
            self.initialize_day0_ca_mask()

        #
        self.initialize_live_frame_shared_memory()

        #
        self.initalize_n_ttl()

        #
        self.initalize_reward_times()

        #
        self.initialize_termination_flag()

        #
        self.initialize_camera_frame_shared_memory()

        #
        self.initialize_dynamic_f0_flag()

        #
        self.initialize_manual_motion_correction_array()

        #
        self.initialize_high_threshold()

        #
        self.initialize_plots()

        #
        self.ctr = 0

        # enter plot update condition
        # optional use sleep to slow down plotting
        while True:
            self.update_plots()

            if self.termination_flag[0]:
                print ("... EXITING PLOTTING CLASS ...")
                break

            # optional decrease plotting speed, may help in some cases
            # time.sleep(self.plotter_sleep_time_between_updates)

    def initialize_day0_ca_mask(self):

        dataset_root = os.path.split(self.fname_rois_pixels_and_thresholds)[0]
        fname = os.path.join(dataset_root, 'day0', "day0_ca_mask.npz")
        if not os.path.exists(fname):
            fname = os.path.join(
                os.path.split(dataset_root)[0], 'day0', "day0_ca_mask.npz"
            )
        self.mask_flag = 0

        if os.path.exists(fname):
            d = np.load(fname, allow_pickle=True)
            self.ca_mask_contours = d['mask_contours']
        else:
            self.ca_mask_contours = []

    #
    def initialize_camera_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,self.video_width,self.video_length), dtype=np.uint8)

        # get the rois_traces from the shared memory name
        self.existing_shm_video_frame = shared_memory.SharedMemory(name=self.shmem_live_video_frame)

        #
        self.live_video_frame = np.ndarray(aa.shape,
                                           dtype=aa.dtype,
                                           buffer=self.existing_shm_video_frame.buf
                                           )

        #self.video_frame[:] = aa[:]

    #
    def initialize_manual_motion_correction_array(self):

        #
        aa = np.zeros((2), dtype=np.int32)

        # get the rois_traces from the shared memory name
        self.existing_shmem_manual_motion_correction_array = shared_memory.SharedMemory(name=self.shmem_manual_motion_correction_array)

        #
        self.manual_motion_correction_array = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.existing_shmem_manual_motion_correction_array.buf)

    #
    def initialize_high_threshold(self):

        #
        aa = np.zeros(1, dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shmem_high_threshold_state = shared_memory.SharedMemory(name=self.shmem_high_threshold_state)

        #
        self.high_threshold = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shmem_high_threshold_state.buf)

        #
        self.high_threshold_initial = self.high_threshold.copy()

        #
        #self.ensemble_state_counter = 0


    #
    def initialize_contingency_degradation(self):

        #
        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_contingency_degradation = shared_memory.SharedMemory(name=self.shmem_contingency_degradation)

        #
        self.contingency_degradation = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_contingency_degradation.buf)

        #
        self.contingency_degradation[0] = 0


    #
    def initialize_ensemble_state(self):

        #
        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_ensemble_state = shared_memory.SharedMemory(name=self.shmem_ensemble_state)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_ensemble_state.buf)

        #
        self.ensemble_state_counter = 0

    #
    def initialize_motion_correction_flag(self):

        #
        aa = np.zeros((1,), dtype=np.float32)

        #
        self.existing_motion_correction_flag = shared_memory.SharedMemory(name=self.shmem_motion_correction_flag)

        #
        self.motion_corection_flag = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.existing_motion_correction_flag.buf)

        #
        self.motion_corection_flag[0] = self.motion_flag

    #
    def initialize_dynamic_f0_flag(self):

        #
        aa = np.zeros((1,), dtype=np.float32)



        #
        self.existing_dynamic_f0_flag = shared_memory.SharedMemory(name=self.shmem_dynamic_f0_flag)

        #
        self.dynamic_f0_flag = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.existing_dynamic_f0_flag.buf)

        #
        self.dynamic_f0_flag[0] = 1

    #
    def initialize_ROIs_contours(self):

        #####################################################
        # load individual cell ROIs as saved by the calibration step
        # TODO: generalize some of this code to allow different #s of cells; - not a priority
        data = np.load(self.fname_rois_pixels_and_thresholds,
                       allow_pickle=True)

        # LOAD ensemble 1 contours
        contours_local = data['ensemble1_contours']
        self.rois_contours_ensemble1 = []
        for k in range(len(contours_local)):
            self.rois_contours_ensemble1.append(contours_local[k][0])

        # LOAD ensemble 2 contours
        contours_local2 = data['ensemble2_contours']
        self.rois_contours_ensemble2 = []
        for k in range(len(contours_local2)):
            self.rois_contours_ensemble2.append(contours_local2[k][0])
        #
        self.contours_all_cells = data['contours_all_cells']

    #
    def initialize_live_image_array(self):

        # save the images as they come in so we can just plot averages not individual frames
        self.live_image_array = np.zeros((30,512,512), dtype=np.uint16)

    #
    def initialize_termination_flag(self):

        #
        aa = np.zeros(1, dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_termination_flag = shared_memory.SharedMemory(name=self.shmem_termination_flag)

        #
        self.termination_flag = np.ndarray(aa.shape,
                                              dtype=aa.dtype,
                                              buffer=self.existing_shm_termination_flag.buf)


    #
    def initalize_reward_times(self):

        #
        #print ("  n_rewards memory name : ", self.shmem_reward_times)

        aa = np.zeros((2,1000), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_reward_times = shared_memory.SharedMemory(name=self.shmem_reward_times)
        #print ("existing shm: ", self.existing_shm_reward_times)

        #
        self.reward_times = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.existing_shm_reward_times.buf)

    #
    def initialize_tone_state(self):
        #
        #print("  ensemble state memory name : ", self.shmem_tone_state)

        aa = np.zeros((1,), dtype=np.float32)

        # get the rois_traces from the shared memory name
        self.existing_shm_tone_state = shared_memory.SharedMemory(name=self.shmem_tone_state)
        #print("existing shm: ", self.existing_shm_tone_state)

        #
        self.tone_state = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_tone_state.buf)

        #
        #print("  TONE state: ", self.tone_state)


    #
    def initialize_live_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)

        # get the rois_traces from the shared memory name
        self.existing_shm_live_frame = shared_memory.SharedMemory(name=self.shmem_live_frame)

        #
        self.live_frame = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_live_frame.buf)

    #
    def initalize_n_ttl(self):

        #
        #print ("  nttl memory name : ", self.shmem_n_ttl)

        aa = np.zeros((1,), dtype=np.int64)

        # get the rois_traces from the shared memory name
        self.existing_shm_n_ttl = shared_memory.SharedMemory(name=self.shmem_n_ttl)
        #existing_shm = shared_memory.SharedMemory(name='testname')

        #print ("existing shm: ", self.existing_shm_n_ttl)


        #
        self.n_ttl2 = np.ndarray(aa.shape,
                                 dtype=aa.dtype,
                                 buffer=self.existing_shm_n_ttl.buf)

        #
        #print ("  loaded n_ttl: ", self.n_ttl2)

        #
        self.n_ttl_last = self.n_ttl2[0].copy()

       #
    #
    def initialize_ensemble_state_array(self):
        
        #
        #self.ensemble_state_array = np.zeros(self.rois_traces.shape[1]+100)  # add a small bufffer in case things don't shut down immediately
        self.ensemble_state_array = np.zeros(self.rois_traces_ensemble1.shape[1]+100) + np.nan # add a small bufffer in case things don't shut down immediately

    #
    def initialize_rois_traces(self):

        #######################################################################
        ######################### LOAD ENSEMBLE 1 #############################
        #######################################################################
        # get the rois_traces from the shared memory name
        self.existing_shm_rois_traces_ensemble1 = shared_memory.SharedMemory(name=self.shmem_rois_traces_ensemble1)

        #
        self.rois_traces_ensemble1 = np.ndarray((self.rois_traces_raw_shape_ensemble1[0],
                                                 self.rois_traces_raw_shape_ensemble1[1]),
                                                 dtype=np.float32,
                                                 buffer=self.existing_shm_rois_traces_ensemble1.buf)

        # also load the F0 computed from the calibration session
        data = np.load(self.fname_rois_pixels_and_thresholds)
        self.cell_f0s_ensemble1 = data['ensemble1_f0s']

        #######################################################################
        ######################### LOAD ENSEMBLE 2 #############################
        #######################################################################
        # get the rois_traces from the shared memory name
        self.existing_shm_rois_traces_ensemble2 = shared_memory.SharedMemory(name=self.shmem_rois_traces_ensemble2)

        #
        self.rois_traces_ensemble2 = np.ndarray((self.rois_traces_raw_shape_ensemble2[0],
                                                 self.rois_traces_raw_shape_ensemble2[1]),
                                                 dtype=np.float32,
                                                 buffer=self.existing_shm_rois_traces_ensemble2.buf)

        # also load the F0 computed from the calibration session
        self.cell_f0s_ensemble2 = data['ensemble2_f0s']

    def show_ca_mask(self):

        if self.mask_flag:
            color = 'pink'
            alpha=.8
        else:
            color = 'black'
            alpha=.8

        # add other cell contours to the data
        ids = np.arange(len(self.ca_mask_contours))
        for c in ids:
            temp = self.ca_mask_contours[c]

            self.ax_image.plot(temp[:,0],
                     temp[:,1],
                     c=color,
                     linewidth=1,
                     alpha=alpha)

            #
        self.fig.canvas.flush_events()

        #
        self.fig.canvas.draw()

        #
        self.fig.canvas.flush_events()

        #
        # self.fig.clf()

        # cache the background
        self.axbackground = []
        if True:
            self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_traces.bbox))
            self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_image.bbox))

        self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_camera.bbox))

        #
        plt.show(block=False)

    def show_other_cell_contours(self):

        #
        self.fig.canvas.restore_region(self.fig.canvas.copy_from_bbox(self.ax_image.bbox))

        # clear previous plotting
        self.ax_image.cla()

        #
        #self.fig.canvas.flush_events()

        #
        self.image_obj = self.ax_image.imshow(self.live_frame[0],
                                              vmin=self.live_image_vmin,
                                              vmax=self.live_image_vmax,
                                              interpolation="none")
        self.image_obj.set_clim([self.smin.val,
                                 self.smax.val])

        ################################################
        ############## PLOT ROI CONTOURS ###############
        ################################################
        # ROI contours ensemble 1
        for c in range(len(self.rois_contours_ensemble1)):
            for k in range(len(self.rois_contours_ensemble1[c]) - 1):
                self.ax_image.plot([self.rois_contours_ensemble1[c][k][0],
                                    self.rois_contours_ensemble1[c][k + 1][0]],
                                   [self.rois_contours_ensemble1[c][k][1],
                                    self.rois_contours_ensemble1[c][k + 1][1]],
                                   c='blue',
                                   linewidth=4)

        # ROI contours ensemble 2
        for c in range(len(self.rois_contours_ensemble2)):
            for k in range(len(self.rois_contours_ensemble2[c]) - 1):
                self.ax_image.plot([self.rois_contours_ensemble2[c][k][0],
                                    self.rois_contours_ensemble2[c][k + 1][0]],
                                   [self.rois_contours_ensemble2[c][k][1],
                                    self.rois_contours_ensemble2[c][k + 1][1]],
                                   c='red',
                                   linewidth=4)
        #

        if self.contour_flag==0:
            ids = np.arange(0,min(100,len(self.contours_all_cells)),1)
        else:
            ids = np.random.choice(np.arange(len(self.contours_all_cells)), min(50,len(self.contours_all_cells)), replace=False)
        print ("random cell ids: ", ids)
        for c in ids:

            temp = self.contours_all_cells[c][0]

            for k in range(len(temp) - 1):
                # print (k, temp, temp.shape)
                self.ax_image.plot([temp[k][0], temp[k + 1][0]],
                         [temp[k][1], temp[k + 1][1]],
                         c='white',
                         linewidth=1)

        #
        # self.fig.canvas.restore_region(self.fig.canvas.copy_from_bbox(self.ax_image.bbox))
        #
        # #
        # self.fig.canvas.flush_events()

        print ("DONE PLOTTING CONTOURS...")

        return None

        #
        self.fig.canvas.flush_events()
        #
        # #
        self.fig.canvas.draw()
        #
        # #
        # self.fig.canvas.flush_events()

    def generate_buttons(self):
        axleft = plt.axes([0.9, 0.3, 0.04, 0.04])
        axright = plt.axes([.95, 0.3, 0.04, 0.04])
        axup = plt.axes([0.925, 0.35, 0.04, 0.04])
        axdown = plt.axes([0.925, 0.25, 0.04, 0.04])

        #
        def shift_fov_right(event):
            self.manual_motion_correction_array[0] += 1
            print("moving FOV right")

        def shift_fov_left(event):
            self.manual_motion_correction_array[0] -= 1
            print("moving FOV left")

        def shift_fov_up(event):
            self.manual_motion_correction_array[1] += 1
            print("moving FOV up")

        def shift_fov_down(event):
            self.manual_motion_correction_array[1] -= 1
            print("moving FOV down")

        #
        bleft = Button(axleft, 'LEFT')
        bleft.on_clicked(shift_fov_left)
        bright = Button(axright, 'RIGHT')
        bright.on_clicked(shift_fov_right)
        bup = Button(axup, 'UP')
        bup.on_clicked(shift_fov_up)
        bdown = Button(axdown, 'DOWN')
        bdown.on_clicked(shift_fov_down)

    ################################################################################################
    ##################################### INITIALIZE PLOTS #########################################
    ################################################################################################
    def initialize_plots(self):

        #
        self.plot_y_scale = 1

        #
        clrs=['blue','magenta','red','orange']

        #
        #if self.calibration_flag==True:
        self.fig = plt.figure(figsize=(8,5))
        #else:
        #    self.fig = plt.figure(figsize=(8,5))

        #
        self.grid = GridSpec(8, 12)#, left=0.55, right=0.98, hspace=0.05)
        
		#########################################################
        ################# PLOT VIDEO IMAGE ######################
        #########################################################
        # TODO: refactor this plot to another function
        #if self.calibration_flag==False:
        self.ax_camera = self.fig.add_subplot(self.grid[5:, :4])
        #else:
        #    self.ax_camera = self.fig.add_subplot(self.grid[:, :])
        
        #
        self.ax_camera.set_xticks([])
        self.ax_camera.set_yticks([])
        
        #
        self.camera_obj = self.ax_camera.imshow(self.live_video_frame[0].T[::self.video_show_downscale_factor,
																			::self.video_show_downscale_factor],
                                                vmin=0,
                                                vmax=255,
                                                aspect='auto',
                                                cmap='binary_r',
                                                interpolation="none"
                                               #
                                              )


        #########################################################
        ######################### PLOT CA IMAGE #################
        #########################################################
        if True: #self.calibration_flag==False:
            # TODO: refactor this plot to another function
            self.ax_image = self.fig.add_subplot(self.grid[:, 4:])

            #
            self.ax_image.set_xlim(0,512)
            self.ax_image.set_ylim(512,0)

            #################################################
            ############ ADD IMAGINING VIS BUTTONS ##########
            #################################################
            # add all the buttons/for visualization
            axmin = self.fig.add_axes([0.55, 0.93, 0.37, 0.03])
            axmax  = self.fig.add_axes([0.55, 0.96, 0.37, 0.03])
            n_frame_ave  = self.fig.add_axes([0.55, 0.90, 0.10, 0.03])

            # settings for image processing
            motion_correction_slider = self.fig.add_axes([0.83, 0.88, 0.10, 0.03])
            dynamic_f0_correction_slider = self.fig.add_axes([0.83, 0.90, 0.10, 0.03])

            #
            self.smin = Slider(axmin, 'Min', 0, self.live_image_vmax, valinit=self.live_image_vmin)
            self.smax = Slider(axmax, 'Max', 0, self.live_image_vmax, valinit=self.live_image_vmax)

            #
            vals_n_frames = np.arange(1,31,1)
            self.n_frame_ave = Slider(n_frame_ave, '# Frames', 1, 30, valinit=self.live_image_average_n_frames,
                                      valstep = vals_n_frames)

            #
            vals_motion = [0,1]
            self.motion_slider = Slider(motion_correction_slider, 'Motion correct', 0, 1, valinit=self.motion_corection_flag[0],
                                      valstep=vals_motion)

            #
            vals_dynamic_f0 = [0,1]
            self.dynamic_f0_slider = Slider(dynamic_f0_correction_slider, 'Dynamic F0', 0, 1,
                                            valinit=self.dynamic_f0_flag[0],
                                            valstep=vals_dynamic_f0)
            #
            def update_clim(val):
                self.image_obj.set_clim([self.smin.val,
                                         self.smax.val])

            #
            def update_n_frames_average(val):
                self.live_image_average_n_frames = int(self.n_frame_ave.val)

            #
            def update_motion_flag(val):
                self.motion_corection_flag[0] = int(self.motion_slider.val)


            #
            def update_dynamic_flag(val):
                self.dynamic_f0_flag[0] = int(self.dynamic_f0_slider.val)

            #
            self.smin.on_changed(update_clim)
            self.smax.on_changed(update_clim)
            self.n_frame_ave.on_changed(update_n_frames_average)
            self.motion_slider.on_changed(update_motion_flag)
            self.dynamic_f0_slider.on_changed(update_dynamic_flag)

            # add other cell contours to the data
            self.contour_flag = 0
            self.show_other_cell_contours()


        #########################################################
        ########## INITIALIZE THRESHOLD SLIDER ##################
        #########################################################
        #
        if True: #self.calibration_flag == False:

            #vals_n_frames = np.arange(1,31,1)
            #self.n_frame_ave = Slider(n_frame_ave, '# Frames', 1, 30, valinit=self.live_image_average_n_frames,
            #                          valstep = vals_n_frames)


            ax_threshold = plt.axes([0.925, 0.65, 0.04, 0.15])

            vals_threshold = np.arange(1, 100, 1)
            self.vals_threshold = Slider(ax_threshold, 'Threshold', 1, 100,
                                         orientation = 'vertical',
                                         valinit=100,
                                         valstep=vals_threshold)


            def threshold_slider(event):
                self.high_threshold[0] = (self.vals_threshold.val*self.high_threshold_initial/100.)

            self.vals_threshold.on_changed(threshold_slider)

        #########################################################
        ########## INITIALIZE MASK TOGGLE BUTTON ################
        #########################################################
        #
        if True: #self.calibration_flag == False:

            contour_slider  = self.fig.add_axes([0.925, 0.15, 0.04, 0.04])

            #
            def update_contour_drawing(val):
                self.contour_flag = int(self.contour_slider.val)
                self.show_other_cell_contours()



            #
            vals_contour_slider = [0,1]
            self.contour_slider = Slider(contour_slider, 'redraw', 0, 1, valinit=0,
                                        # loc= 1,
                                         valstep=vals_contour_slider)

            self.contour_slider.on_changed(update_contour_drawing)


            # def show_mask(event):
            #     print ("TOGGLE MASK")
            #     self.mask_flag= (self.mask_flag+1)%2
            #
            #     #
            #     res = self.show_other_cell_contours()
            #
            #     print ("Button exited: ", res)
            #
            #     #
            #     #return
            #     #self.show_ca_mask()
            #
            # bmask = Button(axca_mask, 'draw-contours')
            # bmask.on_clicked(show_mask)


        #########################################################
        ########## INITIALIZE STOP BUTTON #######################
        #########################################################
        #
        if True: #self.calibration_flag == False:

            axstop = plt.axes([0.925, 0.05, 0.04, 0.04])

            def stop_bmi(event):
                print ("MANUAL STOP DETECTED")
                self.termination_flag[0]=1

            bstop = Button(axstop, 'STOP')
            bstop.on_clicked(stop_bmi)

        #########################################################
        ########## INITIALIZE STOP BUTTON #######################
        #########################################################
        #
        if True: #self.calibration_flag == False:
            axcontdeg = plt.axes([0.925, 0.10, 0.04, 0.04])

            def contingency_button(event):
                self.contingency_degradation[0] = (self.contingency_degradation[0]+1)%2
                print("Switching to contingency degradation ", self.contingency_degradation[0])

            bcontdeg = Button(axcontdeg, 'CD')
            bcontdeg.on_clicked(contingency_button)


        #########################################################
        ########## INITIALIZE DRIFT BUTTONS ######################
        #########################################################
        #
        if True: #self.calibration_flag == False:
            # self.ax_camera = self.fig.add_subplot(self.grid[6:, :8])

            # NOT WORKING CURRENTLY
            # self.generate_buttons()

            def update_vertical_slider(event):
                self.manual_motion_correction_array[1] = self.vertical_slider.val

            def update_horizontal_slider(event):
                self.manual_motion_correction_array[0] = self.horizontal_slider.val

            #
            #ax_threshold = plt.axes([0.925, 0.65, 0.04, 0.35])

            ax_vertical_shift = plt.axes([0.925, 0.23, 0.04, 0.35])
            ax_horizontal_shift = plt.axes([0.500, 0.01, 0.35, 0.04])

            self.max_drift = 20
            vals_threshold = np.arange(-self.max_drift, self.max_drift, 1)
            self.vertical_slider = Slider(ax_vertical_shift, 'y-drift',
                                          -self.max_drift,
                                          self.max_drift,
                                         orientation='vertical',
                                         valinit=0,
                                         valstep=vals_threshold)
            self.horizontal_slider = Slider(ax_horizontal_shift, 'x-drift',
                                          -self.max_drift,
                                          self.max_drift,
                                         orientation='horizontal',
                                         valinit=0,
                                         valstep=vals_threshold)

            self.vertical_slider.on_changed(update_vertical_slider)
            self.horizontal_slider.on_changed(update_horizontal_slider)

        #########################################################
        ##################### PLOT ROI TRACES ###################
        #########################################################
        if True: #self.calibration_flag==False:
            # TODO: refactor this plot to another function
            self.ax_traces = self.fig.add_subplot(self.grid[:4, :4])

            #
            n_cells = 0
            n_cells+= len(self.rois_traces_ensemble1)
            n_cells+= len(self.rois_traces_ensemble2)
            self.ax_traces.set_ylim(-0.25*self.plot_y_scale, self.plot_y_scale*(n_cells*1.5)+4*self.high_threshold)
            self.ax_traces.set_xlim(-self.plotting_window_width,0)
            self.ax_traces.set_xlabel("Time (sec)")

            self.plot_times = np.arange(-self.plotting_window_width*self.sampleRate_2P,0,1)/self.sampleRate_2P

            # make a list to hold the matplotlib line objects
            self.time_course_objects_ensemble1 = []
            self.time_course_objects_ensemble2 = []
            self.time_course_objects_ensemble_sum = []
            self.f0_objects = []

            # plot traces ensemble 1
            ctr=0
            for k in range(self.rois_traces_ensemble1.shape[0]):

                #
                y_values = self.rois_traces_ensemble1[k,:self.plotting_window_width*self.sampleRate_2P]+self.plot_y_scale*k

                #
                lineobject, = self.ax_traces.plot(self.plot_times,
                                           y_values,
                                           c='blue'
                                           )
                #
                self.time_course_objects_ensemble1.append(lineobject)

                # also draw the f0 baseline of each cell!!
                f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                           [0+self.plot_y_scale*k,0+self.plot_y_scale*k],  # plot last X values depending on length of plttimes
                                           '--',
                                           c='blue'
                                           )

                #
                self.f0_objects.append(f0object)
                ctr+=1

            # plot traces ensemble 2
            for k in range(self.rois_traces_ensemble2.shape[0]):

                #
                y_values = self.rois_traces_ensemble2[k,:self.plotting_window_width*self.sampleRate_2P]+self.plot_y_scale*ctr

                #
                lineobject, = self.ax_traces.plot(self.plot_times,
                                           y_values,
                                           c='red'
                                           )
                #
                self.time_course_objects_ensemble2.append(lineobject)

                # also draw the f0 baseline of each cell!!
                f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                           [0+self.plot_y_scale*ctr,0+self.plot_y_scale*ctr],
                                           '--',
                                           c='red'
                                           )
                self.f0_objects.append(f0object)
                ctr+=1

            ########################################################
            ############ INITIALIZE ENSEMBLE STATES ################
            ########################################################
            # plot sum ensemble state
            self.ensemble_state_y_scaling = 1
            self.ensemble_state_y_offset = 2.5
            y_values = self.ensemble_state_array[:self.plotting_window_width*self.sampleRate_2P]*self.ensemble_state_y_scaling+\
                       self.plot_y_scale*(ctr)+self.ensemble_state_y_offset
            #
            lineobject, = self.ax_traces.plot(self.plot_times,
                                       y_values,
                                       linewidth = 4,
                                       c='black'
                                       )
            #
            self.time_course_objects_ensemble_sum.append(lineobject)

            # ADD F0 for ensemble states
            f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                       [0+self.plot_y_scale*(ctr)+self.ensemble_state_y_offset,
                                        0+self.plot_y_scale*(ctr)+self.ensemble_state_y_offset],
                                       '--',
                                        c='black',
                                       linewidth=2,
                                       )

            self.f0_objects.append(f0object)

            # ADD Reward level for ensemble states
            f0object, = self.ax_traces.plot([self.plot_times[0],self.plot_times[-1]],
                                       [self.plot_y_scale*(ctr)+self.ensemble_state_y_offset+self.high_threshold *self.plot_y_scale,
                                        self.plot_y_scale*(ctr)+self.ensemble_state_y_offset+self.high_threshold *self.plot_y_scale],  # plot last X values depending on length of plttimes
                                       '--',
                                        c='green',
                                       linewidth=2,
                                       )  # Return

            self.f0_objects.append(f0object)

        #
        self.fig.canvas.flush_events()

        #
        self.fig.canvas.draw()

        #
        self.fig.canvas.flush_events()

        #
        #self.fig.clf()

        # cache the background
        self.axbackground = []
        if True: #self.calibration_flag==False:
            self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_traces.bbox))
            self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_image.bbox))

        self.axbackground.append(self.fig.canvas.copy_from_bbox(self.ax_camera.bbox))

        #
        plt.show(block=False)

    #
    def update_plots(self):

        '''  Function to dynamically update our plots
            TODO: move it to a separate core/process to not interfere with the main BMI loop/analysis code

        '''

        # update ROI line plots
        self.n_ttl_current = self.n_ttl2[0].copy()
        
        # ensure we have moved over 1 ttl pulse first
        if self.n_ttl_current < (self.n_ttl_last+1):
            return

        #############################################################
        ################## UPDATE ENSEMBLE ARRAY ####################
        #############################################################
        if True: #self.calibration_flag==False:
            # save the current ensembel state
            while self.n_ttl_current > self.ensemble_state_counter:
                self.ensemble_state_array[self.ensemble_state_counter] = self.ensemble_state[0]
                self.ensemble_state_counter+=1

            self.ensemble_state_array[self.n_ttl_current] = self.ensemble_state[0]

        #
        if self.verbose2:
            start = time.time()

        # restore background for all plots
        for k in range(len(self.axbackground)):
            self.fig.canvas.restore_region(self.axbackground[k])

        ####################################
        ########### UPDATE IMAGE ###########
        ####################################
        if True: #self.calibration_flag==False:
            # TODO: note this takes 10ms of time to update,
            #     may wish to plot it only 1 - 5 x per second, no  real information here aside from nasty drift
            #start = time.time()

            # shift all the data one image over
            self.live_image_array[:-1] = self.live_image_array[1:]

            # add current image frame
            self.live_image_array[-1] = self.live_frame[0].copy()

            # dont' update thelive image frame until we have at least min # of frams
            if self.live_image_counter> self.live_image_average_n_frames:

                # reset counter
                #self.live_image_counter = 0

                # compute mean over last n_frames
                temp = np.mean(self.live_image_array[-self.live_image_average_n_frames:],axis=0)

                #
                self.image_obj.set_data(temp)

            #else:
            self.live_image_counter+=1


        ####################################
        ######## UPDATE CAMERA OUTPUT ######
        ####################################
        self.camera_obj.set_data(self.live_video_frame[0].T[::self.video_show_downscale_factor,
															::self.video_show_downscale_factor])

        #########################################
        ######### UPDATE ROI TIME COURSES #######
        #########################################
        if True: #self.calibration_flag==False:
            # make t=0 tick to the current timer in seconds
            x_ticks = np.arange(-30,0.1,5)
            x_ticks_new = np.arange(-30,0.1,5)
            x_ticks_new[-1] = round(self.n_ttl2[0]/self.sampleRate_2P,2)
            self.ax_traces.set_xticks(x_ticks, x_ticks_new)

            #
            self.n_ttl_last = self.n_ttl_current.copy()

            #
            x_values = np.arange(0, min(self.n_ttl_current, self.plotting_window_width*self.sampleRate_2P), 1) / 30 - self.plotting_window_width
            x_values1 = np.array([x_values[0], x_values[-1]])

            # plot roi time courses - ensemble1
            ctr=0
            for k in range(self.rois_traces_ensemble1.shape[0]):

                #
                y_values = self.rois_traces_ensemble1[k,
                                        max(0,self.n_ttl_current-self.plotting_window_width*self.sampleRate_2P):self.n_ttl_current]+self.plot_y_scale*k

                #
                self.time_course_objects_ensemble1[k].set_data(x_values,
                                                    y_values
                                                    )

                # redraw baseline for each cell
                # TODO: note that we want to reset to 0 not the actual f0 values
                y_values1 = np.array([self.plot_y_scale*ctr,self.plot_y_scale*ctr])

                #
                self.f0_objects[ctr].set_data(x_values1,
                                            y_values1
                                           )
                ctr+=1

            # plot roi time courses - ensemble2
            for k in range(self.rois_traces_ensemble2.shape[0]):

                #
                y_values = self.rois_traces_ensemble2[k,
                                        max(0,self.n_ttl_current-self.plotting_window_width*self.sampleRate_2P):self.n_ttl_current]+self.plot_y_scale*ctr

                #
                self.time_course_objects_ensemble2[k].set_data(x_values,
                                                    y_values
                                                    )

                # redraw baseline for each cell
                # TODO: note that we want to reset to 0 not the actual f0 values
                y_values1 = np.array([self.plot_y_scale*ctr,self.plot_y_scale*ctr])

                #
                self.f0_objects[ctr].set_data(x_values1,
                                            y_values1
                                           )
                ctr+=1

        ################################################
        ############ UPDATE ENSEMBLE STATE #############
        ################################################
        if True: #self.calibration_flag==False:
            # update ensemble state
            y_values = self.ensemble_state_array[max(0,self.n_ttl_current-self.plotting_window_width*self.sampleRate_2P):
                                                       self.n_ttl_current]*self.ensemble_state_y_scaling+self.plot_y_scale*(ctr+self.ensemble_state_y_offset)

            #
            #print (" esnemble values: ", y_values)

            #
            self.time_course_objects_ensemble_sum[0].set_data(x_values,
                                                   y_values
                                                    )
            # update F0
            y_values1 = np.array([self.plot_y_scale*(ctr+self.ensemble_state_y_offset),self.plot_y_scale*(ctr+self.ensemble_state_y_offset)])
            self.f0_objects[ctr].set_data(x_values1,
                                          y_values1
                                         )

            # update ensemble state threshold
            y_values1 = np.array([self.plot_y_scale*(ctr+self.ensemble_state_y_offset)+self.high_threshold ,
                                  self.plot_y_scale*(ctr+self.ensemble_state_y_offset)+self.high_threshold ])
            self.f0_objects[ctr+1].set_data(x_values1,
                                        y_values1
                                       )
        
        ################################################
        ################# UPDATE TITLE #################
        ################################################
        if True: # self.calibration_flag==False:
            idx1 = np.where(self.reward_times[0]>-1)[0]
            idx2 = np.where(self.reward_times[1]>-1)[0]
            if self.contingency_degradation[0]:
                self.ax_traces.set_title(" # rewards : "+str(idx1.shape[0])+" "+str(idx2.shape[0]) +
                                     ": Freq: " +str(int(self.tone_state[0]))+"hz"+
                                     "\n Ensemble state: "+str(round(self.ensemble_state[0],2)), fontsize=12,
                                         c='red')
            else:
                self.ax_traces.set_title(" # rewards : "+str(idx1.shape[0])+" "+str(idx2.shape[0]) +
                                     ": Freq: " +str(int(self.tone_state[0]))+"hz"+
                                     "\n Ensemble state: "+str(round(self.ensemble_state[0],2)), fontsize=12,
                                         c='black')

        #####################################################
        ################# REFRESH PLOTS #####################
        #####################################################
        for k in range(len(self.axbackground)):
            self.fig.canvas.restore_region(self.axbackground[k])

        # add the drawn lines to the plot
        if True: # self.calibration_flag==False:
            for k in range(len(self.rois_traces_ensemble1)):
                self.ax_traces.draw_artist(self.time_course_objects_ensemble1[k])

            for k in range(len(self.rois_traces_ensemble1)):
                self.ax_traces.draw_artist(self.time_course_objects_ensemble2[k])

            for k in range(len(self.f0_objects)):
                self.ax_traces.draw_artist(self.f0_objects[k])

        if True: #self.calibration_flag==False:
            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax_traces.bbox)

            #
            self.fig.canvas.blit(self.ax_traces.bbox)
            # self.fig.canvas.blit(self.ax_image.bbox)

        #
        self.fig.canvas.flush_events()

        #
        if self.verbose2:
            print ("time for updating graph: ", time.time()-start)
            print ('')
            print ('')
