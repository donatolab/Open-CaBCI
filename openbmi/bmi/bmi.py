'''
  
  Catalin Mitelut; github: "catubc"; mitelutco@gmail.com

'''
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    from nidaqmx.constants import TerminalConfiguration
except ImportError:
    nidaqmx = None
    AcquisitionType = None
    TerminalConfiguration = None
import tqdm
import pandas as pd
import os
import time
import numpy as np
from multiprocessing import shared_memory
from utils.utils import smooth_ca_time_series4, compute_dff0, compute_dff0_with_reference, get_mode
from drift.drift import apply_shifts
from simulation.simulation import Simulation

#################################################
################## BMI CLASS ####################
#################################################
class BMI():

    ''' BMI class
        Inputs:
            - path of Thorimage memmap file where [ca] data is to be saved
            - ROI values for ROIs to be tracked during BMI
            - path of speaker file where tones are saved
            - ...

        Outputs:
            -

        TODO: we may want to fix the names of all the shared variables
              i.e., don't use random variables, just fix them to something that doesn't change
              - this way, we don't even need to share them between the modules
        TODO:  this way we dont' even have to pass them to the other modules
    '''

    #
    def __init__(self,
                 simulation_mode_bmi,
                 simulation_flag_licking,
                 fname_root_path,
                 fname_fluorescence,
                 fname_ttl,
                 sampleRate_2P,
                 fname_roi_pixels_and_thresholds,
                 max_n_seconds_session,
                 n_frames_session,
                 video_width,
                 video_length,
                 motion_flag,
                 align_flag
                 ):

        #
        self.debug_mode = False

        #
        print ("... initializing BMI parameters...")
        print ("    TODO: consider saving all imaging data to RAM disk (or faster SSD) for improved speeds")

        if self.debug_mode:
            print (">>>>>> NOTE RUNNING IN DEBUG MODE ONLY... BMI WILL NOT RUN/GIVE REWARDS<<<<")

        #
        self.align_flag=align_flag

        #
        self.initialize_alignment_flag()

        # flag which prevents return to rewards until the ensemble state drops substantially
        self.dynamic_reward_lockout = True

        # how far must the ensemble state drop before we reset the reward states
        self.dynamic_reward_lockout_threshold = 0.5

        #
        self.motion_flag = motion_flag

        #
        self.initialize_motion_correction_variable()
        
        #
        self.video_width = video_width
        self.video_length = video_length

        #
        self.simulation_mode_bmi = simulation_mode_bmi

        # Pace simulated imaging frames against an absolute clock when requested.
        self.simulation_realtime = False


        #
        self.simulation_mode_lick_detector = simulation_flag_licking

        #
        self.fname_root_path = fname_root_path
        self.fname_fluorescence = fname_fluorescence
        self.fname_ttl = fname_ttl
        
        #
        self.fname_save_data = os.path.join(os.path.split(fname_fluorescence)[0],"results.npz")

        #
        self.fname_rois_pixels_thresholds = fname_roi_pixels_and_thresholds

        #
        self.shared_memory_variables_names_list = []

        #
        self.bmi_dictionary = []

        # NOT SURE IF REQUIRED... TO DELETE
        # TODO flag was probably used during development toskip the reading step;
        self.read_data_flag = True

        # Define variables
        self.sampleRate_NI = 1E3     # Sample rate of NI card

        #
        self.ttl_pts = 1             # number of values to read from NI card - usually we read a single value to avoid buffering issues

        #
        self.sampleRate_2P = sampleRate_2P      # Sample rate of BScope

        # TODO: externalize these parameters
        self.image_width = 512
        self.image_length = 512
		
		###########################################################
        ################# BINNING PARAMETERS ######################
		###########################################################
        # amount of time to bin [ca] data in seconds
        self.binning_flag = True

        # bining time in seconds
        self.binning_time = 0.200

        #
        self.binning_n_frames = int(self.binning_time * self.sampleRate_2P)
        print ("binning time: ", self.binning_time, " , binning_n_frames: ", self.binning_n_frames)

        #
        self.binning_count = self.binning_n_frames

        # # of binned frames to use for smoothing
        self.smoothing_n_frames = 3

        #
        self.smoothing_flag = True
		###########################################################
		###########################################################
        ###########################################################
        self.max_n_seconds_session = max_n_seconds_session

        # number of frames to run BMI for
        self.n_frames = n_frames_session # OLD WAY OF COMPUTING max_n_seconds_session*sampleRate_2P

        # TODO: why do we have 2 of these variables?
        self.n_frames_to_be_acquired = self.n_frames   # Number of frames from BScope

        #
        #self.rois_smooth_window = 15                 # Number of frames to use to smooth the ROI traces
        #                                            # to be developed/changed further

        # parameter which turns on realitime DFF0 computation only after a certain period of time
        # TODO: determine if online DFF0 is required:
        #  things to evaluate: bleaching type of slow baseline drift...
        #     but for this slow drift we can use very long windows (like 2mins or more)
        # - for faster update not sure this is correct
        self.n_ttl_to_start_applying_dynamic_f0 = 90 *self.sampleRate_2P

        # for dynamic f0 updates how often to update the f0 baseline in frames
        self.update_f0_time = 10 * self.sampleRate_2P

        # start the ttl frame counter at 0
        self.ttl_computed = 0

        # number of frames to search forward in time to see if there is any neural data saved
        #   this is for the ROI reading step
        self.n_frames_search_forward = 5

        # Keep track of each trial start time
        self.last_trial_start_ttl = 0

        # save trial bouts and times
        #                   [start n_ttl, start abs time, end n_ttl, end abs time, reward (0/1/2 <- 2 is random reward)]
        self.trials = np.zeros((1000, 5), 'float64')+np.nan

        #
        self.trial_number = 0

        #
        self.post_reward_state = [0]

        # initialize ROIs
        self.initialize_ROIs()

        # initizlie the realtime value of the ensembel states (i.e. no history)
        # TODO: may wish to hold history somewhere also
        self.ensemble_activity = np.zeros((2, self.n_frames_to_be_acquired))
        
        # this is the differences of the 2 ensemble
        self.ensemble_diff_array = np.zeros(self.n_frames_to_be_acquired)

        # initailize the realtime roi states; these hold the smooth/processed version of the realtime roi
        #self.rois_activity_realtime = np.zeros(len(self.rois_pixels),dtype=np.float32)

        # initialize all arrays to be used, mostly to save data after BMI run
        self.initialize_data_arrays()

        # initialize tone state
        self.initialize_ensemble_state()

        # initalize reward contidions based on ~15mins of pre BMI recorded data
        self.initialize_reward_conditions_and_parameters()

        #
        self.initialize_threshold_shared_memory()

        # initialize rewards counter
        self.initialize_reward_times()

        # intiatlie n_ttl
        self.initialize_n_ttl()

        # initialize tone state
        self.initialize_tone_state()

        # initialize tone state
        self.initialize_white_noise_state()

        # initialize the water reward memory variable
        self.initialize_water_reward()

        #
        self.initialize_termination_flag()

        #
        self.initialize_contingency_degradation_flag()

        #
        self.initialize_live_frame_shared_memory()

        #
        self.initialize_drift_correction()
        
        # start reading the ttl pulses from the 2p scope
        self.initialize_bscope_ttl_pulse_reader()

        # this gets read simultaneously with all other TTL/BNC channels now
        #self.initalize_lick_detector_reader()

        # keeps track of lick values
        self.lick_detector_abstime = [] #np.zeros((self.n_frames,2),dtype=np.float32)

        #
        self.initialize_rotary_encoder()
        
        #
        self.initialize_video_frame()

        #
        self.initialize_dynamic_f0_variable()

        #
        #self.initialize_dynamic_template_flag()

        #
        self.initialize_manual_motion_correction_array()
        
        #
        self.initialize_dynamic_reward_lockout_state()

    
    #
    def initialize_dynamic_reward_lockout_state(self):

        '''
            shared variable indicating whether we are in a reward-lockout state or not
            - required by tone class (possibly others)
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1), dtype=np.int32)
        self.shmem_dynamic_reward_lockout_state = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.dynamic_reward_lockout_state = np.ndarray(aa.shape,
											 dtype=aa.dtype,
											 buffer=self.shmem_dynamic_reward_lockout_state.buf)

        #
        self.dynamic_reward_lockout_state[0] = 0

        #
        # ## flag which indicates whether we are in the period post-reward that we want to lockout
        # self.dynamic_reward_lockout_state = False
        
        
    #
    def initialize_manual_motion_correction_array(self):

        '''
            Left-Right and Up-Down motion correction
            Array index 0 controls left-right shifts
            Array index 1 controls up-down shifts
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((2), dtype=np.int32)
        self.shmem_manual_motion_correction_array = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.manual_motion_correction_array = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.shmem_manual_motion_correction_array.buf)

        #
        self.manual_motion_correction_array[:] = aa[:]

        #

    #
    def initialize_video_frame(self):
        ''' shared variable that keeps current video camera frame in memeory for
        '''

        # make a numpy array to hold the rois_traces
        print ("self.video width: ", self.video_width, " length: ", self.video_length)
        aa = np.zeros((1,self.video_width,self.video_length), dtype=np.uint8)
        self.shmem_live_video_frame = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.live_video_frame = np.ndarray(aa.shape,
                                        dtype=aa.dtype,
                                        buffer=self.shmem_live_video_frame.buf)

        #
        self.live_video_frame[:] = aa[:]

    #
    def initialize_rotary_encoder(self):

        # this keeps track of the rotary encoder wheel rotations
        self.rotary_encoder1_abstime = []
        self.rotary_encoder2_abstime = []


    def initialize_alignment_flag(self):

        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_alignment_flag = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.alignment_flag = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.shmem_alignment_flag.buf)

        #
        self.alignment_flag[0] = self.align_flag

    #
    def initialize_termination_flag(self):

        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_termination_flag = shared_memory.SharedMemory(create=True,
                                                                 size=aa.nbytes)

        #
        self.termination_flag = np.ndarray(aa.shape,
                                     dtype=aa.dtype,
                                     buffer=self.shmem_termination_flag.buf)

        #
        self.termination_flag[:] = aa[:]

        #

    def initialize_contingency_degradation_flag(self):
        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_contingency_degradation = shared_memory.SharedMemory(create=True,
                                                                size=aa.nbytes)

        #
        self.contingency_degradation = np.ndarray(aa.shape,
                                          dtype=aa.dtype,
                                          buffer=self.shmem_contingency_degradation.buf)

        #
        self.contingency_degradation[0] = 0


    def initialize_dynamic_f0_variable(self):
        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_dynamic_f0_flag = shared_memory.SharedMemory(create=True,
                                                                       size=aa.nbytes)

        #
        self.dynamic_f0_flag = np.ndarray(aa.shape,
                                                 dtype=aa.dtype,
                                                 buffer=self.shmem_dynamic_f0_flag.buf)

        #
        self.dynamic_f0_flag[0] = 0

    #
    def initialize_motion_correction_variable(self):

        '''
            Signal that is shared with all cores to indicate termination of BMI
            - 0: keep running
            - 1: end all processing
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int32)
        self.shmem_motion_correction_flag = shared_memory.SharedMemory(create=True,
                                                                     size=aa.nbytes)

        #
        self.motion_correction_flag = np.ndarray(aa.shape,
                                             dtype=aa.dtype,
                                             buffer=self.shmem_motion_correction_flag.buf)

        #
        self.motion_correction_flag[0] = self.motion_flag

    #
    def initialize_water_reward(self):

        '''
            This variable keeps track of the value of the water spout
            - 0: no water reward
            - 1: water reward
            Note: the duration and timing and lockouts of water rewards are controlled by the
            waterreward class

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_water_reward = shared_memory.SharedMemory(create=True,
                                                              size=aa.nbytes)

        #
        self.water_reward = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_water_reward.buf)

        #
        self.water_reward[:] = aa[:]

    #
    def initialize_tone_state(self):

        '''
            This variable keeps track of the tone value computed by the TONE class
            - technically it doesn't have to be initialized here, but we do it for simplicity to easier
              share it with the plotter class (BMI class doesn't need it for now)

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_tone_state = shared_memory.SharedMemory(create=True,
                                                       size=aa.nbytes)

        #
        self.tone_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_tone_state.buf)

        #
        self.tone_state [:] = aa[:]

    #
    def initialize_white_noise_state(self):

        '''
            This variable keeps track of the tone value computed by the TONE class
            - technically it doesn't have to be initialized here, but we do it for simplicity to easier
              share it with the plotter class (BMI class doesn't need it for now)

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_white_noise_state = shared_memory.SharedMemory(create=True,
                                                                  size=aa.nbytes)

        #
        self.white_noise_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_white_noise_state.buf)

        #
        self.white_noise_state [:] = aa[:]

    #
    def initialize_drift_correction(self):

        ''' These 2 variables keep track of the x and y drift
            - template is computed in the calibration step
            - drift gets computed in the Drift class using phase correlation
              between the template and the live image
            - here we just initialize the variables that can be shared with rest of code
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(2, dtype=np.int32)
        self.shmem_drift_xy_values = shared_memory.SharedMemory(create=True,
                                                       size=aa.nbytes)

        #
        self.drift_xy_values = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_drift_xy_values.buf)

        #
        self.drift_xy_values[:] = aa[:]

        # a list to save all the realtime applied drift
        self.drift_array = []

    #
    def initialize_reward_conditions_and_parameters(self):

        ''' This function should load the parameters computed by another script
            Input: filenames of several parameters:
                - minimum frequency (e.g. 1000Hz)
                - maximum frequency (e.g. 180000Hz)
                - threshold_low (e.g. -1.0)
                - threshold_high (e.g. +1.0)
                ... others to add
        '''

        data = np.load(self.fname_rois_pixels_thresholds, allow_pickle=True)

        #
        self.low_threshold = data['low_threshold']
        self.high_threshold_loaded = data['high_threshold']

        #
        self.post_reward_lockout = data['post_reward_lockout']

        # NEED THIS: it is set in the calibration step;  DO NOT CHANGE IT
        self.rois_smooth_window = data['rois_smooth_window']

        #
        self.smooth_diff_function_flag = data['smooth_diff_function_flag']

        # set the last reward time in ttl pulses (might need something better here)
        self.initialize_last_reward_ttl()

        # reward lockout time after a positive reward - in seconds
        self.received_reward_lockout = 3
        print (">>>>>>>>>>>> POST-REWARD LOCKOUT: ", self.received_reward_lockout, "sec")

        # similar to post-reward lockout
        self.missed_reward_lockout = 3
        print (">>>>>>>>>>>> MISSED-REWARD LOCKOUT: ", self.missed_reward_lockout, "sec")

        # counter that track time after last reward
        self.initialize_reward_lockout_counter()

        # the amount of time the mouse has to try and receive a reward - in seconds
        self.max_reward_window = 30

        #
        #self.template = data['calibration_template']

    #
    def initialize_threshold_shared_memory(self):

        '''
            This variable keeps track of the locally computed E1-E2
            - it is shared with a different process which plays tones
            - TODO: perhaps want a better name like neural_state - to disambugate from ensembel states
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_high_threshold_state = shared_memory.SharedMemory(create=True,
                                                              size=aa.nbytes)

        #
        self.high_threshold = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_high_threshold_state.buf)

        #
        self.high_threshold[0] = self.high_threshold_loaded


    #
    def initialize_ensemble_state(self):

        '''
            This variable keeps track of the locally computed E1-E2
            - it is shared with a different process which plays tones
            - TODO: perhaps want a better name like neural_state - to disambugate from ensembel states
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1,dtype=np.float32)
        self.shmem_ensemble_state = shared_memory.SharedMemory(create=True,
                                                       size=aa.nbytes)

        #
        self.ensemble_state = np.ndarray(aa.shape,
                                         dtype=aa.dtype,
                                         buffer=self.shmem_ensemble_state.buf)

        #
        self.ensemble_state[:] = aa[:]


    #
    def initialize_pbar(self):
        self.pbar = tqdm.tqdm(total=self.n_frames_to_be_acquired,
                              desc='% complete',
                              position=0,
                              leave=True,
                              ascii=True)  # Init pbar

    #
    def initialize_data_arrays(self):
        ''' TODO: check to make sure all the possible data being recorded is being saved

        '''
        #
        self.ttl_values = []            # array to hold ttl data being read
        self.ttl_n_computed = []        # number of ttl pulses computed based on time elapsed
        self.ttl_n_detected = []        # number of ttl pulses detected based on TTL from NI board
        self.inter_ttl_time = []        # computed time between each detected TTL pluse
        self.abs_times_ttl_read = []             # Keep of every time TTL is read... important!
        self.abs_times_ca_read = []             # Keep of every time TTL is read... important!
                                        # loop;   might be useful for debugging later on kernel interuptions etc.
        self.ttl_times = []             # ttl times to be saved
        self.previous_trigger=0         # time of the previous TTL trigger to be used to determine if next trigger etc
        self.prev_max = 0               # TTL pulse previous read max value
        self.prev_min = 0               # TTL pulse previous read min value
        self.ttl_voltages = []          # ttl_voltages

        #self.initialize_n_ttl()
        self.rewarded_times_abs = []

    #
    def initialize_last_reward_ttl(self):
        ''' This variable keeps track of the last received reward or missed reward time
            - it is used to reset certain conditions
            TODO: may wish to have separate clocks for received reward vs. missed reward time.
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_last_reward_ttl = shared_memory.SharedMemory(create=True,
                                                          size=aa.nbytes)

        #
        self.last_reward_ttl = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.shmem_last_reward_ttl.buf)

        #
        self.last_reward_ttl[0] = -1

    #
    def initialize_reward_lockout_counter(self):

        '''  This value keeps track of a counter that resets every time there's a reward
            - or a missed reward to prevent rewards during the period

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_reward_lockout_counter= shared_memory.SharedMemory(create=True,
                                                          size=aa.nbytes)

        #
        self.reward_lockout_counter = np.ndarray(aa.shape,
                                    dtype=aa.dtype,
                                    buffer=self.shmem_reward_lockout_counter.buf)

        #
        self.reward_lockout_counter[:] = aa[:]

    #
    def initialize_reward_times(self):

        ''' shared variable that tracks # of rewards

        '''

        # an array to hold the reward times
        aa = np.zeros((2,1000), dtype=np.int64)-1
        self.shmem_reward_times = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.reward_times = np.ndarray(aa.shape,
                                dtype=aa.dtype,
                                buffer=self.shmem_reward_times.buf)

        #
        self.reward_times[:] = aa[:]

    #
    def initialize_live_frame_shared_memory(self):

        ''' shared variable that keeps current image in memeory for plotter to visualize
            NOTE: We actually need 2 independent ones (for now) to send to plotter
            and motion detection algorithm independently.
        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)
        self.shmem_live_frame_plotter = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.live_frame_plotter = np.ndarray(aa.shape,
                                        dtype=aa.dtype,
                                        buffer=self.shmem_live_frame_plotter.buf)

        #
        self.live_frame_plotter[:] = aa[:]

        # Also initialize a live frame for the
        # make a numpy array to hold the rois_traces
        aa = np.zeros((1,512,512), dtype=np.uint16)
        self.shmem_live_frame_motion_detector = shared_memory.SharedMemory(create=True,
                                                             size=aa.nbytes)

        #
        self.live_frame_motion_detector = np.ndarray(aa.shape,
                                        dtype=aa.dtype,
                                        buffer=self.shmem_live_frame_motion_detector.buf)

        #
        self.live_frame_motion_detector[:] = aa[:]
        #

    #
    def initialize_n_ttl(self):

        ''' This variable keeps track of how many frames the BMI has detected
            - it is used to trigger the search for the next imaging frame
            - it is also shared with the plotting algorithm

            TODO:
            - we may actually be able to run the BMI without TTL signals from the microscope
            - that is, we can just actively search (e.g. every 10ms) the raw imaging data to see if any
              new data has been written and take the latest image as proof of this
            - this "nuclear" option could be implemented in systems that are more complex to work with
              or that dont' have easily accessible TTL pulses - but are very good at writing to disk

            - NOTE: this option should probably be implemented using a RAM-drive where the imaging data
              is saved to a ram disk to avoid brekaing spindisks/SSDs

        '''

        # make a numpy array to hold the rois_traces
        aa = np.zeros(1, dtype=np.int64)
        self.shmem_n_ttl = shared_memory.SharedMemory(create=True,
                                                      size=aa.nbytes)

        #
        self.n_ttl = np.ndarray(aa.shape,
                                dtype=aa.dtype,
                                buffer=self.shmem_n_ttl.buf)
        self.n_ttl[:] = aa[:]

        #
        #print(" ttl counter initialized: ", self.n_ttl, self.shmem_n_ttl.name)

    #
    def initialize_ROIs(self):

        '''
            Initialize the ROIs and ensemble arrays to be used below

            TODO: Must properly transfer ROIs to this function not just use a box aroudn a point of interest
        '''

        # TODO: generalize some of this code to allow different #s of cells; - not a priority
        data = np.load(self.fname_rois_pixels_thresholds,
                       allow_pickle=True)

        #############################################################
        #################### LOAD ENSEMBLE 1 DATA ###################
        #############################################################
        self.roi_f0s_ensemble1 = data['ensemble1_f0s']

        # also load the ensemble footprints
        ensemble1_footprints = data['ensemble1_footprints']
        self.rois_pixels_ensemble1=[]
        for k in range(len(ensemble1_footprints)):
            self.rois_pixels_ensemble1.append(ensemble1_footprints[k].T)

        # make a default size matrix that will hold [n_rois, n_frames]
        a = np.zeros((len(self.rois_pixels_ensemble1),self.n_frames),
                      dtype=np.float32)+1E-8

        # rois traces raw: contains the raw ROIs (i.e. summed pixels etc in each ROI)
        self.rois_traces_raw_ensemble1 = np.zeros(a.shape, dtype=np.float32)

        #
        self.shmem_rois_traces_ensemble1 = shared_memory.SharedMemory(create=True,
                                                                   size=a.nbytes)

        #
        self.rois_traces_smooth_ensemble1 = np.ndarray(a.shape,
                                                      dtype=a.dtype,
                                                      buffer=self.shmem_rois_traces_ensemble1.buf)

        #
        self.rois_traces_smooth_ensemble1[:] = a[:]

        #############################################################
        #################### LOAD ENSEMBLE 2 DATA ###################
        #############################################################
        self.roi_f0s_ensemble2 = data['ensemble2_f0s']

        #
        ensemble2_footprints = data['ensemble2_footprints']
        self.rois_pixels_ensemble2 = []
        for k in range(len(ensemble2_footprints)):
            self.rois_pixels_ensemble2.append(ensemble2_footprints[k].T)

        # make a default size matrix that will hold [n_rois, n_frames]
        a = np.zeros((len(self.rois_pixels_ensemble2),self.n_frames),
                      dtype=np.float32)+1E-8

        #
        self.rois_traces_raw_ensemble2 = np.zeros(a.shape, dtype=np.float32)

        #
        self.shmem_rois_traces_ensemble2 = shared_memory.SharedMemory(create=True,
                                                                   size=a.nbytes)

        #
        self.rois_traces_smooth_ensemble2 = np.ndarray(a.shape,
                                              dtype=a.dtype,
                                              buffer=self.shmem_rois_traces_ensemble2.buf)

        #
        self.rois_traces_smooth_ensemble2[:] = a[:]


    def update_bmi_dictionary(self):

        state = {"n_ttl": self.n_ttl[0],
                 #"ensemble1": self.rois_pixels_ensemble1[self.n_ttl[0]],
                 #"ensemble2": self.rois_pixels_ensemble2[self.n_ttl[0]],
                 "current_high_threshold": self.high_threshold[0],
                 "white_noise_state": self.white_noise_state[0],
                 "post_reward_state": self.post_reward_state[0],
                 "reward_lockout_counter": self.reward_lockout_counter[0],
                 "tone_state": self.tone_state[0],
                 "ensemble_state": self.ensemble_state[0],
                 "water_reward": self.water_reward[0],
                 }
        #
        self.bmi_dictionary.append(state)


    #
    def run_BMI(self):

        #
        print('Running BMI (ctrl-c to stop)')
     
        #
        self.now = time.perf_counter() #time.perf_counter_ns()/1E9
        self.previous_trigger = time.perf_counter()-2 # set the previous tirgger 2 sec prior to start

        #
        self.initialize_pbar()

        # abssolute start time
        self.start_time_acquisition = time.time()

        # save initial trial start
        self.trials[self.trial_number,0] = self.n_ttl[0]
        self.trials[self.trial_number,1] = self.start_time_acquisition

        # start recording and acquisition
        # count number of frames; but probably safer to just count time;
        # TODO: merge ttl pulse counting and time tracking into a single while statement
        #while self.ttl_computed < self.n_frames_to_be_acquired - 1:
        # NOTE: n_ttl is bumped to the next value after loading ...
        while self.n_ttl[0] < (self.n_frames_to_be_acquired):

            # read next bscope ttl pulse
            self.read_bscope_ttl()

            # check of ttl pulse when from high ~5 to low ~0
            if self.min_<1 and self.prev_max>=1:
                
                # prefer to save it here for debugging purposes
                self.ttl_times.append(self.now)
                
                #
                #if len(self.ttl_times)>0:
                #    print (" Detected ttl: ", self.ttl_times[-1])
                
                # also have a non_bmi mode for debugging, e.g. to count TTL pulses etc.
                if self.debug_mode==False:
                    
                    # runs the bmi code whenever imaging frame is completed
                    self.bmi_update()

                # update trigger time
                self.previous_trigger = self.now

                # update bmi state dictionary
                self.update_bmi_dictionary()

                #
                self.pbar.update(n=1)

            #
            self.prev_min = self.min_
            self.prev_max = self.max_

            # exit OPTION 2 check if estimated recording time + 2mins have been completed
            if (time.time() - self.start_time_acquisition)>self.max_n_seconds_session:
                print ("Duration of BMI loop: ", time.time() - self.start_time_acquisition, 'sec',
                       "  , total requested: ", self.max_n_seconds_session)

                self.termination_flag[0]=1

            #
            if self.termination_flag[0]:
                break

        # save all data acquried during recording
        # TODO: try to save this on the fly if possible to avoid loosing data during crashes
        self.save_data()
        
        #
        self.bscope_ttl_task.stop()
        self.bscope_ttl_task.close()

        #

    #
    # def save_lick_detector_ttl(self):
    #
    #     # get current lick detector state from NI card output port
    #     #self.lick_detector_ttl_value = self.lick_detector_ttl_task.read(number_of_samples_per_channel=self.ttl_pts)
    #
    #     # this saves a triple: [lick_detector_ttl_value,
    #     #                       self.now,
    #     #                       n_ttl_counter]
    #     #print ("lick detector val: ", self.lick_detector_ttl_value," self.now: ", self.now)
    #
    #     # saves updated values
    #     self.lick_detector_value_abstime_nttl[self.n_ttl[0]] = self.lick_detector_ttl_value, self.now

    def read_ttl(self):

        try:
            read_values = self.bscope_ttl_task.read(number_of_samples_per_channel=self.ttl_pts)
            return True, read_values
        except:
            print (" >>>>>>>>>>> ERROR READING NI CARD TTL <<<<<<<<<<<<<<<<")
            return False, []

    #
    def read_bscope_ttl(self):

        # get current bscope ttl pulse value from NI card output port
        # TODO: this call periodically crashes, not clear why NI card falls behind on read statements
        ttl_flag = False
        while ttl_flag == False:
            ttl_flag, read_values = self.read_ttl()

        # ttl bscope value
        ttl_value = read_values[0]#.copy()

        # lick detector value
        # TODO: IMPORTANT to read out both encoder and lick-detector in high res rate as 30Hz may miss
        #       important information
        self.lick_detector_abstime.append(read_values[1])

        # rotary encoder
        self.rotary_encoder1_abstime.append(read_values[2])
        self.rotary_encoder2_abstime.append(read_values[3])

        #  leave these in just in case we end up reading at higher bit rates and multiple samples at a atime
        # TODO: these might be redundant, not clear
        self.min_ = np.min(ttl_value)
        self.max_ = np.max(ttl_value)
        self.ttl_voltages.append(ttl_value)
        #

        # get time of ttl pulse
        self.now_ttl_pulse = time.perf_counter() #perf_counter_ns()/1E9
        self.now = self.now_ttl_pulse

        # this helps us figure out how fast this loop runs
        # TODO: we may want to introduce a delay of 5ms or so so we don't constanly read TTL pulses
        #     but this is probably not necessary as the NIDAQMX package was made to be pinged a lot
        self.abs_times_ttl_read.append(self.now_ttl_pulse)

    #
    def close(self):
        
        #
        print (" ... closing BMI, # of ttl pulses processed: ", self.n_ttl[0])
        #
        print (" ... SENDING TERMIANTION FLAG SIGNAL TO ALL PROCESSES ...")
        self.termination_flag[0] = 1

        #
        print("... EXITING BMI CLASS...")

        # give the rest of the modules a few sec to complete
        time.sleep(2)

    #
    def initialize_bscope_ttl_pulse_reader(self):

        #
        if self.simulation_mode_bmi == True:
            self.bscope_ttl_task = Simulation(self.fname_ttl)
        else:
            if nidaqmx is None:
                raise RuntimeError(
                    "nidaqmx is required when BMI simulation mode is disabled"
                )
                          
    
            #
            self.bscope_ttl_task = nidaqmx.Task('ttl_reader')
            # set TTL pulse reader from 2p system
            # add ttl pulse channel from bscope
            self.bscope_ttl_task.ai_channels.add_ai_voltage_chan("Dev3/ai0:4",
                                                          terminal_config=TerminalConfiguration.NRSE)
                                                          
            # add lick detector channel
            #self.bscope_ttl_task.ai_channels.add_ai_voltage_chan("Dev3/ai1",
            #                                              terminal_config=TerminalConfiguration.NRSE)
            #c
            self.bscope_ttl_task.timing.cfg_samp_clk_timing(self.sampleRate_NI,
                                                     # samps_per_chan=pointsToPlot*2,
                                                     sample_mode=AcquisitionType.CONTINUOUS)

            # start the TTL reader (not required in simulation mode)
            self.bscope_ttl_task.start()

        # list to add to when reading values
        self.bscope_read_values = np.zeros(2,dtype=np.float32)

	#
    def dynamic_f0_function(self,
                            roi_history,
                            percentile_val=8
                            ):
        ''' Function that updates the baseline f0 for the cells
        '''

        # OPTION #1: not used: take the median of the
        #return np.median(roi_history, axis=0)

        # OPTION #2: use 8 percentile [citation]
#        if True:
#			from scipy import stats
#            from scipy.signal import savgol_filter
#            width = 30
#            roi_history = savgol_filter(roi_history, width, 2)
#
#           return np.percentile(roi_history, percentile_val, axis=0)

        # OPTION #3: use aggregate mode of distribution
        if True:
            mode = get_mode(roi_history)
            return mode
    
    #
    def dynamic_f0(self):

        ''' This function is supposed to counter any potential bleaching artifacts
            it takes the current ROI activity over recent history (set by parameter)
            and then subtracts
            - it recomputes f0 every 1 second (or less is also ok)
        '''

        # only change f0 values at most every 10 or much more seconds;
        # use at lesat 90 seconds history or more is even better
        if self.n_ttl[0]%self.update_f0_time==0:
            if self.n_ttl[0]>(self.n_ttl_to_start_applying_dynamic_f0):

                #
                # loop over ensembel 1
                for p in range(len(self.rois_traces_raw_ensemble1)):

                    # compute median value over the past frames
                    roi_history = self.rois_traces_raw_ensemble1[p,
                                               self.n_ttl[0] - self.n_ttl_to_start_applying_dynamic_f0:
                                               self.n_ttl[0]]

                    #
                    self.roi_f0s_ensemble1[p] = self.dynamic_f0_function(roi_history)
                    
                # loop over ensembel 2
                for p in range(len(self.rois_traces_raw_ensemble2)):

                    # compute median value over the past frames
                    roi_history = self.rois_traces_raw_ensemble2[p,
                                               self.n_ttl[0] - self.n_ttl_to_start_applying_dynamic_f0:
                                               self.n_ttl[0]]

                    #
                    self.roi_f0s_ensemble2[p] = self.dynamic_f0_function(roi_history)
            else:
                pass
                #print (" Too soon to recompute baseline... please wait 90 seconds at least from start")
    #


    def bin_rois(self):

        # check if binning clock reset
        if self.binning_count>0:
            self.binning_count-=1
            return

        ################ BINNING STEP ################
        # TODO: NOTE:# need to work from the raw data
        for p in range(2):
            ############## ENSEMBLE 1 ##########
            # bin step
            temp = self.rois_traces_raw_ensemble1[p, self.n_ttl[0]-self.binning_n_frames:self.n_ttl[0]]
            dff_temp = (temp - self.roi_f0s_ensemble1[p]) / self.roi_f0s_ensemble1[p]
            dff_now = np.mean(dff_temp)

            # extra step where we smooth current dff with previous N values
            if self.smoothing_flag:
                dff_prev_n_frames = []
                for q in range(self.smoothing_n_frames - 1, 0, -1):
                    dff_prev_n_frames.append(self.rois_traces_smooth_ensemble1[p,self.n_ttl[0] - q * self.binning_n_frames])
                dff_all = np.mean(np.hstack((dff_now, dff_prev_n_frames)))
                self.rois_traces_smooth_ensemble1[p,self.n_ttl[0]] = dff_all
            else:
                self.rois_traces_smooth_ensemble1[p,self.n_ttl[0]] = dff_now

            ############ ENSEMBLE 2 #############
            temp = self.rois_traces_raw_ensemble2[p, self.n_ttl[0] - self.binning_n_frames:self.n_ttl[0]]
            dff_temp = (temp - self.roi_f0s_ensemble2[p]) / self.roi_f0s_ensemble2[p]
            dff_now = np.mean(dff_temp)

            # extra step where we smooth current dff with previous N values
            if self.smoothing_flag:
                dff_prev_n_frames = []
                for q in range(self.smoothing_n_frames - 1, 0, -1):
                    dff_prev_n_frames.append(
                        self.rois_traces_smooth_ensemble2[p,self.n_ttl[0] - q * self.binning_n_frames])
                dff_all = np.mean(np.hstack((dff_now, dff_prev_n_frames)))
                self.rois_traces_smooth_ensemble2[p, self.n_ttl[0]] = dff_all
            else:
                self.rois_traces_smooth_ensemble2[p, self.n_ttl[0]] = dff_now

        # reset binning count to
        self.binning_count = self.binning_n_frames


    def bmi_update(self):

        #
        self.compute_frame_number()

        #
        self.load_current_frame_and_apply_drift_correction()

        # load the [ca] imaging and compute activity in each ROI
        self.update_rois()

        # smooth the ROIs using the external function
        self.compute_dff_and_smooth_rois()

        # check if binning the rois
        if self.binning_flag:
            self.bin_rois()

        # check if doing dynamic f0 updates
        if self.dynamic_f0_flag[0]:
            self.dynamic_f0()

        # compute the ensemble activity from ROIs loaded
        self.update_ensembles()

        # check for reward condition:
        self.check_reward_condition()

        # check if > 30 sec has passed since last reward
        self.check_missed_reward_state()

        # decrease reward lockout counter (in case it's on)
        self.reward_lockout_counter[0] -= 1

        # if the counter gets to 0 or less, we turn off the white noise in case it was playing
        if self.reward_lockout_counter[0]<1 :

            # if we were previously in white noise state turn it off
            if self.white_noise_state[0]==1:

                # need to check that not in the dynamic phase lockout period
                print ("WHITE NOISE OFF # ttl: ", self.n_ttl[0], "(post-reward-state: ", self.post_reward_state)
                self.white_noise_state[0] = 0
                self.last_trial_start_ttl = self.n_ttl[0]  # This is a dummy reset so that dynamics lockout doesn't break so badly

            # if we were in a post-rewards state turn it off
            if self.post_reward_state[0]==1:

                #
                print ("POST REWARD STATE OFF # ttl: ", self.n_ttl[0], "(white_noise_state: ", self.white_noise_state)
                self.post_reward_state[0] = 0
                self.last_trial_start_ttl = self.n_ttl[0]  # This is a dummy reset so that dynamics lockout doesn't break so badly

        # save meta data
        self.ttl_n_computed.append(int(self.ttl_computed))
        self.ttl_n_detected.append(int(self.n_ttl[0]))

        #
        self.n_ttl+=1
    #

    def compute_dff_and_smooth_rois(self):

        ''' Function that smooths the raw roi traces;
            - this is required for both visualization but also ensemble computations as
              we do not run algorithms on noisy raw data directly

        '''

        # if we made threhsods using smoothing, then need to run them on data also
        # TODO:  IMPORTANT: implement the identical algorithm used in the calibration step to compute
        #        this step; currently only the smoothing step is shared; need to share DFF0 computation also
        # wait a few seconds until get enough data to smooth out
        if self.smooth_diff_function_flag and self.n_ttl[0]>self.rois_smooth_window:

            # Ensemble 1
            for p in range(len(self.rois_traces_raw_ensemble1)):
                #
                roi_history = self.rois_traces_raw_ensemble1[p,self.n_ttl[0]-self.rois_smooth_window:self.n_ttl[0]]

                #
                rois_dff = (roi_history - self.roi_f0s_ensemble1[p])/self.roi_f0s_ensemble1[p]

                # smooth data only if not in binning mode
                if self.binning_flag==False:
                    self.rois_traces_smooth_ensemble1[p,self.n_ttl[0]] = smooth_ca_time_series4(rois_dff)
                # if binning then just copy last value until it's overwritten by binning function
                else:
                    self.rois_traces_smooth_ensemble1[p, self.n_ttl[0]] = self.rois_traces_smooth_ensemble1[p, self.n_ttl[0]-1]

            # Ensemble 2
            for p in range(len(self.rois_traces_raw_ensemble2)):
                #
                roi_history = self.rois_traces_raw_ensemble2[p,self.n_ttl[0]-self.rois_smooth_window:self.n_ttl[0]]

                #
                rois_dff = (roi_history - self.roi_f0s_ensemble2[p])/self.roi_f0s_ensemble2[p]

                #
                if self.binning_flag==False:
                    self.rois_traces_smooth_ensemble2[p,self.n_ttl[0]] = smooth_ca_time_series4(rois_dff)
                else:
                    self.rois_traces_smooth_ensemble2[p, self.n_ttl[0]] = self.rois_traces_smooth_ensemble2[p, self.n_ttl[0]-1]

        else:
            #
            for p in range(len(self.rois_traces_raw_ensemble1)):
                self.rois_traces_smooth_ensemble1[p,self.n_ttl[0]] = self.rois_traces_raw_ensemble1[p,self.n_ttl[0]]

            #
            for p in range(len(self.rois_traces_raw_ensemble2)):
                self.rois_traces_smooth_ensemble2[p, self.n_ttl[0]] = self.rois_traces_raw_ensemble2[p, self.n_ttl[0]]


    #
    def compute_frame_number(self):
        
        ''' Function that computes which frame to read from [Ca] file based on how many TTL pulses
            were generated via passage of time (using start time, current time and sample rate.  The 
            goal is to figure out which imaging frame we should load next from the memmory map of 
            the [Ca] data
            - alternative is to just count TTL pulses and index into those (this variable is already
            available in this lise: self.ttl_n_computed)
            - however, just counting this list may be incorrect due to possible operating system lockoups
            or kernel issues;
            - TODO: still should test which if these methods gives more accurate location in the imaging
            stack
            
            NOTE #1: 
            In simulation mode, the TTL pulses computed will be incorrect because we will 
            be reading the TTL pulses from a file and this will be superfast compared to waiting for
            them to be generated by the 2P system; 
            - so for simulation mode we have to bypass the "computed ttl" method and just assume that 
            everytime we enter into this function we are ok to read the next frame
            
            NOTE #2: 
            Function also creates a memory map file which is used for storying the imaging frames 
            as they are read from the hard disk. This method is great to limit memory use, but as we
            read more frames from the disk, they do get stored in memory up to and until the entire 
            file is loaded into memory
            - beause some the of the imaging datasets can be 30GB or more we might run out of 
            memory, unless we have 128GB or much more ram
            - TODO: we will implement a method that destroys/deletes the memory map and starts over 
            perhaps every 10 minutes or so.  Restarting the memmap seems to take Order (1ms) so it
            should not be an issue, but we do have to test it to ensure we are freeing up memory
            
            Input: 
            - self.ttl_n_computed = contains the number of ttl pulses computed based on passage of time
            - self.now = contains the realtime of the last read ttl pulse (usually in millsec; to check)
            - self.fname_fluorescence = path of the fluorescence file
            - self.n_frames_to_be_acquired = total number of imaging frames for the session
            - self.sampleRate_2P = samplerate of the 2P microscope, usually 30FPS
            
            Output:
            - self.newfp = this is the memory map that holds all our calcium data
            - self.ttl_computed = the frame location based on passage of time; we use this to reach into
            the memory map [Ca] file and grab a specific frame
        
        '''

        # initialize raw data arrays
        if len(self.ttl_n_computed)==0:

            #
            if self.read_data_flag:
                #import mmap
                #ss = time.time()
                print ("  setting up memory map: shape: ", (self.n_frames_to_be_acquired,512,512))
                
                if True:
                    movie_frame_bytes = self.image_width * self.image_length * np.dtype('uint16').itemsize
                    movie_frame_count, remainder = divmod(
                        os.path.getsize(self.fname_fluorescence), movie_frame_bytes
                    )
                    if remainder or movie_frame_count < 1:
                        raise ValueError("Invalid 512x512 uint16 fluorescence movie size")
                    if not self.simulation_mode_bmi and movie_frame_count < self.n_frames_to_be_acquired:
                        raise ValueError("Fluorescence movie is shorter than the requested session")
                    self.movie_frame_count = movie_frame_count
                    self.newfp = np.memmap(self.fname_fluorescence,
                                           dtype='uint16',
                                           mode='r',
                                           shape=self.movie_frame_count*512*512)
                
                # if False:
                #     fp = open(self.fname_fluorescence, "r")
                #     byts = self.n_frames_to_be_acquired*self.image_width*self.image_length
                #
                #     self.newfp = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                #     # print ("OPTION @@@@@@@@@@@@@@@@: ", self.newfp)
                #
                #     #
                #     mview = memoryview(self.newfp)
                #     print (" memroy view: ", mview)
                #     self.newfp = np.asarray(mview).reshape(self.n_frames_to_be_acquired,512,512)
                #     print (" final array view: ", self.newfp.shape)


                # 
                self.newfp = self.newfp.reshape(self.movie_frame_count,512,512)

            # reset start time: requird becaues we start the BMI a few seconds before the BScope
            self.start=self.now

        # after arrays initialized
        else:

            # in simulation mode we just assume that we have correctly dected a TTL pulse and add 1 extra
            #   ttl pulse to the stack
            if self.simulation_mode_bmi==True:
                self.ttl_computed = int(self.n_ttl[0]) + 1  # move to next ttl.
                if self.simulation_realtime:
                    target_time = self.start + self.n_ttl[0] / self.sampleRate_2P
                    time.sleep(max(0, target_time - time.perf_counter()))
                else:
                    time.sleep(self.sleep_time_sec)
                
            else:
                time_passed = self.now-self.start
                self.ttl_computed = round((self.now-self.start)*self.sampleRate_2P)

                # 
                if self.verbose:
                    print (" time passed: ", time_passed, "   bmi_update self.ttl_computed: ", self.ttl_computed)

    #
    def trigger_reward(self):

        #
        print("\n ")

        if self.contingency_degradation[0]:
            print (">>> GIVING RANDOM REWARD <<<")
        else:
            print(" >>>>> reached high reward condition: ")
            print(" ensemble state: ", self.ensemble_state)
            print(" high_threshold: ", self.high_threshold)

        # search for the first empty slot in the reward times list
        # TODO: this is a poor way to save these values;
        #   TODO: have a shared memory variable separate from one that keeps track of the times
        for k in range(self.reward_times.shape[1]):
            if self.reward_times[1, k] == -1:
                self.reward_times[1, k] = self.n_ttl[0]  # save current reward time
                break
        #
        self.rewarded_times_abs.append([1, self.n_ttl[0], self.abs_times_ttl_read[-1], self.abs_times_ca_read[-1]])

        # reset last reward time to current time
        self.last_reward_ttl[0] = self.n_ttl[0]

        # generate water reward only if we are not in a reward lockout state
        print (" giving reward at time: ", self.n_ttl)

        #
        self.water_reward[0] = 1

    #
    def check_missed_reward_state(self):

        # if mouse does not perform in e.g. 30 sec, do a lockout
        # TODO: May wish to play white noise to distinguish it from post-reward state
		# so here we can't evaluate or reset the white noise as long as we're in one of these 3 states
		#  white noise is still playing, i.e. post miss state
		#  post-reward lockout of about 3 seconds
		#  or we still haven't returned the neural activity to below required baseline
        if self.white_noise_state[0]==1 or self.post_reward_state[0]==1 or self.dynamic_reward_lockout_state[0]==1:
            return

        # here we want to check not against last reward, but against the last trial commencement
        if (self.n_ttl[0]-self.last_trial_start_ttl)>(self.max_reward_window*self.sampleRate_2P):

            ## start reward downward counter
            self.reward_lockout_counter[0] = self.missed_reward_lockout * self.sampleRate_2P

            # also set a dummy value to last trial start;
            # TODO: check if this is correct
            self.last_trial_start_ttl = self.n_ttl[0]

            #
            print ("\n")
            print (">>>> reached end of trial without reward")
            print ("WHITE NOISE ON # ttl: ", self.n_ttl[0], "post_reward_state: ", self.post_reward_state, "white noise state: ", self.white_noise_state)
            self.white_noise_state[0]=1

            # close missed trial
            print ("ENDING TRIAL @ # TTL: ", self.n_ttl)
            self.trials[self.trial_number,2] = self.n_ttl[0]
            self.trials[self.trial_number,3] = time.time()
            self.trials[self.trial_number,4] = 0

            # also reset dynamic reward lockout to blcok rewawards
            self.dynamic_reward_lockout_state[0] = 1
            print ("DYNAMIC STATE LOCKOUT ON (post-trial timeout): ", self.n_ttl[0])

    #
    def check_baseline_condition(self):

        # check if ensemble activity back to baseline; e.g. within 1 x of std
        # if abs(ensemble...[ensbmel_ID1] - ensbmel..[ensemble_ID2])< std x 1?
        #     return True
        # else:
        #     return False

        pass

    #
    def check_reward_condition(self):

        ''' We check if reward condition was reached

        '''

        # cant' give reward during white noise lockout or any other lockouts
        # TODO: check if it matters to do the dynamic reward lockout check always (so it can turn off anytime)
        if self.white_noise_state[0] == 1 or self.post_reward_state[0] ==1:
            return

        # CD case
        if self.contingency_degradation[0]:

            # draw random probability:
            #threshold = 0.00027778
            threshold = 0.00025
            x = np.random.rand()
            #print (self.n_ttl, "Rnadom draw: ", x)
            if x <= threshold:

                # reward mouse
                self.trigger_reward()

                # close reward trial
                print("RANDOM REWARD GIVEN @ # TTL: ", self.n_ttl)
                self.trials[self.trial_number, 2] = self.n_ttl[0]
                self.trials[self.trial_number, 3] = time.time()
                self.trials[self.trial_number, 4] = 1

                # enter post reward lockout
                self.post_reward_state[0] = 1

                # reset the lockout counter for rewards
                self.reward_lockout_counter[0] = self.received_reward_lockout * self.sampleRate_2P

                # here we don't do dynamic lockouts only the static ones
                self.dynamic_reward_lockout_state[0] = 0
                print("DYNAMIC LOCKOUT ON (post-reward) # ttl: ", self.n_ttl[0])

            return


        # if we are in a lockout period; mouse not eligible for reward
        if self.dynamic_reward_lockout_state[0]==1:

            # check to see if ensembles dropped below threshold:
            if self.ensemble_state[0] <= (self.high_threshold[0]*self.dynamic_reward_lockout_threshold):

                # reset the state back so mouse can get rewards again
                print ("DYNAMIC STATE LOCKOUT OFF # ttl: ", self.n_ttl[0])
                self.dynamic_reward_lockout_state[0] = 0  # we can now receive rewards

                # we also set the last time a reward was had
                self.last_trial_start_ttl = self.n_ttl[0]
                print ("STARTING NEW TRIAL @ # TTL: ", self.last_trial_start_ttl)

                #
                self.trial_number+=1
                self.trials[self.trial_number,0] = self.n_ttl[0]
                self.trials[self.trial_number,1] = time.time()

        # mouse can get rewards if ensemble over threshold AND the static counter has counted down
        # TODO: no need to check reward lockout counter any more....
        elif (self.ensemble_state[0] >= self.high_threshold[0]):

            # reward mouse
            self.trigger_reward()

            # close reward trial
            print ("ENDING TRIAL @ # TTL: ", self.n_ttl)
            self.trials[self.trial_number,2] = self.n_ttl[0]
            self.trials[self.trial_number,3] = time.time()
            self.trials[self.trial_number,4] = 1

            # enter post reward lockout
            self.post_reward_state[0] = 1

            # reset the lockout counter for rewards
            self.reward_lockout_counter[0] = self.received_reward_lockout * self.sampleRate_2P

            #
            self.dynamic_reward_lockout_state[0]=1
            print ("DYNAMIC LOCKOUT ON (post-reward) # ttl: ", self.n_ttl[0])

    #
    def load_current_frame_and_apply_drift_correction(self):

        ''' This is an overkill function which cheks whether the n_ttl detected from TTL pulses
            doe sindeed have values in it, if so it

            TODO: either drop this first check - or implement it in full - which means going forward in time
                  until we get zeros in the data

        '''

        #
        if self.verbose:
            print ("self.ttl_computed: ", self.ttl_computed)
            print("  detected frame #: ", self.n_ttl,
               " computed_frame : ", self.ttl_computed)

        # Before updating ROIS - must find the correct frame number and load the frame
        # for the first ROI: we loop over the data from -1 frames back to up to n_frames_search_forward in the future
        #  - we are looking for the last frame that has data in it;
        #    we then exit and keep the counter in memroy

        # IMPORTANT #1
        # TODO: 2 options for computing activity in a cell ROI: sum vs. mean (there are others 
        # IMPORTANT #2
        # TODO: this algorithm essentially uses empirical data to check how far our imaging system has gone
        # - it is probably the best way to ensure that we are up todate with real time (at least realtime with the 2p + writing times
        # - more to think about whether this can go wrong
        # - but for now, this next loop is quasi-guarantee that we are in real time

        # search the very first ROI in time from previous frame to future frames until we get a non-zero pixel values;
        #  then we set the time i.e. n_ttl
        for z in range(-1,self.n_frames_search_forward,1):

            # check
            #roi_sum0 = self.newfp[self.n_ttl[0]+z,
            #                      self.rois[0][0]-self.roi_width:self.rois[0][0]+self.roi_width,
            #                      self.rois[0][1]-self.roi_width:self.rois[0][1]+self.roi_width].sum()

            # TODO ;could just check any part of the FOV to see if there is non zero values
            frame_index = self.n_ttl[0] + z
            if self.simulation_mode_bmi:
                frame_index %= self.movie_frame_count
            roi_sum0 = self.newfp[frame_index][self.rois_pixels_ensemble1[0]].sum()

            #
            if roi_sum0 != 0:

                # TODO: reset the n_ttl value here - check that this is safe!!!
                # self.n_ttl[0] = self.n_ttl[0]+z

                break

        # TODO: we should reset the n_ttl here
        # - if we find that we needed to search x steps forward,
        #   we should then add x to n_ttl - and vice versa

        # TODO: update latest image for imaging purposese
        # this raw frame is fed to the drift correction algorithm (anywhere else!?)

        # this is the same raw frame but now it is fixed for purpose of computing ROIs!!
        #  - this is the latest frame extracted
        # NOTE: outside functions do not see it unless explicitly copied
        frame_index = self.n_ttl[0] + z
        if self.simulation_mode_bmi:
            frame_index %= self.movie_frame_count
        self.live_frame_local = self.newfp[frame_index].copy()

        # # # simulate drift....
        # if False:
        #     simulated_shift = int(self.n_ttl[0]/300)
        #     self.live_frame_local = np.roll(self.live_frame_local,
        #                                     simulated_shift, axis=0)
        #     print ("Simulated drift -------> ", simulated_shift)

        # motion detector gets this frame; and returns drift_xy_values
        self.live_frame_motion_detector[0] = self.live_frame_local.copy()
        
        #
        self.abs_times_ca_read.append(time.perf_counter()) #perf_counter_ns()/1E9

        #
        if self.motion_correction_flag[0]:

            # save most recent drift values from drift module
            self.drift_array.append([self.drift_xy_values[0],
                                     self.drift_xy_values[1]])

            # NOTE: the drift_xy values could be the previously saved ones
            self.live_frame_local_drift_corrected = apply_shifts(self.live_frame_local.copy(),
                                                                 self.drift_xy_values[0],
                                                                 self.drift_xy_values[1])

        else:
            self.live_frame_local_drift_corrected = self.live_frame_local.copy()

        # manual drift correction is always online;
        # TODO: Check to see how slow this roll is... shouldn't be, but in case!
        if True:
            #print ("motion correcting: ", self.manual_motion_correction_array)
            self.live_frame_local_drift_corrected = np.roll(self.live_frame_local_drift_corrected,
                                                            self.manual_motion_correction_array[0],
                                                            axis=1)
            self.live_frame_local_drift_corrected = np.roll(self.live_frame_local_drift_corrected,
                                                            -self.manual_motion_correction_array[1],
                                                            axis=0)

        # this is the frame that the plotting function sees
        self.live_frame_plotter[0] = self.live_frame_local_drift_corrected.copy()

    #
    def update_rois(self):

        # update ROIs ensemble 1
        for p in range(0,len(self.rois_pixels_ensemble1),1):

            # new way use exact pixel location
            temp = self.live_frame_local_drift_corrected[self.rois_pixels_ensemble1[p].T[:, 0],        # broadcast/index into the frame as per ROI pixels
                                                         self.rois_pixels_ensemble1[p].T[:, 1]]

            # divide by the number of pixels in the ROI - NOT SURE IF THIS IS CORRECT?!
            # TODO: these algorithms must match the default water disposal algorithms
            # TODO: USE A FUNCTION OVER THIS AND FOLLOWING STEP THAT IS SHARED WITH CALIBRATION CODE
            roi_sum0 = temp / self.rois_pixels_ensemble1[p][0].shape[0]

            # sum
            # TODO: not sure this is the correct function; to check literature
            # TODO: also this part shoudl be refactored to a callabale function by both calibration and BMI classes
            roi_sum0 = np.nansum(roi_sum0)

            # Note: Do not remove baseline yet; this is done in the smoothing step;
            # TODO: make sure that this approach is correct
            self.rois_traces_raw_ensemble1[p,self.n_ttl[0]] = roi_sum0

        # update ROIS ensemble 2
        for p in range(0,len(self.rois_pixels_ensemble2),1):

            # new way use exact pixel location
            temp = self.live_frame_local_drift_corrected[self.rois_pixels_ensemble2[p].T[:, 0],        # broadcast/index into the frame as per ROI pixels
                                                         self.rois_pixels_ensemble2[p].T[:, 1]]

            # divide by the number of pixels in the ROI - NOT SURE IF THIS IS CORRECT?!
            # TODO: these algorithms must match the default water disposal algorithms
            # TODO: USE A FUNCTION OVER THIS AND FOLLOWING STEP THAT IS SHARED WITH CALIBRATION CODE
            roi_sum0 = temp / self.rois_pixels_ensemble2[p][0].shape[0]

            # sum
            # TODO: not sure this is the correct function; to check literature
            # TODO: also this part shoudl be refactored to a callabale function by both calibration and BMI classes
            roi_sum0 = np.nansum(roi_sum0)

            # Note: Do not remove baseline yet; this is done in the smoothing step;
            # TODO: make sure that this approach is correct
            self.rois_traces_raw_ensemble2[p,self.n_ttl[0]] = roi_sum0

        #
        if self.verbose:
            print ("")
            print ("")

    #
    def update_ensembles(self):

        # wait for at least some frames to go by first
        if self.n_ttl[0]>self.rois_smooth_window:

            # compute ensemble 1
            for k in range(len(self.rois_traces_smooth_ensemble1)):
                self.ensemble_activity[0, self.n_ttl[0]]+= self.rois_traces_smooth_ensemble1[k, self.n_ttl[0]]

            # compute ensemble 1
            for k in range(len(self.rois_traces_smooth_ensemble2)):
                self.ensemble_activity[1, self.n_ttl[0]]+= self.rois_traces_smooth_ensemble2[k, self.n_ttl[0]]

        # Compute the E1-E2 for current time point
        # this value goes to the tone package which converts it into a tone
        # TODO: this value is sometimes zero, not clear why, perhaps we are readng too far ahead

        # use the same diff funtion as in the calibration set
        self.ensemble_state[0] = (self.ensemble_activity[0, self.n_ttl[0]] -
                                  self.ensemble_activity[1, self.n_ttl[0]])

        #
        self.ensemble_diff_array[self.n_ttl[0]] = self.ensemble_state[0]

    #
    def tone_off(self):

        # turn toneplayback off
        #  freq = 0
        #  np.save(self.fname_freq,freq)

        # need to also pass the time out counter
        #
        # pass a zero neural state vector??!?!


        pass

    #
    def save_data(self):

        '''  TO FILL OUT
             need better description of variables
             TODO:  other variables we might want to save including
             - tone frequencies, or tone state of the speaker
             - the camera frames/informatin
             - IR light info
             - EMG data
             - lick detector information
             - treadmill/ball walking distance

        '''

        print("...Saving BMI meta/data...")

        #
        if self.align_flag==True:
            print ("  alignment session... skipping data save")
            return

        # find # of rewards
        for k in range(self.reward_times.shape[1]):
            if self.reward_times[1,k]==-1:
                break

        #
        print (" ----> # OF REWARDS: ", k, ", water volume dispensed (@ 10uL per reward): ",k*0.010, "mL")

        #
        #df = pd.DataFrame(data=self.bmi_dictionary, index=[0])
        df = pd.DataFrame.from_dict(self.bmi_dictionary)

        df.to_excel(self.fname_save_data[:-4]+'.xlsx')

        #
        np.savez(self.fname_save_data,
                 ttl_voltages = self.ttl_voltages,
                 ttl_n_computed = self.ttl_n_computed,
                 ttl_n_detected = self.ttl_n_detected,
                 abs_times_ttl_read = self.abs_times_ttl_read,
                 abs_times_ca_read = self.abs_times_ca_read,
                 ttl_times = self.ttl_times,
                 rois_pixels_ensemble1 = np.hstack(self.rois_pixels_ensemble1),
                 rois_pixels_ensemble2 = np.hstack(self.rois_pixels_ensemble2),
                 rois_traces_raw_ensemble1 = np.array(self.rois_traces_raw_ensemble1,dtype='object'),
                 rois_traces_raw_ensemble2 = np.array(self.rois_traces_raw_ensemble2,dtype='object'),
                 rois_traces_smooth1 = np.array(self.rois_traces_smooth_ensemble1,dtype='object'),
                 rois_traces_smooth2 = np.array(self.rois_traces_smooth_ensemble2,dtype='object'),
                 reward_times = self.reward_times,
                 rewarded_times_abs = np.array(self.rewarded_times_abs,dtype='object'),
                 ensemble_activity = self.ensemble_activity,
                 ensemble_diff_array = self.ensemble_diff_array,
                 received_reward_lockout = self.received_reward_lockout,
                 max_reward_window = self.max_reward_window,
                 missed_reward_lockout = self.missed_reward_lockout,
                 trials = self.trials,

                #
                 high_threshold = self.high_threshold[0],

                 #
                 sampleRate_NI = self.sampleRate_NI,
                 ttl_pts = self.ttl_pts,
                 sampleRate_2P = self.sampleRate_2P,
                 image_width = self.image_width,
                 image_length = self.image_length ,
                 max_n_seconds_session = self.max_n_seconds_session,

                #
                 n_frames = self.n_frames,
                 n_frames_to_be_acquired = self.n_frames_to_be_acquired,                #
                 rois_smooth_window = self.rois_smooth_window,
                 n_ttl_to_start_applying_dynamic_f0 = self.n_ttl_to_start_applying_dynamic_f0,
                 n_frames_search_forward = self.n_frames_search_forward,
                 drift_array = self.drift_array,
                 #template = self.template,  # no need to save this; at least not now
                 lick_detector_abstime = self.lick_detector_abstime,
                 rotary_encoder1_abstime = self.rotary_encoder1_abstime,
                 rotary_encoder2_abstime = self.rotary_encoder2_abstime,
                 )
