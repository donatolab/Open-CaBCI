import numpy as np

#################################################
############# SIMULATION CLASS ##################
#################################################
class Simulation():
    ''' This class simulates the TTL pulses coming out of the Bscope
        by reading a ttl pulse file and returning values as requested
    '''

    def __init__(self,
                 fname_ttl):
        # set location of reading index to 0 at beginning
        self.index = 0

        # read ttl pulses from file
        self.ttl = np.load(fname_ttl)

    def read(self, number_of_samples_per_channel):
        if self.index >= len(self.ttl):
            self.index = 0

        # added a 2nd value for the lick detector
        ttl_val_bscope = self.ttl[self.index:self.index + number_of_samples_per_channel]
        ttl_val_lick_detector = 0
        ttl_val_rotary_encoder_1 = 0
        ttl_val_rotary_encoder_2 = 0

        self.index += number_of_samples_per_channel

        out = np.hstack((ttl_val_bscope[0],
                         ttl_val_lick_detector,
                         ttl_val_rotary_encoder_1,
                         ttl_val_rotary_encoder_2))

        #
        return out

    def stop(self):
        # not required in simulation mode
        pass

    def close(self):
        # not required in simulation mode
        pass
