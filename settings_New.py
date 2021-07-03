 #
# Settings for SEGAN
#

class settings:

    def __init__(self):

        #   Precision Mode
        self.halfprec = True                        # 16bit or not

        #   Image settings
        #self.size               = (64,64)           # Input Size (64 by 64)
        #self.size = (128, 128)                           # Input Size (128 by 128)
        #self.size = (256, 256)                         #Input Size (256 by 256)
        #   Training
        self.batch_size = 20                       # Batch size
        self.batch_size_test = 1
        self.epoch      = 200 # Epoch
        self.learning_rate = 0.0001                 # Learning Rate　　

        # Retrain
        self.retrain    = 0

        # Save path
        self.model_save_path    = 'params'          # Network model path
        self.model_save_path2 = 'params2'  # Network model path
        self.model_save_cycle   = 100               # Epoch cycle for saving model (init:1)
        self.result_save_path   = 'result'          # Network model path

        # Data path
        self.train_data_path    = './resize/eximages_80'     # Folder containing training image (train)
        self.train_mask_path     = './resize/exreference_80'
        self.train_data_num     = 4536

        self.test_data_path = './resize/eximages_20'
        self.test_mask_path = './resize/exreference_20'
        self.test_data_num = 1132

        self.eval_data_path = './resize/Girl/256x256'
        self.eval_mask_path = './resize/Girl/256x256_mask'
        self.eval_data_num = 30

        self.eval_data_path = './resize/paprica_256x256'
        self.eval_mask_path = './resize/paprica_256x256'
        self.eval_data_num = 30


        # Pkl files
        self.pkl_path     = './pkl'             # Folder of pkl files for train

