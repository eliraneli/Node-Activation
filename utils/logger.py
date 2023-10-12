import logging


class Logger(logging.Logger):

    def __init__(self, name, file_path):
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # print the logs to the streamer
        stream = logging.StreamHandler()
        stream.setLevel(logging.DEBUG)
        stream.setFormatter(formatter)

        # write the logs to the files
        file = logging.FileHandler(file_path, mode='a+')
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)

        self.addHandler(stream)
        self.addHandler(file)