import pyarrow.plasma as plasma
import subprocess
import tempfile


class PlasmaArray(object):
    """
    Wrapper around numpy arrays that automatically moves the data to shared
    memory upon serialization. This is particularly helpful when passing numpy
    arrays through multiprocessing, so that data is not unnecessarily
    duplicated or pickled.
    """

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.object_id = None
        self.path = None

        # variables with underscores shouldn't be pickled
        self._client = None
        self._server = None
        self._server_tmp = None

    def start_server(self):
        assert self._server is None
        assert self.object_id is None
        assert self.path is None
        self._server_tmp = tempfile.NamedTemporaryFile()
        self.path = self._server_tmp.name
        self._server = subprocess.Popen([
            'plasma_store',
            '-m', str(int(1.05 * self.array.nbytes)),
            '-s', self.path,
        ])

    @property
    def client(self):
        if self._client is None:
            assert self.path is not None
            self._client = plasma.connect(self.path)
        return self._client

    def __getstate__(self):
        if self.object_id is None:
            self.start_server()
            self.object_id = self.client.put(self.array)
        state = self.__dict__.copy()
        del state['array']
        state['_client'] = None
        state['_server'] = None
        state['_server_tmp'] = None
        state['_plasma'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.array = self.client.get(self.object_id)

    def __del__(self):
        if self._server is not None:
            self._server.kill()
            self._server = None
            self._server_tmp.close()
            self._server_tmp = None

    def move_to_plasma(self):
        self.start_server()
        self.object_id = self.client.put(self.array)
        self.array = self.client.get(self.object_id)
