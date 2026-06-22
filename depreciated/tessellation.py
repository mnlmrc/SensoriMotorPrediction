class Tessellation():
    def __init__(self, path=None, participants_id=None, n_tessels=None, n_jobs=None):
        self.path = path
        self.participants_id = participants_id
        self.n_tessels = n_tessels
        self.n_jobs = n_jobs

        # define atlas
        self.atlas, _ = am.get_atlas('fs32k')

        # define structures
        self.struct = ['CortexLeft', 'CortexRight']
