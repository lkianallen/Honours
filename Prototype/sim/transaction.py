class Transaction:
    payload_size: int
    jurisdiction: bytes
    is_read: bool

    def __init__(self, payload_size: int, jurisdiction: bytes):
        self.payload_size = payload_size
        self.jurisdiction = jurisdiction
        self.is_read = False