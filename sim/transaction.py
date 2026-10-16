import itertools


class Transaction:
    """A single write or read on the ledger.

    A write is a new block going onto some node. A read looks up an existing block via ref_block_id, and if that block lives on a different shard it picks up a cross-shard latency cost (Eq. 5, in Simulator).
    """

    _id_counter = itertools.count(1)

    id: int
    payload_size: int
    jurisdiction: bytes
    is_read: bool
    ref_block_id: int | None
    created_tick: int
    origin_node_id: int | None
    gas: int | None
    source_tx_hash: str | None
    allowed_jurisdictions: list[int]

    def __init__(
        self,
        payload_size: int,
        jurisdiction: bytes,
        is_read: bool = False,
        ref_block_id: int | None = None,
        created_tick: int = 0,
        origin_node_id: int | None = None,
        gas: int | None = None,
        source_tx_hash: str | None = None,
        allowed_jurisdictions: list[int] | None = None,
    ):
        self.id = next(Transaction._id_counter)
        self.payload_size = payload_size
        self.jurisdiction = jurisdiction
        self.is_read = is_read
        self.ref_block_id = ref_block_id
        self.created_tick = created_tick
        self.origin_node_id = origin_node_id
        # gas and source_tx_hash come from the real Sepolia trace (see data_sources.py).
        # They're metadata only; the resource model (Section 3.3.3) runs off payload_size.
        self.gas = gas
        self.source_tx_hash = source_tx_hash

        # Mmeta (Section 3.2): the jurisdictions this record is legally allowed to live in.
        # Not the same as `jurisdiction`, which is just where it came from. Defaults to its
        # own jurisdiction only; pass a list to allow more (e.g. an adequacy-decision one).
        self.allowed_jurisdictions = (
            allowed_jurisdictions
            if allowed_jurisdictions is not None
            else [int(jurisdiction.decode())]
        )

        if is_read and ref_block_id is None:
            raise ValueError("Read transactions must reference a ref_block_id.")
