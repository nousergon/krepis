"""Writer identities shared by the S3 telemetry sinks."""

import uuid


def new_writer_id() -> str:
    """Return a per-instance identity, independent of the run identifier."""
    return uuid.uuid4().hex[:8]
