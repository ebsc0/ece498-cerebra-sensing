from preprocessor import Preprocessor

class MultiOptodePreprocessor:
    def __init__(self, optode_ids):
        """
        Args:
            optode_ids: iterable of optode identifiers
        """
        self.processors = {
            optode_id: Preprocessor()
            for optode_id in optode_ids
        }

    def process_sample(self, frame: dict):
        """
        Args:
            frame = {
                "optode_id": int or str,
                "long_860": float,
                "long_740": float,
                "short_860": float,
                "short_740": float,
                "dark": float
            }

        Returns:
            processed result dict OR None (until baseline ready)
        """

        optode_id = frame["optode_id"]

        if optode_id not in self.processors:
            raise ValueError(f"Unknown optode_id: {optode_id}")

        return self.processors[optode_id].process_sample(frame)