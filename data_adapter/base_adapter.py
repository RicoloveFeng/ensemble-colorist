import pandas as pd

class BaseAdapter:
    def convert(self, src_path: str) -> pd.DataFrame:
        """
        Read data from src_path and return a dataframe
        """
        raise NotImplementedError
    
    def convert_and_output(self, src_path: str, dst_path: str):
        df = self.convert(src_path)
        df.to_csv(dst_path, header=True, index=False, sep=',')
        