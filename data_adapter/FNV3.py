import pandas as pd
import sys
import os

# 添加父目录到Python路径，以便导入base_adapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_adapter.base_adapter import BaseAdapter

class FNV3Adapter(BaseAdapter):
    def convert(self, src_path: str) -> pd.DataFrame:
        """
        读取FNV3.csv格式的数据，提取指定字段并重命名
        
        Args:
            src_path: CSV文件路径
            
        Returns:
            处理后的DataFrame，包含track,sample,hours,lat,lon,pressure字段
        """
        # 读取CSV文件
        df = pd.read_csv(src_path)
        
        # 提取需要的字段并重命名
        result_df = df[['track_id', 'sample', 'lead_time_hours', 'lat', 'lon', 'minimum_sea_level_pressure_hpa']].copy()
        
        # 重命名列
        result_df = result_df.rename(columns={
            'track_id': 'track',
            'lead_time_hours': 'hours',
            'minimum_sea_level_pressure_hpa': 'pressure'
        })
        
        return result_df
    
if __name__ == "__main__":
    adapter = FNV3Adapter()
    # 使用相对于当前文件的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "raw_data", "FNV3.csv")
    df = adapter.convert(csv_path)
    print(df.head())
    adapter.convert_and_output(csv_path, "FNV3_example.csv")