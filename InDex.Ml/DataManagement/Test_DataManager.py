from DataManagement.data_manager import DataManager
import pandas as pd

dr = DataManager()
dr.set_data(pd.read_csv('../Notes/CarPrice_Assignment.csv'))
print(dr.print_columns())

dr2 = DataManager()
dr2.set_data(pd.read_csv('../Notes/Ames_Housing_Data.tsv'))
print(dr2.get_instance().print_columns())